"""Gemma 3 12B-it + LoRA fall checker on Modal (serverless GPU). The app calls it only when VideoMAE raises an alarm.

Same scoring as the offline experiment (verify_alarms.py): 32 JPEG frames in time order, answer forced to start with
{"verdict": "  , P(fall) from the next-token probabilities (gemma_binary_scoring.py), then the answer text is generated
for the "evidence" sentence.

One-time setup (`pip install modal`, `modal token new`, and a Modal secret "huggingface-secret" holding HF_TOKEN for an
account that has accepted the Gemma terms):
    modal run modal_app.py::download            # Gemma 3 12B-it from Hugging Face into the volume (~24 GB, gated)
    modal run modal_app.py::download_qwen       # Qwen2-VL-7B-Instruct into the volume (~16 GB)
    modal run modal_app.py::download_adapters   # our LoRA adapters from MODEL_REPO (CSC490/fall-models)
    modal deploy modal_app.py                   # creates the "fall-verifier" app (classes Verifier and QwenVerifier)
Call it:  modal.Cls.from_name("fall-verifier", "Verifier")().check.remote([jpeg bytes, ...])   (Gemma)
          modal.Cls.from_name("fall-verifier", "QwenVerifier")().check.remote([jpeg bytes, ...])  (Qwen)
"""
from pathlib import Path

import modal

APP, VOLUME, MNT = "fall-verifier", "fall-models", "/models"
BASE_REPO = "google/gemma-3-12b-it"
QWEN_REPO, MODEL_REPO = "Qwen/Qwen2-VL-7B-Instruct", "CSC490/fall-models"  # MODEL_REPO: our trained adapters
QWEN_BASE, QWEN_ADAPTER = f"{MNT}/qwen2-vl-7b-instruct", f"{MNT}/qwen_adapter"
BASE, ADAPTER = f"{MNT}/gemma-3-12b-it", f"{MNT}/adapter_reasons"  # old adapter (before reason training): f"{MNT}/adapter"
HERE = Path(__file__).parent

packages = (modal.Image.debian_slim(python_version="3.11")
            .pip_install("torch", "transformers==5.19.0", "peft", "accelerate", "pillow", "huggingface_hub"))


def with_code(img):
    """Our scoring code, added when a container starts (no image rebuild when it changes)."""
    for f in ("gemma_scoring.py", "gemma_binary_scoring.py", "qwen_scoring.py"):
        img = img.add_local_file(HERE / f, f"/root/{f}")
    return img


image = with_code(packages)  # Gemma: the same packages as before
qwen_image = with_code(packages.pip_install("torchvision"))  # Qwen2-VL's processor also needs torchvision
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
app = modal.App(APP, image=image)


@app.function(volumes={MNT: volume}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=3600)
def download():
    import os

    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    snapshot_download(BASE_REPO, local_dir=BASE, token=token,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.jinja"])
    volume.commit()
    print("downloaded", sorted(p.name for p in Path(BASE).iterdir()))


@app.function(volumes={MNT: volume}, timeout=3600)
def download_qwen():
    """Qwen2-VL-7B-Instruct (public, ~16 GB) from Hugging Face into the volume."""
    from huggingface_hub import snapshot_download
    snapshot_download(QWEN_REPO, local_dir=QWEN_BASE,
                      allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.jinja"])
    volume.commit()
    print("downloaded", sorted(p.name for p in Path(QWEN_BASE).iterdir()))


@app.function(volumes={MNT: volume}, secrets=[modal.Secret.from_name("huggingface-secret")], timeout=1800)
def download_adapters():
    """Our LoRA adapters from MODEL_REPO into the volume: gemma/adapter_reasons -> ADAPTER, qwen/r16_all3 -> QWEN_ADAPTER."""
    import os
    import shutil
    import tempfile

    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(MODEL_REPO, local_dir=tmp, token=token,
                          allow_patterns=["gemma/adapter_reasons/*", "qwen/r16_all3/*"])
        for src, dst in ((f"{tmp}/gemma/adapter_reasons", ADAPTER), (f"{tmp}/qwen/r16_all3", QWEN_ADAPTER)):
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(dst, sorted(p.name for p in Path(dst).iterdir()))
    volume.commit()


@app.cls(gpu="L40S", volumes={MNT: volume}, timeout=600, scaledown_window=300, max_containers=1)
class Verifier:
    @modal.enter()
    def load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        import gemma_binary_scoring as gs
        self.gs = gs
        self.processor = AutoProcessor.from_pretrained(BASE)
        self.pre, self.plan, bad = gs.label_plan(self.processor.tokenizer)
        assert not bad, bad
        model = Gemma3ForConditionalGeneration.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
        self.model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()

    @modal.method()
    def check(self, jpegs):
        """jpegs: list of JPEG bytes in time order (32 in the experiment) -> {p_fall, text, seconds}."""
        import io
        import time

        import torch
        from PIL import Image
        t, gs = time.perf_counter(), self.gs
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs]
        content = [{"type": "text", "text": gs.instruction(len(frames))}]
        for i, im in enumerate(frames):
            tag = "earliest" if i == 0 else "latest" if i == len(frames) - 1 else "later"
            content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
        inp = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt"
                                                 ).to(self.model.device, dtype=torch.bfloat16)
        probs, mass, _ = gs.to_probs(gs.score_labels(self.model, inp, self.pre, self.plan))
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=100, do_sample=False)
        text = self.processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        return {"p_fall": float(probs["fall"]), "mass": float(mass), "text": text,
                "seconds": time.perf_counter() - t}


@app.cls(gpu="L40S", image=qwen_image, volumes={MNT: volume}, timeout=600, scaledown_window=300, max_containers=1)
class QwenVerifier:
    """Qwen2-VL-7B-Instruct + Solomon's LoRA: same prompt, forced prefix and P(fall) as Verifier (qwen_scoring.py)."""

    @modal.enter()
    def load(self):
        from qwen_scoring import QwenScorer
        self.qs = QwenScorer(QWEN_BASE, QWEN_ADAPTER).load()

    @modal.method()
    def check(self, jpegs):
        """jpegs: list of JPEG bytes in time order (32) -> {p_fall, mass, text, seconds}."""
        import io
        import time

        from PIL import Image
        t = time.perf_counter()
        r = self.qs.check([Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs])
        return {**r, "seconds": time.perf_counter() - t}
