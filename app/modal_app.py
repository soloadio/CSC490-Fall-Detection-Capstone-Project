"""Gemma 3 12B-it + LoRA fall checker on Modal (serverless GPU). The app calls it only when VideoMAE raises an alarm.

Same scoring as the offline experiment (verify_alarms.py): 32 JPEG frames in time order, answer forced to start with
{"verdict": "  , P(fall) from the next-token probabilities (gemma_binary_scoring.py), then the answer text is generated
for the "evidence" sentence.

One-time setup (from the cluster, with `modal token new` done):
    modal run modal_app.py::download                        # base model from Hugging Face into the volume (~24 GB)
    modal volume put fall-models <adapter folder> /adapter  # the LoRA adapter (250 MB)
    modal deploy modal_app.py                               # creates the "fall-verifier" app
Call it:  modal.Cls.from_name("fall-verifier", "Verifier")().check.remote([jpeg bytes, ...])
"""
from pathlib import Path

import modal

APP, VOLUME, MNT = "fall-verifier", "fall-models", "/models"
BASE_REPO = "google/gemma-3-12b-it"
BASE, ADAPTER = f"{MNT}/gemma-3-12b-it", f"{MNT}/adapter"
HERE = Path(__file__).parent

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch", "transformers==5.19.0", "peft", "accelerate", "pillow", "huggingface_hub")
         .add_local_file(HERE / "gemma_scoring.py", "/root/gemma_scoring.py")
         .add_local_file(HERE / "gemma_binary_scoring.py", "/root/gemma_binary_scoring.py"))
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
