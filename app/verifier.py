"""Second opinion on a VideoMAE alarm: a vision-language model looks at the last few seconds of each camera.

Offline experiment (verify_alarms.py / verify_mv.py in the project): the clip is N_FRAMES=32 frames evenly spaced over
the 3 s ending 1 s after the alarm, at the camera's own resolution; Gemma 3 12B-it + the LoRA adapter is forced to
start its answer with {"verdict": "  and P(fall) is read off the next-token probabilities (gemma_binary_scoring.py);
the answer text is then generated too, so the camera can say WHY ("evidence").

A checker is any object with  .name  and  .check(frames) -> dict(verdict, p_fall, reason, seconds)  where frames is a
list of PIL images in time order. To add another model (e.g. Qwen + LoRA) write another class like GemmaVerifier and
add it to load_verifiers().
"""
import json
import os
import re
import threading
import time
from collections import deque

import numpy as np
from PIL import Image

try:  # ZeroGPU Spaces: GPU work must run inside a @spaces.GPU function
    import spaces
except ImportError:
    spaces = None

N_FRAMES, CLIP_SECONDS, MAX_SIDE = 32, 3.0, 896


def gpu(duration):
    return spaces.GPU(duration=duration) if spaces else (lambda f: f)


class RawBuffer:
    """Recent (time, frame) pairs of one camera at (near) native resolution; cuts the clip a checker looks at."""

    def __init__(self, keep=8.0):
        self.keep, self.items = keep, deque()

    def push(self, t, rgb):
        im = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
        if max(im.size) > MAX_SIDE:  # Gemma resizes to 896 anyway; keeps memory small
            s = MAX_SIDE / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
        self.items.append((t, im.convert("RGB")))
        while self.items and self.items[0][0] < t - self.keep:
            self.items.popleft()

    def last_time(self):
        return self.items[-1][0] if self.items else None

    def clip(self, t_end, length=CLIP_SECONDS, n=N_FRAMES):
        """n frames at the even midpoints of n bins over [t_end - length, t_end] (nearest buffered frame), as in
        verify_alarms.clip_frames. None if the camera has sent nothing for that stretch."""
        items = list(self.items)
        if not items or items[-1][0] < t_end - length:
            return None
        times = np.array([t for t, _ in items])
        tg = t_end - length + (np.arange(n) + 0.5) * length / n
        j = np.clip(np.searchsorted(times, tg), 1, max(1, len(times) - 1))
        idx = np.zeros(n, dtype=int) if len(times) == 1 else np.where(
            np.abs(times[j - 1] - tg) <= np.abs(times[j] - tg), j - 1, j)
        return [items[i][1] for i in idx]


def parse_reason(text):
    """The "evidence" sentence of Gemma's JSON answer (tolerates cut-off JSON)."""
    m = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)', text or "")
    return m.group(1).strip() if m else (text or "").strip()[:200]


class GemmaVerifier:
    name = "Gemma 3 12B + LoRA"

    def __init__(self, base, adapter=None, token=None, device=None):
        self.base, self.adapter, self.token = base, adapter, token
        self.device = device or ("cuda" if (spaces or _cuda()) else "cpu")
        self.lock, self.ready = threading.Lock(), False

    def load(self):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        import gemma_binary_scoring as gs
        self.gs = gs
        kw = {"token": self.token} if self.token else {}
        self.processor = AutoProcessor.from_pretrained(self.base, **kw)
        self.pre, self.plan, bad = gs.label_plan(self.processor.tokenizer)
        assert not bad, bad
        model = Gemma3ForConditionalGeneration.from_pretrained(self.base, dtype=torch.bfloat16, **kw).eval()
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter, **kw).merge_and_unload().eval()
        self.model = model.to(self.device) if self.device != "cpu" else model
        self.ready = True
        print(f"{self.name} ready on {self.device}", flush=True)

    @gpu(duration=90)
    def _run(self, frames):
        import torch
        gs = self.gs
        content = [{"type": "text", "text": gs.instruction(len(frames))}]
        for i, im in enumerate(frames):
            tag = "earliest" if i == 0 else "latest" if i == len(frames) - 1 else "later"
            content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
        dev = next(self.model.parameters()).device
        inp = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt"
                                                 ).to(dev, dtype=torch.bfloat16)
        probs, mass, _ = gs.to_probs(gs.score_labels(self.model, inp, self.pre, self.plan))
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=100, do_sample=False)
        text = self.processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        return float(probs["fall"]), text

    def check(self, frames, tau=0.5):
        t = time.perf_counter()
        with self.lock:
            p, text = self._run(frames)
        return {"verifier": self.name, "verdict": "fall" if p >= tau else "no_fall", "p_fall": p,
                "reason": parse_reason(text), "seconds": time.perf_counter() - t}


class ModalVerifier:
    """The same Gemma 3 12B + LoRA check, run on a Modal GPU (modal_app.py). Needs the `modal` package and a Modal token
    (env MODAL_TOKEN_ID / MODAL_TOKEN_SECRET on a Space, or `modal token new` on the cluster)."""
    name = "Gemma 3 12B + LoRA"

    def __init__(self, app="fall-verifier"):
        self.app, self.ready = app, False

    def load(self):
        import modal
        self.fn = modal.Cls.from_name(self.app, "Verifier")().check
        self.ready = True
        print(f"{self.name} on Modal ({self.app}) ready", flush=True)

    def check(self, frames, tau=0.5):
        import io
        t, jpegs = time.perf_counter(), []
        for im in frames:  # JPEG quality 95, as the offline clips were saved
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=95)
            jpegs.append(buf.getvalue())
        r = self.fn.remote(jpegs)
        return {"verifier": self.name, "verdict": "fall" if r["p_fall"] >= tau else "no_fall", "p_fall": r["p_fall"],
                "reason": parse_reason(r["text"]), "seconds": time.perf_counter() - t}


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def load_verifiers():
    """Checkers named in env VERIFIERS (comma separated; default none): gemma (on this machine's GPU) or gemma-modal
    (on Modal, see modal_app.py; env MODAL_APP). Gemma: GEMMA_BASE (hub id or local folder),
    GEMMA_ADAPTER (hub id or local folder, optional), HF_TOKEN (Space secret; Gemma is a gated model)."""
    out = []
    for name in filter(None, (x.strip() for x in os.environ.get("VERIFIERS", "").split(","))):
        if name == "gemma":
            out.append(GemmaVerifier(os.environ.get("GEMMA_BASE", "google/gemma-3-12b-it"),
                                     os.environ.get("GEMMA_ADAPTER") or None, os.environ.get("HF_TOKEN")))
        elif name == "gemma-modal":
            out.append(ModalVerifier(os.environ.get("MODAL_APP", "fall-verifier")))
        else:
            raise ValueError(f"unknown verifier {name!r}")
    return out


def decide(reports):
    """Combine the votes of all (checker, camera) pairs: the alarm stands if at least half say fall (a tie keeps the
    alarm: a missed fall is worse than a false alarm). No reports (checker failed): keep the alarm."""
    votes = [r["verdict"] == "fall" for r in reports if "verdict" in r]
    return not votes or sum(votes) * 2 >= len(votes)
