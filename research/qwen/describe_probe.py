#!/usr/bin/env python3
"""Why does Qwen + LoRA write the same sentence for most falls? Generate descriptions of a few alarm clips three ways:

  base_describe   base Qwen2-VL, plain prompt "describe the motion" (can Qwen's vision see kneeling / sitting at all?)
  lora_normal     Qwen + LoRA, the checker prompt, free generation (what the app gets: verdict first, then evidence)
  lora_evidence   Qwen + LoRA, the checker prompt, answer forced to start with {"evidence": " (evidence before verdict)

    python3 describe_probe.py --adapter <dir> --max-pixels 345600 --out <file.jsonl>
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

SPACE = "/scratch/wuhungma/deploy/space"
CLIPS = Path("/scratch/wuhungma/verify_gemma_reasons")
SIX = ["val_mv0032_cam1", "val_mv0033_cam1", "val_mv0034_cam1", "val_mv0042_cam2", "test_mv0074_cam1", "test_mv0075_cam1"]
DESCRIBE = ("These are {k} consecutive frames, in time order, of one person filmed by a single camera. Describe what "
            "the person does: their pose in the first frame, how they move (direction and speed), and their exact "
            "pose in the last frame (for example standing, kneeling, sitting, on hands and knees, lying on their "
            "back, lying face down). Two or three sentences.")


def pick(n_per_act, seed=0):
    """The six disagreement clips + n real-fall alarm clips per UP-Fall activity 1-5 (single-camera, wait1)."""
    random.seed(seed)
    by = {}
    for c in map(json.loads, open(CLIPS / "clips.jsonl")):
        m = re.search(r"Activity(\d+)", c.get("video", ""))
        if c.get("variant") == "wait1" and m and c["alarm"].get("deployed", {}).get("kind") == "hit":
            by.setdefault(m.group(1), []).append(c["clip_id"])
    out = [(c, "six") for c in SIX]
    for a in sorted(by):
        out += [(c, f"A{a}") for c in random.sample(by[a], min(n_per_act, len(by[a])))]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/scratch/wuhungma/models/qwen2-vl-7b-instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--max-pixels", type=int, default=345600)
    ap.add_argument("--per-activity", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path[:0] = [SPACE]
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    import gemma_binary_scoring as gs
    proc = AutoProcessor.from_pretrained(a.base)
    proc.image_processor.size["longest_edge"] = a.max_pixels
    clips = pick(a.per_activity)

    def gen(model, cid, instruction, prefix=""):
        frames = [Image.open(CLIPS / "clips" / cid / f"f{i:02d}.jpg").convert("RGB") for i in range(32)]
        content = [{"type": "text", "text": instruction}]
        for i, im in enumerate(frames):
            tag = "earliest" if i == 0 else "latest" if i == 31 else "later"
            content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False) + prefix
        inp = proc(text=[text], images=frames, return_tensors="pt").to(model.device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            g = model.generate(**inp, max_new_tokens=120, do_sample=False)
        return prefix + proc.decode(g[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda").eval()
    rows = {cid: {"clip_id": cid, "group": g} for cid, g in clips}
    for cid, _ in clips:
        rows[cid]["base_describe"] = gen(model, cid, DESCRIBE.format(k=32))
        print(cid, "base done", flush=True)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    for cid, _ in clips:
        rows[cid]["lora_normal"] = gen(model, cid, gs.instruction(32))
        rows[cid]["lora_evidence"] = gen(model, cid, gs.instruction(32), prefix='{"evidence": "')
        print(cid, "lora done", flush=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))


if __name__ == "__main__":
    main()
