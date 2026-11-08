#!/usr/bin/env python3
"""Make "evidence" sentences for the LoRA training clips (rationale distillation).

Gemma 3 12B-it (no adapter) sees one training clip (32 frames of one camera, as in train_lora.py) and is TOLD the
correct OmniFall label with its definition; it describes only the visible evidence for it: how the person moved
across the frames, whether they used their arms or hands, and where they ended up. The sentences become the
"evidence" field of the training answers, so the retrained LoRA learns to say why, not only to say the verdict.

    python3 make_reasons.py --pilot 24            # a few clips of every label, to read before the full run
    python3 make_reasons.py                       # every training clip (resumable)
Output: /scratch/wuhungma/reasons/reasons{_pilot}.jsonl  (clip_id, event_id, cam, label_name, verdict, evidence, occlusion_flag, raw)
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + [str(p) for p in HERE.parent.iterdir() if p.is_dir()]  # sibling folders of research/
MODEL = "/home/wuhungma/scratch/models/gemma-3-12b-it"

# OmniFall label definitions (gemma_scoring.DEFS), phrased as facts about this clip.
TRUTH = {  # what the label says; for states, how the person got there happened BEFORE the clip and is not visible
    "fall": "the person FALLS during these frames (an uncontrolled drop, not a deliberate movement).",
    "fallen": "the person is already on the ground or a mattress after a fall; the fall itself happened BEFORE these frames.",
    "lie_down": "the person lies down DELIBERATELY during these frames (a controlled movement, often supported with the arms).",
    "lying": "the person is already lying down, a position they got into deliberately BEFORE these frames.",
    "sit_down": "the person sits down during these frames, on a chair, bed or the ground.",
    "sitting": "the person is already sitting.",
    "stand_up": "the person stands up during these frames, from the ground, lying or sitting.",
    "standing": "the person is standing without walking.",
    "walk": "the person walks or moves around.",
    "other": "the person does some other everyday activity (not falling).",
}
POSITIVE = {"fall", "fallen"}


def prompt(k, label):
    return f"""You are watching {k} consecutive frames, in time order, of one person filmed by a single camera.
Ground truth for this clip (it is correct, do not question it): {TRUTH[label]}

Write the visible evidence for this, as a careful observer would describe it to someone who cannot see the video.
In ONE or TWO short sentences: the person's position and place in the first frame; how their body moves between the
first and the last frame (speed, direction, whether they use their arms or hands for support), or that they stay
still; and the position they end in. Describe ONLY what these frames show: if the person is already lying, sitting or
standing in the first frame, say so and do not invent how they got there. Do not mention the ground truth, labels,
or the words "ground truth".

Answer with a single JSON object and nothing else:
{{"evidence": "<one or two short sentences>", "occlusion_flag": <true if the person is hidden or partly hidden, else false>}}"""


def clips_of(manifest, k=32):
    idx = [round(j * 31 / (k - 1)) for j in range(k)]  # same frames as train_lora.load_clips
    out = []
    for line in open(manifest):
        e = json.loads(line)
        for cam, v in e["cams"].items():
            if len(v.get("frames", [])) >= 32:
                out.append({"clip_id": f"{e['event_id']}#cam{cam}", "event_id": e["event_id"], "cam": cam,
                            "label_name": e["label_name"], "frames": [v["frames"][i] for i in idx]})
    return out


def parse(text):
    t = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        d = json.loads(t)
        return str(d.get("evidence", "")).strip(), bool(d.get("occlusion_flag", False))
    except Exception:  # noqa: BLE001 - fall back to a regex on cut-off JSON
        m = re.search(r'"evidence"\s*:\s*"((?:[^"\\]|\\.)*)', t)
        return (m.group(1).strip() if m else ""), bool(re.search(r'"occlusion_flag"\s*:\s*true', t))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=HERE.parent / "splits" / "frames32_train.jsonl")
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/reasons"))
    ap.add_argument("--pilot", type=int, default=0, help="only this many clips, spread over all labels")
    ap.add_argument("--tag", default="", help="suffix of the output file name")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    clips = clips_of(a.train)
    if a.pilot:
        by = defaultdict(list)
        for c in clips:
            by[c["label_name"]].append(c)
        rng, pick = random.Random(0), []
        while len(pick) < a.pilot and any(by.values()):
            for lab in sorted(by):
                if by[lab] and len(pick) < a.pilot:
                    pick.append(by[lab].pop(rng.randrange(len(by[lab]))))
        clips = pick
    a.out.mkdir(parents=True, exist_ok=True)
    out_file = a.out / (("reasons_pilot" if a.pilot else "reasons") + a.tag + ".jsonl")
    done = {json.loads(l)["clip_id"] for l in open(out_file)} if out_file.exists() else set()
    todo = [c for c in clips if c["clip_id"] not in done]
    print(f"{len(clips)} clips, {len(done)} done, {len(todo)} to do -> {out_file}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    t0 = time.time()
    with open(out_file, "a") as f:
        for n, c in enumerate(todo, 1):
            content = [{"type": "text", "text": prompt(len(c["frames"]), c["label_name"])}]
            for i, p in enumerate(c["frames"]):
                tag = "earliest" if i == 0 else "latest" if i == len(c["frames"]) - 1 else "later"
                content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"},
                            {"type": "image", "image": Image.open(p).convert("RGB")}]
            inp = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                tokenize=True, return_dict=True, return_tensors="pt"
                                                ).to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                gen = model.generate(**inp, max_new_tokens=90, do_sample=False)
            raw = processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            ev, occ = parse(raw)
            f.write(json.dumps({"clip_id": c["clip_id"], "event_id": c["event_id"], "cam": c["cam"],
                                "label_name": c["label_name"],
                                "verdict": "fall" if c["label_name"] in POSITIVE else "no_fall",
                                "evidence": ev, "occlusion_flag": occ, "raw": raw}) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} | {(time.time() - t0) / n:.1f} s/clip", flush=True)


if __name__ == "__main__":
    main()
