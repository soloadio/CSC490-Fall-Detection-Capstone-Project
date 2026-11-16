#!/usr/bin/env python3
"""Can a base vision-language model tell HOW FAST a person goes down? Images vs video input, Qwen2-VL vs Qwen3-VL.

Zero-shot, no adapter. Clips: the 6 knee/chair falls the checkers disagreed on, 25 other real falls (alarm clips,
UP-Fall activities 1-5), and 24 controlled movements from the training split (12 sit_down, 12 lie_down) that end in
similar poses. Each model describes start / change / speed / end; the speed answer is a category, so it can be
scored: falls should be "under 1 second", sit_down / lie_down "1-2 seconds" or "more than 2 seconds".

Conditions:
  q2_images   Qwen2-VL-7B, 32 separate images with "Frame N" labels (how the checker is fed today)
  q2_video    Qwen2-VL-7B, the same 32 frames as one video at 8 fps (time positions inside the model)
  q3_video    Qwen3-VL-8B, the same 32 frames as one video at 8 fps (adds "<t seconds>" timestamps)

    python3 video_probe.py --out /scratch/wuhungma/video_probe/probe.jsonl [--conditions q2_images,q2_video,q3_video]
    python3 video_probe.py --report --out /scratch/wuhungma/video_probe/probe.jsonl
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

GROUP = Path("/project/def-ajbonner/group12")
CLIPS = Path("/scratch/wuhungma/verify_gemma_reasons")
Q2 = "/scratch/wuhungma/models/qwen2-vl-7b-instruct"
Q3 = "/scratch/wuhungma/models/qwen3-vl-8b-instruct"
SIX = ["val_mv0032_cam1", "val_mv0033_cam1", "val_mv0034_cam1", "val_mv0042_cam2", "test_mv0074_cam1", "test_mv0075_cam1"]
PROMPT = """This {what} shows one person over 4 seconds, filmed by a single camera. Answer in exactly this format:
start: <the person's pose and where they are at the beginning>
change: <what their body does, or "no change">
speed: <how long the change takes: "under 1 second", "1-2 seconds", "more than 2 seconds", or "no change">
end: <their pose at the end and which body parts touch the floor, mattress or furniture>"""


def pick(seed=0):
    rng = random.Random(seed)
    out = [dict(id=c, kind="fall", group="six", frames=[str(CLIPS / "clips" / c / f"f{i:02d}.jpg") for i in range(32)])
           for c in SIX]
    by = collections.defaultdict(list)
    for c in map(json.loads, open(CLIPS / "clips.jsonl")):
        m = re.search(r"Activity(\d+)", c.get("video", ""))
        if c.get("variant") == "wait1" and m and m.group(1) in "12345" and \
                c["alarm"].get("deployed", {}).get("kind") == "hit":
            by[m.group(1)].append(c["clip_id"])
    for a in sorted(by):
        for c in rng.sample(by[a], 5):
            out.append(dict(id=c, kind="fall", group=f"A{a}",
                            frames=[str(CLIPS / "clips" / c / f"f{i:02d}.jpg") for i in range(32)]))
    ctl = collections.defaultdict(list)
    for e in map(json.loads, open(GROUP / "splits" / "frames32_train.jsonl")):
        if e["label_name"] in ("sit_down", "lie_down"):
            for cam, v in e["cams"].items():
                if len(v.get("frames", [])) >= 32:
                    ctl[e["label_name"]].append(dict(id=f"{e['event_id']}#cam{cam}", kind=e["label_name"],
                                                     group=e["label_name"], frames=v["frames"][:32]))
    for lab in ("sit_down", "lie_down"):
        out += rng.sample(ctl[lab], min(12, len(ctl[lab])))
    return out


def speed_of(text):
    m = re.search(r"speed:\s*(.*)", text, re.I)
    s = (m.group(1) if m else "").lower()
    if "under 1" in s or "less than 1" in s:
        return "fast"
    if "1-2" in s or "more than 2" in s or "1 to 2" in s:
        return "slow"
    return "none" if "no change" in s else "?"


def report(path):
    rows = [json.loads(l) for l in open(path)]
    conds = [k for k in rows[0] if k.startswith("q")]
    print(f"speed answers ({len(rows)} clips): fast = 'under 1 second', slow = '1-2' or 'more than 2 seconds'")
    print(f"{'condition':10} {'group':9} {'n':>3} {'fast':>5} {'slow':>5} {'no chg':>6} {'?':>3}")
    for cnd in conds:
        for g in ("six", "falls", "sit_down", "lie_down"):
            rs = [r for r in rows if (r["group"] == g or (g == "falls" and r["kind"] == "fall" and r["group"] != "six"))]
            c = collections.Counter(speed_of(r[cnd]) for r in rs)
            print(f"{cnd:10} {g:9} {len(rs):3d} {c['fast']:5d} {c['slow']:5d} {c['none']:6d} {c['?']:3d}")
        ok = sum((speed_of(r[cnd]) == "fast") == (r["kind"] == "fall") for r in rows)
        print(f"{cnd:10} correct speed class (fast for falls, not fast for sit/lie down): {ok}/{len(rows)}\n")
    print("the six clips:")
    for r in rows[:6]:
        print(f"\n{r['id']}")
        for cnd in conds:
            print(f"  {cnd:10} " + " | ".join(l.strip() for l in r[cnd].splitlines() if l.strip()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", default="q2_images,q2_video,q3_video")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.report:
        return report(a.out)
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    import vl_inputs

    clips = pick()[:a.limit or None]
    rows = {c["id"]: {k: c[k] for k in ("id", "kind", "group")} for c in clips}
    if a.out.exists():  # resume: keep finished conditions
        for r in map(json.loads, open(a.out)):
            if r["id"] in rows:
                rows[r["id"]].update(r)
    for cnd in a.conditions.split(","):
        if all(cnd in r for r in rows.values()):
            continue
        base = Q3 if cnd.startswith("q3") else Q2
        proc = AutoProcessor.from_pretrained(base)
        model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda").eval()
        for c in clips:
            frames = [Image.open(p).convert("RGB") for p in c["frames"]]
            what = "sequence of 32 frames (8 per second)" if cnd.endswith("images") else "video"
            inp = vl_inputs.build(proc, frames, PROMPT.format(what=what), "images" if cnd.endswith("images") else "video")
            inp = inp.to(model.device)
            with torch.inference_mode():
                g = model.generate(**inp, max_new_tokens=160, do_sample=False)
            rows[c["id"]][cnd] = proc.decode(g[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            print(cnd, c["id"], "|", rows[c["id"]][cnd].replace("\n", " | ")[:200], flush=True)
            a.out.write_text("".join(json.dumps(r) + "\n" for r in rows.values()))
        del model
        torch.cuda.empty_cache()
    report(a.out)


if __name__ == "__main__":
    main()
