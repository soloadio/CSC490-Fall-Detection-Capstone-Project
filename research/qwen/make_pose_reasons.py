#!/usr/bin/env python3
"""Pose-only evidence sentences for the training clips (v2 of group12/make_reasons.py).

Why v2: the first reasons were written by a teacher that was told the label as "an uncontrolled drop" (fall) or "a
DELIBERATE, controlled movement" (lie down / sit down) and was asked about speed, so it wrote "rapidly" for every fall
(393 of 1086 fall reasons, 0 non-fall) and "slowly" for every deliberate movement (33 of 42 sit_down). The checkers
learned "slowly -> no_fall" and one stock sentence for falls ("rapidly rotates and falls ... prone"), even for falls
onto the knees or into a sitting position (research/qwen/describe_probe.py).

v2: same teacher (Gemma 3 12B-it, no adapter), told only WHERE the person ends up, asked for checkable facts: the pose in
the first frame, whether it changes, the final pose and what touches the floor, and whether the hands reach out first.
(Asking for the frame numbers of the change did not work in the pilots: with an example the teacher copied the
example's numbers, without one it wrote "frame 1 to frame 32".) Words about speed or
intent and the word "fall" are forbidden in the evidence; the verdict must come from what the frames show.

    python3 make_pose_reasons.py --pilot 24 --out /scratch/wuhungma/reasons_v2
    python3 make_pose_reasons.py --shard 0/2 --out /scratch/wuhungma/reasons_v2      # and --shard 1/2 on another GPU
    cat /scratch/wuhungma/reasons_v2/reasons_shard*.jsonl > /scratch/wuhungma/reasons_v2/reasons.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

GROUP = "/project/def-ajbonner/group12"
sys.dont_write_bytecode = True
sys.path.insert(0, GROUP)
from make_reasons import POSITIVE, clips_of, parse  # noqa: E402  same clips and JSON parsing as v1

MODEL = "/scratch/wuhungma/models/gemma-3-12b-it"
WHAT = {  # where the person is / ends up; no speed, no intent
    "fall": "during these frames the person goes from upright to the ground or a mattress.",
    "fallen": "the person is on the ground or a mattress for the whole clip.",
    "lie_down": "during these frames the person goes from upright or sitting to lying.",
    "lying": "the person is lying for the whole clip.",
    "sit_down": "during these frames the person goes from standing to sitting.",
    "sitting": "the person is sitting for the whole clip.",
    "stand_up": "during these frames the person goes from the ground, lying or sitting to standing.",
    "standing": "the person stands for the whole clip.",
    "walk": "the person walks or moves around.",
    "other": "the person does some everyday activity.",
}
BANNED = re.compile(r"\b(slow\w*|quick\w*|rapid\w*|sudden\w*|gradual\w*|careful\w*|control\w*|deliberate\w*|"
                    r"intentional\w*|purpose\w*|fall\w*|fell|collaps\w*|uncontrolled)\b", re.I)


def prompt(k, label):
    return f"""You are watching {k} consecutive frames (8 frames per second), in time order, of one person filmed by a single camera.
What happens in this clip (it is correct): {WHAT[label]}

Describe what the frames show, as facts someone could check frame by frame, in ONE or TWO short sentences:
1. the person's pose in the first frame (standing, walking, sitting, kneeling, lying on their back / front / side, ...);
2. whether their pose changes during the clip;
3. their pose in the last frame and which body parts touch the floor, mattress or furniture (for example "kneeling with both hands on the mattress", "sitting on the floor", "lying on their back on the mattress");
4. whether their hands or arms reach out to a surface before the body gets there.
Do NOT use words about speed or intention (slowly, quickly, rapidly, suddenly, gradually, carefully, controlled, deliberately) and do NOT use the words fall, fell or falling. Do not mention labels.

Answer with a single JSON object and nothing else:
{{"evidence": "<one or two short sentences>", "occlusion_flag": <true if the person is hidden or partly hidden, else false>}}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path(GROUP) / "splits" / "frames32_train.jsonl")
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/reasons_v2"))
    ap.add_argument("--pilot", type=int, default=0, help="only this many clips, spread over all labels")
    ap.add_argument("--shard", default="0/1", help="i/n: this job does every n-th clip starting at i")
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    clips = clips_of(a.train)
    if a.pilot:  # same pilot picker as v1, but the labels that matter most first
        by = defaultdict(list)
        for c in clips:
            by[c["label_name"]].append(c)
        rng, pick = random.Random(1), []
        order = ["fall", "lie_down", "sit_down", "fallen"] + sorted(set(by) - {"fall", "lie_down", "sit_down", "fallen"})
        while len(pick) < a.pilot and any(by.values()):
            for lab in order:
                if by[lab] and len(pick) < a.pilot:
                    pick.append(by[lab].pop(rng.randrange(len(by[lab]))))
        clips = pick
    clips = clips[si::sn]
    a.out.mkdir(parents=True, exist_ok=True)
    out_file = a.out / ("reasons_pilot.jsonl" if a.pilot else f"reasons_shard{si}of{sn}.jsonl")
    done = {json.loads(l)["clip_id"] for l in open(out_file)} if out_file.exists() else set()
    todo = [c for c in clips if c["clip_id"] not in done]
    print(f"{len(clips)} clips, {len(done)} done, {len(todo)} to do -> {out_file}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    t0, n_banned = time.time(), 0
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
                gen = model.generate(**inp, max_new_tokens=110, do_sample=False)
            raw = processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            ev, occ = parse(raw)
            n_banned += bool(BANNED.search(ev))
            f.write(json.dumps({"clip_id": c["clip_id"], "event_id": c["event_id"], "cam": c["cam"],
                                "label_name": c["label_name"],
                                "verdict": "fall" if c["label_name"] in POSITIVE else "no_fall",
                                "evidence": ev, "occlusion_flag": occ, "banned_word": bool(BANNED.search(ev)),
                                "raw": raw}) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} | {(time.time() - t0) / n:.1f} s/clip | banned words in {n_banned}", flush=True)


if __name__ == "__main__":
    main()
