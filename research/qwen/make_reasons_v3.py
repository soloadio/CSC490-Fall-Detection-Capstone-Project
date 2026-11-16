#!/usr/bin/env python3
"""Training explanations v3: specific, structured, not repetitive (for train_lora_qwen.py --reasons).

What went wrong before:
  v1 (group12/make_reasons.py)   the teacher was told "uncontrolled drop" / "deliberate, controlled movement" and
                                 wrote "rapidly" for every fall and "slowly" for every sit/lie down: the label leaked
                                 through speed words, and Qwen learned one stock fall sentence.
  v2 (make_pose_reasons.py)      no speed words, but every sentence was "In the first frame, the person is standing.
                                 ... in the last frame, they are ..." (53% distinct, one sentence 218 times), the
                                 teacher (base Gemma) wrote "lying on their back" for most forward falls, and Qwen
                                 learned the frame of the sentence instead of its content (16% distinct outputs).

v3:
  * no boilerplate: three short fields, "start: ...; change: ...; end: ...", all content
  * details that differ between clips: where the person is, furniture and objects near them, which way they face,
    what their hands do, which body parts touch the floor or furniture at the end
  * the writer is a base Qwen model (--model, --mode images|video): base Qwen described the knee and sitting falls
    correctly in describe_probe.py, base Gemma did not
  * repeats are capped: any explanation written more than --max-repeat times is rewritten with sampling and a request
    for one more distinguishing detail (up to --retries times)
  * still no speed or intent words and no "fall"; optional --speed adds a "speed:" field (only if video_probe.py shows
    the writer gets speed right)

    python3 make_reasons_v3.py --pilot 24 --out <dir>           # read before the full run
    python3 make_reasons_v3.py --shard 0/3 --out <dir>          # x3, then:  python3 make_reasons_v3.py --dedupe --out <dir>
"""
import argparse
import collections
import json
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
GROUP = "/project/def-ajbonner/group12"
WHAT = {  # where the person is / ends up; no speed, no intent (same as make_pose_reasons.py)
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
POSITIVE = {"fall", "fallen"}
BANNED = re.compile(r"\b(slow\w*|quick\w*|rapid\w*|sudden\w*|gradual\w*|careful\w*|control\w*|deliberate\w*|"
                    r"intentional\w*|purpose\w*|fall\w*|fell|collaps\w*|uncontrolled)\b", re.I)


def prompt(label, mode, speed, extra=""):
    what = "video" if mode == "video" else "sequence of 32 frames (8 per second)"
    sp = ('speed: <how long the change takes: "under 1 second", "1-2 seconds", "more than 2 seconds" or "no change">\n'
          if speed else "")
    return f"""This {what} shows one person over 4 seconds, filmed by a single camera.
What happens (it is correct): {WHAT[label]}

Describe THIS clip in exactly this format, one short line per field, with details specific to this clip (where the person is, furniture or objects near them, which way they face, what their hands do):
start: <pose and place at the beginning>
change: <what the body does, or "none">
{sp}end: <pose at the end and which body parts touch the floor, mattress or furniture>
Do not use the words fall, fell, falling, slowly, quickly, rapidly, suddenly, gradually, carefully, controlled or deliberately. Do not mention labels.{extra}"""



def fields(text, speed):
    """'start: a\\nchange: b\\nend: c' -> 'start: a; change: b; end: c' ('' if a field is missing)."""
    keys = ["start", "change"] + (["speed"] if speed else []) + ["end"]
    got = {}
    for line in text.splitlines():
        m = re.match(r"\s*\**\s*(start|change|speed|end)\s*\**\s*:\s*(.+)", line, re.I)
        if m:
            got[m.group(1).lower()] = m.group(2).strip().rstrip(".").strip("*").strip()
    return "; ".join(f"{k}: {got[k]}" for k in keys) if all(k in got for k in keys) else ""


def clips_of(manifest):
    idx = list(range(32))
    out = []
    for line in open(manifest):
        e = json.loads(line)
        for cam, v in e["cams"].items():
            if len(v.get("frames", [])) >= 32:
                out.append({"clip_id": f"{e['event_id']}#cam{cam}", "event_id": e["event_id"], "cam": cam,
                            "label_name": e["label_name"], "frames": [v["frames"][i] for i in idx]})
    return out


class Writer:
    def __init__(self, model_dir, mode, speed):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.mode, self.speed = torch, mode, speed
        self.proc = AutoProcessor.from_pretrained(model_dir)
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                                 device_map="cuda").eval()

    def write(self, c, sample=False, extra=""):
        import vl_inputs
        from PIL import Image
        frames = [Image.open(p).convert("RGB") for p in c["frames"]]
        inp = vl_inputs.build(self.proc, frames, prompt(c["label_name"], self.mode, self.speed, extra),
                              self.mode).to(self.model.device)
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
        with self.torch.inference_mode():
            g = self.model.generate(**inp, max_new_tokens=140, **kw)
        raw = self.proc.decode(g[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        return fields(raw, self.speed), raw


def record(c, ev, raw):
    return {"clip_id": c["clip_id"], "event_id": c["event_id"], "cam": c["cam"], "label_name": c["label_name"],
            "verdict": "fall" if c["label_name"] in POSITIVE else "no_fall", "evidence": ev,
            "occlusion_flag": False, "banned_word": bool(BANNED.search(ev)), "raw": raw}


def stats(rows, tag):
    ev = [r["evidence"] for r in rows]
    c = collections.Counter(ev)
    print(f"[{tag}] {len(ev)} explanations, {100 * len(c) / max(1, len(ev)):.0f}% distinct, most repeated "
          f"{c.most_common(1)[0][1] if c else 0}x, empty {sum(not e for e in ev)}, banned words "
          f"{sum(r['banned_word'] for r in rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path(GROUP) / "splits" / "frames32_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="/scratch/wuhungma/models/qwen3-vl-8b-instruct")
    ap.add_argument("--mode", choices=["images", "video"], default="video")
    ap.add_argument("--speed", action="store_true", help="add a speed: field")
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--dedupe", action="store_true", help="merge the shards, rewrite repeats -> reasons.jsonl")
    ap.add_argument("--max-repeat", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    clips = clips_of(a.train)

    if a.dedupe:
        rows = {}
        for f in sorted(a.out.glob("reasons_shard*.jsonl")):
            for r in map(json.loads, open(f)):
                rows[r["clip_id"]] = r
        stats(rows.values(), "before dedupe")
        by_id = {c["clip_id"]: c for c in clips}
        w = Writer(a.model, a.mode, a.speed)
        for attempt in range(a.retries):
            cnt = collections.Counter(r["evidence"] for r in rows.values())
            redo = [cid for cid, r in rows.items() if not r["evidence"] or r["banned_word"]
                    or cnt[r["evidence"]] > a.max_repeat]
            # keep max_repeat copies of each repeated explanation, rewrite the rest
            seen = collections.Counter()
            todo = []
            for cid in redo:
                ev = rows[cid]["evidence"]
                seen[ev] += 1
                if not ev or rows[cid]["banned_word"] or seen[ev] > a.max_repeat:
                    todo.append(cid)
            print(f"rewrite round {attempt + 1}: {len(todo)} explanations", flush=True)
            for cid in todo:
                ev, raw = w.write(by_id[cid], sample=True, extra=" Mention one more detail that is specific to this "
                                  "clip (for example the colour of the person's clothes, the furniture, or which way "
                                  "they face).")
                if ev and not BANNED.search(ev):
                    rows[cid] = record(by_id[cid], ev, raw)
        rows = [r for r in rows.values() if r["evidence"] and not r["banned_word"]]
        stats(rows, "after dedupe (empty / banned dropped)")
        (a.out / "reasons.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return

    if a.pilot:
        by = collections.defaultdict(list)
        for c in clips:
            by[c["label_name"]].append(c)
        rng, pick = random.Random(1), []
        order = ["fall", "lie_down", "sit_down", "fallen"] + sorted(set(by) - {"fall", "lie_down", "sit_down", "fallen"})
        while len(pick) < a.pilot and any(by.values()):
            for lab in order:
                if by[lab] and len(pick) < a.pilot:
                    pick.append(by[lab].pop(rng.randrange(len(by[lab]))))
        clips = pick
    si, sn = map(int, a.shard.split("/"))
    clips = clips[si::sn]
    out_file = a.out / ("reasons_pilot.jsonl" if a.pilot else f"reasons_shard{si}of{sn}.jsonl")
    done = {json.loads(l)["clip_id"] for l in open(out_file)} if out_file.exists() else set()
    todo = [c for c in clips if c["clip_id"] not in done]
    print(f"{len(clips)} clips, {len(done)} done, {len(todo)} to do -> {out_file}", flush=True)
    w = Writer(a.model, a.mode, a.speed)
    t0 = time.time()
    with open(out_file, "a") as f:
        for n, c in enumerate(todo, 1):
            ev, raw = w.write(c)
            f.write(json.dumps(record(c, ev, raw)) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} | {(time.time() - t0) / n:.1f} s/clip", flush=True)
    stats([json.loads(l) for l in open(out_file)], out_file.name)


if __name__ == "__main__":
    main()
