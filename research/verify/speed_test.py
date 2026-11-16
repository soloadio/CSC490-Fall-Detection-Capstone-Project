#!/usr/bin/env python3
"""Do the trained checkers use SPEED? Falls vs deliberate downward movements that end in similar poses.

Held-out clips (val + test, 32 frames, frames32_{val,test}.jsonl):
  falls                 label "fall", 20 per group: UP-Fall activities 1-5 (hands, knees, backward, sideways, into an
                        empty chair), EDF, MCFD
  deliberate movements  label "sit_down", "lie_down", and UP-Fall activity 9 "other" (bending down to pick something
                        up): the body goes down like in a fall, but in a controlled way
A checker that uses speed scores falls high and deliberate movements low; one that judges by pose alone cannot tell a
fall that ends sitting from a sit-down. Reported: share called "fall" per group and AUROC (falls vs deliberate).

    python3 speed_test.py --out /scratch/wuhungma/speed_test     (GPU; resumable per checker)
    python3 speed_test.py --out /scratch/wuhungma/speed_test --report
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "app"), "/project/def-ajbonner/group12"]
SPLITS = Path("/project/def-ajbonner/group12/splits")
M = "/scratch/wuhungma/models"
CHECKERS = {  # name: (kind, adapter, max_pixels, evidence_first)
    "gemma": ("gemma", "/scratch/wuhungma/lora/r16_reasons_ep1/adapter", None, False),
    "gemma_syn": ("gemma", "/scratch/wuhungma/lora/gemma_r16_reasons_syn/adapter", None, False),
    "qwen_current": ("qwen", "/scratch/wuhungma/lora/qwen_r16_reasons/adapter", 78000, False),
    "qwen_v3": ("qwen", "/scratch/wuhungma/lora/qwen_v3/adapter", 345600, True),
    "qwen_v3_syn": ("qwen", "/scratch/wuhungma/lora/qwen_v3_syn/adapter", 345600, True),  # + OF-Syn lie-downs
}
OFSYN = Path("/scratch/wuhungma/ofsyn")  # OF-Syn val + test clips (build_ofsyn_clips.py): synthetic, held out


def group_of(e):
    if e["dataset"] == "of_syn":
        return "fall: OF-Syn (synthetic)" if e["label_name"] == "fall" else "deliberate: lie_down, OF-Syn (synthetic)"
    if e["dataset"] == "up_fall":
        a = int(re.search(r"Activity(\d+)", e["event_id"]).group(1))
        if e["label_name"] == "fall" and a <= 5:
            return f"fall: UP-Fall A{a}"
        if a == 9 and e["label_name"] == "other":
            return "deliberate: pick up object"
    elif e["label_name"] == "fall":
        return f"fall: {e['dataset'].upper()}"
    if e["label_name"] in ("sit_down", "lie_down"):
        return f"deliberate: {e['label_name']}"
    return None


def pick(n_fall=20, seed=0):
    by = collections.defaultdict(list)
    for sp, f in [(sp, d / f"frames32_{sp}.jsonl") for sp in ("val", "test") for d in (SPLITS, OFSYN)]:
        if not f.exists():
            continue
        for e in map(json.loads, open(f)):
            g = group_of(e)
            if g:
                for cam, v in e["cams"].items():
                    if len(v.get("frames", [])) >= 32:
                        by[g].append({"id": f"{e['event_id']}#cam{cam}", "group": g, "split": sp,
                                      "frames": v["frames"][:32]})
    rng, out = random.Random(seed), []
    for g in sorted(by):
        out += by[g] if g.startswith("deliberate") or "OF-Syn" in g else rng.sample(by[g], min(n_fall, len(by[g])))
    return out


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    return sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))


def report(out):
    rows = [json.loads(l) for l in open(out / "scores.jsonl")]
    names = [c for c in CHECKERS if any(r["checker"] == c for r in rows)]
    groups = sorted({r["group"] for r in rows}, key=lambda g: (not g.startswith("fall"), g))
    print("note: the original speed-test numbers are the 'all real' rows; OF-Syn rows are synthetic and held out\n")
    s = {(r["checker"], r["id"]): r["p_fall"] for r in rows}
    ids = collections.defaultdict(set)
    for r in rows:
        ids[r["group"]].add(r["id"])
    print("share of clips called fall (P >= 0.5); falls should be high, deliberate movements low")
    print(f"{'group':30} {'n':>3} " + " ".join(f"{c:>13}" for c in names))
    for g in groups:
        print(f"{g:30} {len(ids[g]):3d} " + " ".join(
            f"{sum(s[(c, i)] >= 0.5 for i in ids[g] if (c, i) in s):>6}/{sum((c, i) in s for i in ids[g]):<6}"
            for c in names))
    print(f"\n{'AUROC falls vs deliberate':34} " + " ".join(f"{c:>13}" for c in names))
    real = [g for g in groups if "OF-Syn" not in g]
    for sub in ("all real", "fall: UP-Fall A2", "fall: UP-Fall A5", "fall: MCFD", "real falls vs real lie_down",
                "real falls vs OF-Syn lie_down"):
        pos_ids = [i for g in real if g.startswith("fall") and (sub.startswith(("all", "real")) or g == sub)
                   for i in ids[g]]
        neg_g = {"real falls vs real lie_down": ["deliberate: lie_down"],
                 "real falls vs OF-Syn lie_down": ["deliberate: lie_down, OF-Syn (synthetic)"]}.get(
            sub, [g for g in real if g.startswith("deliberate")])
        neg_ids = [i for g in neg_g for i in ids[g]]
        print(f"{('  ' + sub):34} " + " ".join(
            f"{auroc([s[(c, i)] for i in pos_ids if (c, i) in s], [s[(c, i)] for i in neg_ids if (c, i) in s]):>13.3f}"
            for c in names))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--checkers", default=",".join(CHECKERS))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.report:
        return report(a.out)
    import torch
    from PIL import Image
    clips = pick()
    (a.out / "clips.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips))
    f_out = a.out / "scores.jsonl"
    done = {(r["checker"], r["id"]) for r in map(json.loads, open(f_out))} if f_out.exists() else set()
    for name in a.checkers.split(","):
        kind, adapter, px, ef = CHECKERS[name]
        todo = [c for c in clips if (name, c["id"]) not in done]
        print(f"[{name}] {len(todo)} clips to score", flush=True)
        if not todo:
            continue
        if kind == "gemma":
            import gemma_binary_scoring as gs
            from verifier import GemmaVerifier
            v = GemmaVerifier(f"{M}/gemma-3-12b-it", adapter, device="cuda")
            v.load()

            def score(frames):  # P(fall) only, no text generation
                content = [{"type": "text", "text": gs.instruction(len(frames))}]
                for i, im in enumerate(frames):
                    tag = "earliest" if i == 0 else "latest" if i == len(frames) - 1 else "later"
                    content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
                inp = v.processor.apply_chat_template([{"role": "user", "content": content}],
                                                      add_generation_prompt=True, tokenize=True, return_dict=True,
                                                      return_tensors="pt").to("cuda", dtype=torch.bfloat16)
                probs, _, _ = gs.to_probs(gs.score_labels(v.model, inp, v.pre, v.plan))
                return float(probs["fall"]), ""
            model = v.model
        else:
            from qwen_scoring import QwenScorer
            qs = QwenScorer(f"{M}/qwen2-vl-7b-instruct", adapter, max_pixels=px, evidence_first=ef).load()
            model = qs.model

            def score(frames):
                r = qs.check(frames)
                return r["p_fall"], r["text"]
        with open(f_out, "a") as f:
            for n, c in enumerate(todo, 1):
                frames = [Image.open(p).convert("RGB") for p in c["frames"]]
                p, text = score(frames)
                f.write(json.dumps({"checker": name, "id": c["id"], "group": c["group"], "split": c["split"],
                                    "p_fall": round(p, 5), "text": text}) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}", flush=True)
        del model
        torch.cuda.empty_cache()
    report(a.out)


if __name__ == "__main__":
    main()
