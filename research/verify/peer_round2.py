#!/usr/bin/env python3
"""Round 2 for the multi-camera checkers: does a checker change its verdict when it reads the other cameras' reports?

Round 1 = the saved per-clip results of Gemma 3 + LoRA (reasons) and Qwen2-VL + LoRA on the multi-camera alarm events
(events_mv.jsonl). In round 2 each (camera, checker) pair is scored again on its own camera's 32 frames, with the
round-1 verdicts and evidence sentences of the OTHER cameras' checkers appended to the prompt. Only P(fall) is
re-scored (forced prefix, no text generation).

    score   (GPU job)  writes <out>/round2.jsonl: one record per (event, camera, checker), resumable
    report  (CPU)      compares round 1 and round 2 under the app's room rule and the safe "only no -> yes" rule

    python3 peer_round2.py score --out /scratch/wuhungma/peer_round2
    python3 peer_round2.py report --out /scratch/wuhungma/peer_round2
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE = "/scratch/wuhungma/deploy/space"  # gemma_scoring.py, gemma_binary_scoring.py, qwen_scoring.py
G_DIR = Path("/scratch/wuhungma/verify_gemma_reasons")
Q_DIR = Path("/scratch/wuhungma/verify_qwen")
CHECKERS = {"G": "Gemma 3 12B + LoRA", "Q": "Qwen2-VL 7B + LoRA"}
TAU = 0.5


def load_round1():
    def res(d):
        return {r["clip_id"]: r for r in map(json.loads, open(d / "gemma_lora_clips_mv.jsonl")) if "p_fall" in r}
    r1 = {"G": res(G_DIR), "Q": res(Q_DIR)}
    events = [json.loads(l) for l in open(G_DIR / "events_mv.jsonl")]
    return events, r1


def evidence(text):
    try:
        return json.loads(text.strip().removeprefix("```json").removesuffix("```").strip()).get("evidence", "")
    except Exception:  # noqa: BLE001 - malformed answer: fall back to the raw text
        return text.strip()[:200]


def peer_notes(event, r1, cam):
    """Text block with the other cameras' round-1 reports (checkers named neutrally: A = Gemma, B = Qwen)."""
    lines = []
    for c, cid in sorted(event["clips"].items()):
        if c == cam:
            continue
        for m, name in (("G", "A"), ("Q", "B")):
            r = r1[m][cid]
            v = "fall" if r["p_fall"] >= TAU else "no_fall"
            lines.append(f'- Camera {c}, checker {name}: verdict {v}. "{evidence(r["text"])}"')
    return ("Other cameras in the same room filmed the same moment from different viewpoints. Their checkers reported:\n"
            + "\n".join(lines)
            + "\nThese reports can be wrong, and the person may look different or be hidden in other views. "
              "Use them as extra context, then give your answer for THIS camera's frames.")


def build_content(gs, frames, notes):
    content = [{"type": "text", "text": gs.instruction(len(frames))}]
    for i, im in enumerate(frames):
        tag = "earliest" if i == 0 else "latest" if i == len(frames) - 1 else "later"
        content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
    if notes:
        content.append({"type": "text", "text": notes})
    return [{"role": "user", "content": content}]


def make_scorer(m, a):
    """Load checker m ("G" Gemma, "Q" Qwen) with its adapter -> (p_fall(messages) -> (P(fall), mass), model)."""
    sys.path[:0] = [SPACE]
    import torch

    import gemma_binary_scoring as gs
    from gemma_scoring import to_probs
    if m == "G":
        from peft import PeftModel
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        from gemma_scoring import score_labels
        proc = AutoProcessor.from_pretrained(a.gemma_base)
        pre, plan, bad = gs.label_plan(proc.tokenizer)
        assert not bad, bad
        model = Gemma3ForConditionalGeneration.from_pretrained(a.gemma_base, dtype=torch.bfloat16,
                                                               device_map="cuda").eval()
        model = PeftModel.from_pretrained(model, a.gemma_adapter).merge_and_unload().eval()

        def p_fall(msgs):
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                probs, mass, _ = to_probs(score_labels(model, inp, pre, plan))
            return probs["fall"], mass
    else:
        from qwen_scoring import QwenScorer, score_labels as q_score
        qs = QwenScorer(a.qwen_base, a.qwen_adapter).load()
        model = qs.model

        def p_fall(msgs):
            inp = qs.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to(qs.model.device)
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            probs, mass, _ = to_probs(q_score(qs.model, inp, qs.pre, qs.plan))
            return probs["fall"], mass
    return p_fall, model


def score(a):
    sys.path[:0] = [SPACE]
    import torch
    from PIL import Image

    import gemma_binary_scoring as gs
    from gemma_scoring import to_probs
    events, r1 = load_round1()
    a.out.mkdir(parents=True, exist_ok=True)
    out_file = a.out / "round2.jsonl"
    good = [r for r in map(json.loads, open(out_file)) if "p_fall" in r] if out_file.exists() else []
    out_file.write_text("".join(json.dumps(r) + "\n" for r in good))
    done = {(r["event_id"], r["cam"], r["checker"], r["cond"]) for r in good}
    # control: no notes, on the first N events (checks round 1 is reproduced); peers: notes, on every pair
    jobs = [(e, cam, cond) for k, e in enumerate(events) for cam in sorted(e["clips"])
            for cond in (["control"] if k < a.control else []) + ["peers"]]

    for m in ("G", "Q"):
        todo = [(e, cam, cond) for e, cam, cond in jobs if (e["event_id"], cam, m, cond) not in done]
        print(f"[{CHECKERS[m]}] {len(todo)} to do", flush=True)
        if not todo:
            continue
        p_fall, model = make_scorer(m, a)

        with open(out_file, "a") as f:
            for n, (e, cam, cond) in enumerate(todo, 1):
                cid = e["clips"][cam]
                d = G_DIR / "clips" / cid
                frames = [Image.open(d / f"f{i:02d}.jpg").convert("RGB") for i in range(32)]
                notes = peer_notes(e, r1, cam) if cond == "peers" else ""
                p, mass = p_fall(build_content(gs, frames, notes))
                f.write(json.dumps({"event_id": e["event_id"], "cam": cam, "checker": m, "cond": cond,
                                    "clip_id": cid, "p1": r1[m][cid]["p_fall"], "p2": round(float(p), 5),
                                    "mass": round(float(mass), 4)}) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}", flush=True)
        del model
        torch.cuda.empty_cache()


def room_keeps(votes):
    """The app's decide(): keep the alarm if at least half of the (camera, checker) votes say fall."""
    return sum(votes) * 2 >= len(votes)


def report(a):
    events, r1 = load_round1()
    r2 = {}
    for r in map(json.loads, open(a.out / "round2.jsonl")):
        r2[(r["event_id"], r["cam"], r["checker"], r["cond"])] = r["p2"]

    ctl = [(k, p) for k, p in r2.items() if k[3] == "control"]
    if ctl:
        diffs = [abs(p - r1[k[2]][next(e for e in events if e["event_id"] == k[0])["clips"][k[1]]]["p_fall"])
                 for k, p in ctl]
        print(f"control (no notes) vs saved round 1: {len(diffs)} pairs, max |dP| = {max(diffs):.4f}\n")

    flips = Counter()
    rows = Counter()
    for e in events:
        kind = "fall" if e["kind"] == "hit" else "false alarm"
        pairs = [(cam, m) for cam in sorted(e["clips"]) for m in ("G", "Q")]
        if any((e["event_id"], cam, m, "peers") not in r2 for cam, m in pairs):
            continue
        v1 = {(cam, m): r1[m][e["clips"][cam]]["p_fall"] >= TAU for cam, m in pairs}
        v2 = {(cam, m): r2[(e["event_id"], cam, m, "peers")] >= TAU for cam, m in pairs}
        yes_cams = {cam for (cam, m), v in v1.items() if v}
        # safe rule: a "no" may become "yes" only if another camera said yes in round 1; a "yes" never changes
        vs = {k: v1[k] or (v2[k] and bool(yes_cams - {k[0]})) for k in pairs}
        for k in pairs:
            if v1[k] != v2[k]:
                flips[(e["split"], kind, m_name(k[1]), "no->yes" if v2[k] else "yes->no")] += 1
        for rule, votes in (("round 1 (now in the app)", v1), ("round 2, full exchange", v2),
                            ("round 2, safe (no->yes only)", vs)):
            rows[(e["split"], rule, kind)] += room_keeps(list(votes.values()))
        rows[(e["split"], "n", kind)] += 1

    print(f"Room rule: keep the alarm if >= half of the (camera, checker) votes say fall (tau = {TAU})")
    print(f"{'split':5} {'rule':30} {'falls kept':>11} {'false alarms kept':>18}")
    for split in ("val", "test"):
        for rule in ("round 1 (now in the app)", "round 2, full exchange", "round 2, safe (no->yes only)"):
            nf, na = rows[(split, "n", "fall")], rows[(split, "n", "false alarm")]
            print(f"{split:5} {rule:30} {rows[(split, rule, 'fall')]:>5}/{nf:<5} {rows[(split, rule, 'false alarm')]:>11}/{na}")
    print("\nVerdict changes from round 1 to round 2 (full exchange), per (camera, checker) vote:")
    for k, v in sorted(flips.items()):
        print(f"  {k[0]:5} {k[1]:12} {k[2]:7} {k[3]:8} {v}")
    if not flips:
        print("  none")


def m_name(m):
    return {"G": "Gemma", "Q": "Qwen"}[m]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "report"])
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/peer_round2"))
    ap.add_argument("--gemma-base", default="/scratch/wuhungma/models/gemma-3-12b-it")
    ap.add_argument("--gemma-adapter", default="/scratch/wuhungma/lora/r16_reasons_ep1/adapter")
    ap.add_argument("--qwen-base", default="/scratch/wuhungma/models/qwen2-vl-7b-instruct")
    ap.add_argument("--qwen-adapter", default="/scratch/wuhungma/models/qwen2vl-7b-fall-lora")
    ap.add_argument("--control", type=int, default=10, help="also score the first N events without notes")
    ap.add_argument("--q-dir", type=Path, default=Q_DIR, help="round-1 Qwen results (verify_qwen.py --out)")
    a = ap.parse_args()
    globals()["Q_DIR"] = a.q_dir  # used by load_round1()
    score(a) if a.cmd == "score" else report(a)


if __name__ == "__main__":
    main()
