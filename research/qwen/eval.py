"""Metrics for a run directory written by run_probs_qwen.py (or group12's run_probs.py: same format).

    python eval.py runs/qwen_k32
    python eval.py runs/qwen_bin_k32 --threshold 0.5

10-class runs: accuracy / macro-F1 over the 10 labels, plus the binary view fall = P(fall) + P(fallen).
Binary runs: p_fall = probs["fall"]. Gold fall = label in {fall, fallen}. Broken down by dataset and split_cs.
"""
import argparse, glob, json
from collections import Counter, defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--threshold", type=float, default=0.5, help="predict fall if p_fall >= this")
a = ap.parse_args()

recs, errors = {}, 0
for fn in sorted(glob.glob(f"{a.run_dir}/results_shard*.jsonl")):
    for l in open(fn):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue
        if "error" in r: errors += 1; continue
        recs[(r["event_id"], r["cam"])] = r                    # last good record wins
recs = list(recs.values())
if not recs: raise SystemExit(f"no finished records in {a.run_dir}")
binary_run = set(recs[0]["probs"]) == {"fall", "no_fall"}
FALL = {"fall", "fallen"}


def p_fall(r):
    return r["probs"]["fall"] if binary_run else r["probs"]["fall"] + r["probs"]["fallen"]


def binary(rows):
    tp = sum(1 for r in rows if r["gold"] in FALL and p_fall(r) >= a.threshold)
    fn = sum(1 for r in rows if r["gold"] in FALL and p_fall(r) < a.threshold)
    fp = sum(1 for r in rows if r["gold"] not in FALL and p_fall(r) >= a.threshold)
    tn = len(rows) - tp - fn - fp
    d = lambda n, m: n / m if m else 0.0
    prec, rec = d(tp, tp + fp), d(tp, tp + fn)
    return f"TP={tp} FP={fp} TN={tn} FN={fn}  acc={d(tp + tn, len(rows)):.3f} prec={prec:.3f} rec={rec:.3f} " \
           f"f1={d(2 * prec * rec, prec + rec):.3f} spec={d(tn, tn + fp):.3f}"


def multiclass(rows):
    labels = sorted({r["gold"] for r in rows} | {r["argmax"] for r in rows})
    acc = sum(r["argmax"] == r["gold"] for r in rows) / len(rows)
    f1s = []
    for c in labels:
        tp = sum(r["argmax"] == c and r["gold"] == c for r in rows)
        fp = sum(r["argmax"] == c and r["gold"] != c for r in rows)
        fn = sum(r["argmax"] != c and r["gold"] == c for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return f"acc={acc:.3f} macroF1={sum(f1s) / len(f1s):.3f}"


groups = defaultdict(list)
for r in recs:
    groups["ALL"].append(r); groups[r["dataset"]].append(r); groups[f"{r['dataset']}/{r['split_cs']}"].append(r)
print(f"{a.run_dir}: {len(recs)} clips ({'binary' if binary_run else '10-class'} run), {errors} error lines, "
      f"mean label mass {sum(r['mass'] for r in recs) / len(recs):.3f}")
for g in sorted(groups, key=lambda g: (g != "ALL", g)):
    rows = groups[g]
    print(f"\n{g}  n={len(rows)}")
    if not binary_run: print("  10-class :", multiclass(rows))
    print(f"  fall@{a.threshold}:", binary(rows))
if not binary_run:
    print("\nconfusion (gold -> predicted, ALL):")
    for gold in sorted({r["gold"] for r in recs}):
        c = Counter(r["argmax"] for r in recs if r["gold"] == gold)
        print(f"  {gold:9s} " + ", ".join(f"{k}:{v}" for k, v in c.most_common()))
