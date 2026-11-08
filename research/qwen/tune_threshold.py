"""Choose the P(fall) threshold on val, then report val and test at that threshold.

Same events, camera vote and metric definitions as group12/eval_binary.py (which only uses 0.5).
One threshold for all datasets: the one that maximises the mean camera-vote balanced accuracy over
the datasets on val. Test is never used to choose it.

    python3 tune_threshold.py runs/qwen_bin_k32/results_shard*.jsonl
"""
import argparse, json
from collections import defaultdict

GROUP = "/project/def-ajbonner/group12"
POS = {"fall", "fallen"}

ap = argparse.ArgumentParser()
ap.add_argument("results", nargs="+")
a = ap.parse_args()


def split_events(split):
    ids = set()
    for p in (f"{GROUP}/splits/{split}.jsonl", f"{GROUP}/splits/up_fall_events_{split}.jsonl"):
        ids |= {json.loads(l)["event_id"] for l in open(p)}
    return ids


rows = {}
for p in a.results:
    for l in open(p):
        r = json.loads(l)
        if "error" not in r:
            rows[(r["event_id"], r["cam"])] = r


def votes(split):
    """dataset -> (camera-vote scores, labels) for the events of this split."""
    keep, ev = split_events(split), defaultdict(lambda: defaultdict(list))
    for r in rows.values():
        if r["event_id"] in keep:
            ev[r["dataset"]][r["event_id"]].append(r)
    return {ds: ([sum(r["probs"]["fall"] for r in v) / len(v) for v in e.values()], [v[0]["gold"] in POS for v in e.values()])
            for ds, e in ev.items()}


def metrics(s, y, t):
    tp = sum(x > t and g for x, g in zip(s, y)); fp = sum(x > t and not g for x, g in zip(s, y))
    fn = sum(g and not x > t for x, g in zip(s, y)); tn = len(y) - tp - fp - fn
    rec, spec = tp / (tp + fn), tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(n=len(y), bal_acc=(rec + spec) / 2, recall=rec, precision=prec, f1=2 * prec * rec / (prec + rec) if tp else 0.0)


val, test = votes("val"), votes("test")
cands = sorted({round(x, 4) for s, _ in val.values() for x in s} | {0.5})
score = lambda t: sum(metrics(s, y, t)["bal_acc"] for s, y in val.values()) / len(val)
best = max(cands, key=lambda t: (score(t), -abs(t - 0.5)))      # ties: stay closest to 0.5
print(f"chosen on val: threshold {best:.4f} (mean val bal_acc {score(best):.1%}; at 0.5: {score(0.5):.1%})\n")
print(f"{'split':5s} {'dataset':8s} {'thr':>6s} {'n':>4s} {'bal_acc':>8s} {'recall':>7s} {'prec':>7s} {'F1':>7s}")
for name, d in (("val", val), ("test", test)):
    for ds in sorted(d):
        for t in (0.5, best):
            m = metrics(*d[ds], t)
            print(f"{name:5s} {ds:8s} {t:6.3f} {m['n']:4d} {m['bal_acc']:8.1%} {m['recall']:7.1%} {m['precision']:7.1%} {m['f1']:7.1%}")
