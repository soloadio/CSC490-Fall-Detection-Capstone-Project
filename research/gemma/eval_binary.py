#!/usr/bin/env python3
"""Fall / no-fall metrics for Gemma result files, per dataset, on one official split.

Each result row needs event_id, dataset, cam, gold, probs.fall (run_probs.py output).
Rows are kept only if their event is in the requested split of the official manifests
(splits/{split}.jsonl for edf/mcfd, splits/up_fall_events_{split}.jsonl for up_fall).

"single camera" scores every camera clip on its own; "camera vote" averages P(fall)
over an event's cameras and makes one decision per event. Threshold 0.5.

Usage:
    python3 eval_binary.py --split val  RESULTS.jsonl [RESULTS.jsonl ...]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
POS = {"fall", "fallen"}


def split_events(split):
    ids = set()
    for p in (HERE.parent / "splits" / f"{split}.jsonl", HERE.parent / "splits" / f"up_fall_events_{split}.jsonl"):
        with open(p) as f:
            ids |= {json.loads(l)["event_id"] for l in f}
    return ids


def auroc(scores, labels):
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = (i + j + 1) / 2
        i = j
    n1 = sum(labels)
    n0 = len(labels) - n1
    return (sum(r for r, y in zip(ranks, labels) if y) - n1 * (n1 + 1) / 2) / (n1 * n0)


def metrics(scores, labels):
    pred = [s > 0.5 for s in scores]
    tp = sum(p and y for p, y in zip(pred, labels))
    fp = sum(p and not y for p, y in zip(pred, labels))
    fn = sum(y and not p for p, y in zip(pred, labels))
    tn = len(labels) - tp - fp - fn
    rec = tp / (tp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    return {
        "n": len(labels),
        "bal_acc": (rec + tn / (tn + fp)) / 2,
        "recall": rec,
        "precision": prec,
        "f1": 2 * prec * rec / (prec + rec) if tp else 0.0,
        "auroc": auroc(scores, labels),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args()

    keep = split_events(args.split)
    rows = {}
    for p in args.results:
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if "error" not in r and r["event_id"] in keep:
                    rows[(r["event_id"], r["cam"])] = r  # dedupe across shard/merged files

    by_ds = defaultdict(list)
    for r in rows.values():
        by_ds[r["dataset"]].append(r)

    print(f"split={args.split}  threshold=0.5")
    print(f"{'dataset':8s} {'method':14s} {'n':>5s} {'bal_acc':>8s} {'recall':>7s} {'prec':>7s} {'F1':>7s} {'AUROC':>6s}")
    for ds in sorted(by_ds):
        cams = by_ds[ds]
        events = defaultdict(list)
        for r in cams:
            events[r["event_id"]].append(r)
        groups = {
            "single camera": ([r["probs"]["fall"] for r in cams], [r["gold"] in POS for r in cams]),
            "camera vote": ([sum(r["probs"]["fall"] for r in v) / len(v) for v in events.values()],
                            [v[0]["gold"] in POS for v in events.values()]),
        }
        for name, (s, y) in groups.items():
            m = metrics(s, y)
            print(f"{ds:8s} {name:14s} {m['n']:5d} {m['bal_acc']:8.1%} {m['recall']:7.1%} "
                  f"{m['precision']:7.1%} {m['f1']:7.1%} {m['auroc']:6.3f}")


if __name__ == "__main__":
    main()
