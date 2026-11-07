#!/usr/bin/env python3
"""Split the edf and mcfd events in manifest.jsonl into train/val/test (70/15/15).

Each dataset is split on its own, then the results are merged, so every dataset
ends up close to 70/15/15 and the combined train/val/test files contain both.

--unit scenario (default)
    Whole scenarios (edf subject / mcfd chute) go to a single split. Events from
    the same recording are strongly correlated (adjacent windows of one video),
    so splitting by event leaks near-duplicates between train and test. The
    search picks the scenario assignment whose event counts AND per-label
    counts are closest to the target ratios.

--unit event
    Random split of individual events, stratified by label. Gives exact ratios
    but the splits share recordings, so test scores will be optimistic.

Outputs (in --out-dir): train.jsonl, val.jsonl, test.jsonl. Each line is the
original manifest row plus a "split" field. frames32/<dataset>/<scenario>/<n>
can be looked up from event_id ("<dataset>/<scenario>/<n>").

Usage:
    python3 split_dataset.py
    python3 split_dataset.py --seed 7 --unit event
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")


def load_manifest(path, datasets):
    rows = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["dataset"] in datasets:
                rows.append(row)
    return rows


def split_cost(counts, label_counts, totals, label_totals, ratios):
    """Weighted L1 distance from the target ratios (0 = perfect)."""
    n = sum(totals)
    cost = sum(abs(counts[s] / n - ratios[s]) for s in range(3))
    for label, lt in label_totals.items():
        dev = sum(abs(label_counts[s][label] / lt - ratios[s]) for s in range(3))
        cost += (lt / n) * dev
    return cost


def split_by_scenario(rows, ratios, rng, restarts=300):
    groups = defaultdict(list)
    for r in rows:
        groups[r["scenario"]].append(r)
    names = sorted(groups)
    if len(names) < 3:
        raise SystemExit(f"need at least 3 scenarios to split by scenario, got {names}")

    sizes = [len(groups[g]) for g in names]
    g_labels = [Counter(r["label_name"] for r in groups[g]) for g in names]
    label_totals = Counter(r["label_name"] for r in rows)
    n = len(rows)
    r_list = [ratios[s] for s in SPLITS]

    def evaluate(assign):
        counts = [0, 0, 0]
        lcounts = [Counter(), Counter(), Counter()]
        for i, s in enumerate(assign):
            counts[s] += sizes[i]
            lcounts[s].update(g_labels[i])
        if min(counts) == 0:
            return float("inf")
        return split_cost(counts, lcounts, [n], label_totals, r_list)

    best, best_cost = None, float("inf")
    for _ in range(restarts):
        assign = [rng.randrange(3) for _ in names]
        cost = evaluate(assign)
        improved = True
        while improved:  # hill-climb: single moves, then pairwise swaps
            improved = False
            for i in range(len(names)):
                for s in range(3):
                    if s == assign[i]:
                        continue
                    old, assign[i] = assign[i], s
                    c = evaluate(assign)
                    if c < cost - 1e-12:
                        cost, improved = c, True
                    else:
                        assign[i] = old
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    if assign[i] == assign[j]:
                        continue
                    assign[i], assign[j] = assign[j], assign[i]
                    c = evaluate(assign)
                    if c < cost - 1e-12:
                        cost, improved = c, True
                    else:
                        assign[i], assign[j] = assign[j], assign[i]
        if cost < best_cost - 1e-12:
            best, best_cost = list(assign), cost

    scenario_split = {g: SPLITS[best[i]] for i, g in enumerate(names)}
    return {r["event_id"]: scenario_split[r["scenario"]] for r in rows}


def split_by_event(rows, ratios, rng):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label_name"]].append(r["event_id"])
    out = {}
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        n_train = round(len(ids) * ratios["train"])
        n_val = round(len(ids) * ratios["val"])
        for i, eid in enumerate(ids):
            out[eid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out


def print_summary(dataset, rows, assignment):
    n = len(rows)
    print(f"\n== {dataset}: {n} events ==")
    by_split = {s: [r for r in rows if assignment[r["event_id"]] == s] for s in SPLITS}
    for s in SPLITS:
        scen = sorted({r["scenario"] for r in by_split[s]})
        print(f"  {s:5s} {len(by_split[s]):4d} ({len(by_split[s]) / n:6.1%})  scenarios: {', '.join(scen)}")
    labels = sorted({r["label_name"] for r in rows})
    print(f"  {'label':10s}" + "".join(f"{s:>7s}" for s in SPLITS))
    for label in labels:
        c = [sum(r["label_name"] == label for r in by_split[s]) for s in SPLITS]
        flag = "  <-- missing from a split" if 0 in c else ""
        print(f"  {label:10s}" + "".join(f"{x:7d}" for x in c) + flag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=HERE / "manifest.jsonl")
    ap.add_argument("--out-dir", type=Path, default=HERE / "splits")
    ap.add_argument("--datasets", nargs="+", default=["edf", "mcfd"])
    ap.add_argument("--unit", choices=["scenario", "event"], default="scenario")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit(f"ratios must sum to 1, got {sum(args.ratios)}")
    ratios = dict(zip(SPLITS, args.ratios))

    rows = load_manifest(args.manifest, set(args.datasets))
    if not rows:
        raise SystemExit(f"no rows for datasets {args.datasets} in {args.manifest}")

    assignment = {}
    for i, dataset in enumerate(args.datasets):
        ds_rows = [r for r in rows if r["dataset"] == dataset]
        if not ds_rows:
            print(f"warning: no events for dataset '{dataset}'")
            continue
        rng = random.Random(f"{args.seed}:{dataset}")  # independent of dataset order
        split_fn = split_by_scenario if args.unit == "scenario" else split_by_event
        ds_assign = split_fn(ds_rows, ratios, rng)
        assignment.update(ds_assign)
        print_summary(dataset, ds_rows, ds_assign)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: open(args.out_dir / f"{s}.jsonl", "w") for s in SPLITS}
    try:
        for r in rows:
            r = {**r, "split": assignment[r["event_id"]]}
            handles[r["split"]].write(json.dumps(r) + "\n")
    finally:
        for h in handles.values():
            h.close()

    total = Counter(assignment.values())
    print(f"\nCombined: " + ", ".join(f"{s} {total[s]} ({total[s] / len(rows):.1%})" for s in SPLITS))
    print(f"Wrote {', '.join(f'{s}.jsonl' for s in SPLITS)} to {args.out_dir}")


if __name__ == "__main__":
    main()
