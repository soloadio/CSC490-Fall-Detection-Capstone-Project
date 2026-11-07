#!/usr/bin/env python3
"""Split the UP-Fall dataset (/scratch/blt/up_fall) into train/val/test by subject.

UP-Fall has 17 subjects x 11 activities x 3 trials. Trials of one subject are
near-duplicates (same person, same room, same clothes), so whole subjects are
assigned to a split. Every subject performs the same 33 trials, so any
subject-level split gives identical activity/label proportions in each split.

With 17 subjects 70/15/15 is not exactly reachable; the default is 12/2/3
subjects (70.6% / 11.8% / 17.6%). Override with --counts.

Outputs:
    splits/up_fall_{train,val,test}.jsonl   one row per trial, with "split"
    up_fall_split/<split>/SubjectN          symlink -> <up_fall root>/SubjectN

Nothing under the up_fall root is modified. Re-runnable (stale links removed).

Usage:
    python3 split_upfall.py
    python3 split_upfall.py --seed 7 --counts 11 3 3
"""
import argparse
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")

# From the UP-Fall paper: activities 1-5 are falls, 6-11 are daily activities.
ACTIVITIES = {
    1: "fall_forward_hands", 2: "fall_forward_knees", 3: "fall_backward",
    4: "fall_sideward", 5: "fall_sitting_chair", 6: "walking", 7: "standing",
    8: "sitting", 9: "pick_up_object", 10: "jumping", 11: "laying",
}


def scan_trials(root):
    trials = []
    for sub in sorted(root.glob("Subject*"), key=lambda p: int(p.name[7:])):
        for act in sorted(sub.glob("Activity*"), key=lambda p: int(p.name[8:])):
            for trial in sorted(act.glob("Trial*"), key=lambda p: int(p.name[5:])):
                pre = f"{sub.name}{act.name}{trial.name}"
                a = int(act.name[8:])
                trials.append({
                    "event_id": f"up_fall/{sub.name}/{act.name}/{trial.name}",
                    "dataset": "up_fall",
                    "scenario": sub.name,
                    "subject": int(sub.name[7:]),
                    "activity": a,
                    "label_name": ACTIVITIES[a],
                    "is_fall": a <= 5,
                    "trial": int(trial.name[5:]),
                    "dir": str(trial),
                    "sensor_csv": str(trial / f"{pre}.csv"),
                    "cam1_zip": str(trial / f"{pre}Camera1.zip"),
                    "cam2_zip": str(trial / f"{pre}Camera2.zip"),
                    "has_cam1": (trial / f"{pre}Camera1.zip").exists(),
                    "has_cam2": (trial / f"{pre}Camera2.zip").exists(),
                })
    return trials


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/scratch/blt/up_fall"))
    ap.add_argument("--splits-dir", type=Path, default=HERE.parent / "splits")
    ap.add_argument("--link-dir", type=Path, default=HERE.parent / "up_fall_split")
    ap.add_argument("--counts", nargs=3, type=int, metavar=("TRAIN", "VAL", "TEST"),
                    help="number of subjects per split (default: 70%% rounded, rest halved, test gets the odd one)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    trials = scan_trials(args.root)
    if not trials:
        raise SystemExit(f"no Subject*/Activity*/Trial* found under {args.root}")
    subjects = sorted({t["subject"] for t in trials})
    n = len(subjects)

    if args.counts:
        counts = args.counts
        if sum(counts) != n or min(counts) < 1:
            raise SystemExit(f"--counts must be >=1 each and sum to {n} subjects, got {counts}")
    else:
        n_train = round(n * 0.70)
        n_val = (n - n_train) // 2
        counts = [n_train, n_val, n - n_train - n_val]

    order = subjects[:]
    random.Random(args.seed).shuffle(order)
    subject_split = {}
    i = 0
    for split, c in zip(SPLITS, counts):
        for s in order[i:i + c]:
            subject_split[s] = split
        i += c

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    handles = {s: open(args.splits_dir / f"up_fall_{s}.jsonl", "w") for s in SPLITS}
    try:
        for t in trials:
            t["split"] = subject_split[t["subject"]]
            handles[t["split"]].write(json.dumps(t) + "\n")
    finally:
        for h in handles.values():
            h.close()

    # symlinks: up_fall_split/<split>/SubjectN -> <root>/SubjectN
    desired = {args.link_dir / subject_split[s] / f"Subject{s}": args.root / f"Subject{s}" for s in subjects}
    removed = 0
    if args.link_dir.exists():
        for link in args.link_dir.glob("*/Subject*"):
            if link.is_symlink() and link not in desired:
                link.unlink()
                removed += 1
    for link, src in desired.items():
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            if os.readlink(link) == str(src):
                continue
            link.unlink()
        elif link.exists():
            raise SystemExit(f"refusing to overwrite non-symlink: {link}")
        os.symlink(src, link)

    print(f"up_fall: {len(trials)} trials, {n} subjects, seed {args.seed}")
    for split in SPLITS:
        subs = sorted(s for s, sp in subject_split.items() if sp == split)
        nt = sum(1 for t in trials if t["split"] == split)
        nf = sum(1 for t in trials if t["split"] == split and t["is_fall"])
        print(f"  {split:5s} {len(subs):2d} subjects, {nt:3d} trials ({nt / len(trials):5.1%}), "
              f"{nf} fall / {nt - nf} ADL   subjects: {', '.join(map(str, subs))}")
    no_cam = [t["event_id"] for t in trials if not (t["has_cam1"] and t["has_cam2"])]
    if no_cam:
        print(f"  trials missing a camera zip ({len(no_cam)}): {', '.join(no_cam)}")
    print(f"Wrote {args.splits_dir}/up_fall_{{train,val,test}}.jsonl and links in {args.link_dir} "
          f"({removed} stale links removed)")


if __name__ == "__main__":
    main()
