#!/usr/bin/env python3
"""Arrange frames32/ into train/val/test folders using symlinks.

Reads splits/{train,val,test}.jsonl (split_dataset.py) and, when present,
splits/up_fall_{train,val,test}.jsonl (split_upfall.py), and creates

    frames32_split/<split>/<dataset>/<scenario>/<n>  ->  ../../../../frames32/<dataset>/<scenario>/<n>

up_fall frames live on /scratch (--upfall-frames-dir), with n = Activity<A>_Trial<T>.

Nothing under frames32/ is moved, copied or modified. Links are relative, so
they survive the /project path aliases. Safe to re-run: missing links are
added, links that no longer match the split files are removed, and events whose
frames have not been extracted yet are skipped and reported.

Usage:
    python3 link_split_frames.py
    python3 link_split_frames.py --dry-run
"""
import argparse
import json
import os
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=HERE / "splits")
    ap.add_argument("--frames-dir", type=Path, default=HERE / "frames32")
    ap.add_argument("--upfall-frames-dir", type=Path, default=Path("/scratch/wuhungma/frames32/up_fall"),
                    help="where extract_upfall_frames.py wrote the up_fall frames")
    ap.add_argument("--out-dir", type=Path, default=HERE / "frames32_split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # desired link path (relative to out-dir) -> event dir
    desired = {}
    missing = Counter()
    present = Counter()
    frames_roots = {"up_fall": args.upfall_frames_dir}
    for split in SPLITS:
        for name in (f"{split}.jsonl", f"up_fall_{split}.jsonl"):
            path = args.splits_dir / name
            if not path.exists():
                continue
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    # edf/mcfd: dataset/scenario/n; up_fall: up_fall/SubjectN/ActivityA/TrialT -> ActivityA_TrialT
                    dataset, scenario, *rest = r["event_id"].split("/")
                    n = "_".join(rest)
                    src = frames_roots.get(dataset, args.frames_dir / dataset) / scenario / n
                    if not src.is_dir():
                        missing[(dataset, split)] += 1
                        continue
                    present[(dataset, split)] += 1
                    desired[Path(split) / dataset / scenario / n] = src

    # drop symlinks that no longer match (e.g. after re-splitting with a new seed)
    stale = []
    if args.out_dir.exists():
        for dirpath, dirnames, _ in os.walk(args.out_dir):
            for name in list(dirnames):
                p = Path(dirpath) / name
                if p.is_symlink():
                    dirnames.remove(name)  # don't descend into links
                    if p.relative_to(args.out_dir) not in desired:
                        stale.append(p)
    for p in stale:
        if not args.dry_run:
            p.unlink()

    created = 0
    for rel, src in desired.items():
        link = args.out_dir / rel
        target = os.path.relpath(src, link.parent)
        if link.is_symlink():
            if os.readlink(link) == target:
                continue
            if not args.dry_run:
                link.unlink()
        elif link.exists():
            raise SystemExit(f"refusing to overwrite non-symlink: {link}")
        if not args.dry_run:
            link.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(target, link)
        created += 1

    tag = " (dry run)" if args.dry_run else ""
    print(f"linked {created} new, removed {len(stale)} stale{tag}; out dir: {args.out_dir}")
    print(f"{'dataset':8s}" + "".join(f"{s:>16s}" for s in SPLITS) + "   (linked / not yet extracted)")
    for dataset in sorted({d for d, _ in present | missing}):
        cells = [f"{present[(dataset, s)]:>6d} / {missing[(dataset, s)]:<6d}" for s in SPLITS]
        print(f"{dataset:8s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    main()
