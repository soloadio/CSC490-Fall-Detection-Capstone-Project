#!/usr/bin/env python3
"""VideoMAE on the replayed recordings at the deployed setting: falls caught, delay, and what the person was doing at
each false alarm (lying down, sitting down, picking something up, ...). Compare several eval_stream.py outputs.

    python3 activity_report.py /scratch/wuhungma/stream_eval/videomae_fallonly /scratch/wuhungma/stream_eval/<new>
"""
import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.argv, _argv = sys.argv[:1], sys.argv  # stream_metrics parses nothing at import, but be safe
import stream_metrics as S  # noqa: E402

sys.argv = _argv
NAMES = {"0": "walk", "1": "fall", "2": "fallen", "3": "sit_down", "4": "sitting", "5": "lie_down", "6": "lying",
         "7": "stand_up", "8": "standing", "9": "other"}
ORDER = ["lie_down", "lying", "sit_down", "sitting", "other", "stand_up", "walk", "standing"]


def run(d, split, w="3.0", th=0.30, k=1, cooldown=10.0, pre=0.5, post=2.0, gap=1.0):
    recs = S.load_scores(Path(d) / f"scores_{split}.jsonl")
    truth = S.load_truth(recs)
    falls = hits = 0
    delays, fa, up11 = [], collections.Counter(), 0
    for r in recs:
        segs = truth.get(r["video"], [])
        eps = S.episodes(segs, gap)
        fl = [e for e in eps if e["onset"] is not None]
        falls += len(fl)
        win = r["windows"][w]
        hit = set()
        for x in S.alarms(win["t"], win["p"], th, k, cooldown):
            m = [i for i, e in enumerate(fl) if e["onset"] - pre <= x <= e["end"] + post]
            if m:
                if m[0] not in hit:
                    hit.add(m[0])
                    delays.append(x - fl[m[0]]["onset"])
                continue
            if any(e["start"] - pre <= x <= e["end"] + post for e in eps):
                continue
            cur = [NAMES[l] for s, e, l in segs if s <= x <= e]
            if cur:
                fa[cur[0]] += 1
                up11 += "Activity11" in r["video"]
        hits += len(hit)
    return falls, hits, float(np.median(delays)) if delays else float("nan"), fa, up11


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    print("deployed setting: 3 s window, k = 1, threshold 0.30, cooldown 10 s")
    print(f"{'model':28} {'split':5} {'falls caught':>13} {'delay':>6} {'false alarms':>13} | " + " ".join(f"{o:>8}" for o in ORDER) + " | UP-Fall laying")
    for d in a.dirs:
        for sp in ("val", "test"):
            if not (Path(d) / f"scores_{sp}.jsonl").exists():
                continue
            falls, hits, dl, fa, up11 = run(d, sp)
            print(f"{Path(d).name:28} {sp:5} {hits:>6}/{falls:<6} {dl:5.1f}s {sum(fa.values()):>13} | "
                  + " ".join(f"{fa[o]:>8}" for o in ORDER) + f" | {up11}")


if __name__ == "__main__":
    main()
