#!/usr/bin/env python3
"""Write one run_probs.py-compatible frame manifest per split, for edf + mcfd + up_fall.

    splits/frames32_{train,val,test}.jsonl

edf/mcfd rows come from splits/{split}.jsonl, with frames in frames32/<event_id>/camC_fXX.png
(a camera is included only if all 32 frames exist). up_fall rows come from
splits/up_fall_events_{split}.jsonl, which already list their frames (run
build_upfall_events.py --extract <split> first).

Usage:
    python3 make_frames_manifest.py            # all three splits
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAMES = HERE.parent / "frames32"
N_FRAMES = 32


def edf_mcfd_rows(split):
    with open(HERE.parent / "splits" / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            cams = {}
            for c in r["cams"]:
                paths = [FRAMES / r["event_id"] / f"cam{c}_f{i:02d}.png" for i in range(N_FRAMES)]
                if all(p.exists() for p in paths):
                    cams[c] = {"frames": [str(p) for p in paths], "n_distinct": N_FRAMES, "window_clipped_s": 0.0}
            if cams:
                yield {**{k: v for k, v in r.items() if k != "cams"}, "cams": cams}


def upfall_rows(split):
    with open(HERE.parent / "splits" / f"up_fall_events_{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if all(Path(p).exists() for c in r["cams"].values() for p in c["frames"]):
                yield r


def main():
    for split in ("train", "val", "test"):
        rows = list(edf_mcfd_rows(split))
        n_up = 0
        for r in upfall_rows(split):
            rows.append(r)
            n_up += 1
        with open(HERE.parent / "splits" / f"frames32_{split}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        clips = sum(len(r["cams"]) for r in rows)
        print(f"{split:5s} {len(rows):5d} events ({len(rows) - n_up} edf/mcfd, {n_up} up_fall), {clips} camera clips")


if __name__ == "__main__":
    main()
