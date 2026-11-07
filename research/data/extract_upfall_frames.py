#!/usr/bin/env python3
"""Extract 32 frames per camera per trial from the UP-Fall camera zips.

Mirrors the existing frames32 layout (frames32/<dataset>/<scenario>/<n>/camN_fXX.png):

    <out-root>/SubjectN/Activity<A>_Trial<T>/cam1_f00.png ... cam1_f31.png
                                             cam2_f00.png ... cam2_f31.png

Frames are sampled at the midpoints of 32 equal bins over the whole trial
(same convention as the edf/mcfd frame_times), and the PNG bytes are copied out
of the zip unchanged (640x480, lossless). Only the 32 members are read, not the
whole zip. A trial is written to a .tmp dir and renamed when complete, so a
killed run never leaves a half-written trial; re-running skips finished trials.

Trial list and splits come from splits/up_fall_{train,val,test}.jsonl
(written by split_upfall.py). Also writes splits/up_fall_frame_index.jsonl with
the source frame names (their timestamps line up with the sensor CSV).

Default --out-root is on /scratch because /project is nearly full (~18 GB needed).

Usage:
    python3 extract_upfall_frames.py --workers 8
    python3 extract_upfall_frames.py --limit 4 --out-root /tmp/test_frames
"""
import argparse
import json
import shutil
import sys
import zipfile
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
N_FRAMES = 32


def pick_indices(n, k=N_FRAMES):
    return [int((i + 0.5) * n / k) for i in range(k)]


def process_trial(job):
    row, out_root = job
    dest = Path(out_root) / row["scenario"] / f"Activity{row['activity']}_Trial{row['trial']}"
    tmp = dest.with_name(dest.name + ".tmp")
    cams = [c for c in (1, 2) if row[f"has_cam{c}"]]
    expected = {f"cam{c}_f{i:02d}.png" for c in cams for i in range(N_FRAMES)}
    done = dest.is_dir() and {p.name for p in dest.iterdir()} == expected

    if not done:
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)

    info = {"event_id": row["event_id"], "split": row["split"], "dir": str(dest), "cams": {}}
    for c in cams:
        with zipfile.ZipFile(row[f"cam{c}_zip"]) as z:
            names = sorted(i.filename for i in z.infolist() if i.filename.lower().endswith(".png"))
            if len(names) < N_FRAMES:
                raise RuntimeError(f"{row['event_id']} cam{c}: only {len(names)} frames")
            picked = [names[i] for i in pick_indices(len(names))]
            if not done:
                for i, name in enumerate(picked):
                    (tmp / f"cam{c}_f{i:02d}.png").write_bytes(z.read(name))  # z.read checks the CRC
        info["cams"][str(c)] = {"n_frames_total": len(names), "frames": picked}
    if not done:
        tmp.rename(dest)
    info["skipped"] = done
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits-dir", type=Path, default=HERE.parent / "splits")
    ap.add_argument("--out-root", type=Path, default=Path("/scratch/wuhungma/frames32/up_fall"))
    ap.add_argument("--index", type=Path, default=HERE.parent / "splits" / "up_fall_frame_index.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N trials (for testing)")
    args = ap.parse_args()

    rows = []
    for s in SPLITS:
        with open(args.splits_dir / f"up_fall_{s}.jsonl") as f:
            rows += [json.loads(l) for l in f]
    rows.sort(key=lambda r: (r["subject"], r["activity"], r["trial"]))
    if args.limit:
        rows = rows[: args.limit]

    args.out_root.mkdir(parents=True, exist_ok=True)
    results, skipped = [], 0
    with Pool(args.workers) as pool:
        for k, info in enumerate(pool.imap_unordered(process_trial, [(r, str(args.out_root)) for r in rows]), 1):
            results.append(info)
            skipped += info["skipped"]
            if k % 25 == 0 or k == len(rows):
                print(f"{k}/{len(rows)} trials ({skipped} already done)", flush=True)

    results.sort(key=lambda r: r["event_id"])
    with open(args.index, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"done: {len(results)} trials -> {args.out_root}\nindex: {args.index}")


if __name__ == "__main__":
    sys.exit(main())
