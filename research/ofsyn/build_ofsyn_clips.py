#!/usr/bin/env python3
"""OF-Syn (OmniFall synthetic videos) -> 32-frame clips in the frames32 manifest format, to add lie-down examples.

Why: the checkers were trained on 580 fall clips but only 26 lie-down clips and call every lie-down a fall
(research/verify/speed_test.py). OF-Syn has ~1,200 synthetic lie-down videos (and 1,200 falls, kept so that "looks
synthetic" does not become a shortcut for "not a fall").

Kept videos: adults (18+) and static wide shots, closest to a room camera (many others are close-ups of children).
Clips: every "fall" (1) and "lie_down" (5) segment; 32 frames at the midpoints of 32 equal bins over [start, end]
(same convention as build_upfall_events.py), resized to 640x360 PNG. Split: OmniFall's own syn/random split.

    python3 build_ofsyn_clips.py --videos /scratch/wuhungma/ofsyn --out /scratch/wuhungma/ofsyn   (Slurm, needs ffmpeg)
"""
import argparse
import csv
import json
import os
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

OMNI = Path("/scratch/wuhungma/omnifall")
ADULT = {"young_adults_18_34", "middle_aged_35_64", "elderly_65_plus"}
KEEP = {"1": "fall", "5": "lie_down"}
N, MIN_DURATION = 32, 0.1


def clip(job):
    video, segs, out = job
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["ffmpeg", "-v", "error", "-c:v", os.environ.get("AV1_DECODER", "libdav1d"), "-i", video, "-vf", "scale=640:360", f"{tmp}/%04d.png"], check=True)
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=r_frame_rate", "-of", "csv=p=0", video], capture_output=True, text=True)
        a, b = probe.stdout.strip().split("/")
        fps = float(a) / float(b)
        frames = sorted(Path(tmp).glob("*.png"))
        for k, (start, end, label, meta) in enumerate(segs):
            dest = Path(out) / meta["path"] / f"seg{k}"
            dest.mkdir(parents=True, exist_ok=True)
            idx = [min(len(frames) - 1, round((start + (i + 0.5) * (end - start) / N) * fps)) for i in range(N)]
            paths = []
            for i, j in enumerate(idx):
                p = dest / f"cam1_f{i:02d}.png"
                p.write_bytes(frames[j].read_bytes())
                paths.append(str(p))
            rows.append({"event_id": f"of_syn/{meta['path']}/seg{k}", "dataset": "of_syn", "label": int(label),
                         "label_name": KEEP[label], "start": start, "end": end, "split": meta["split"],
                         "age_group": meta["age_group"], "environment": meta["environment_category"],
                         "cams": {"1": {"frames": paths, "n_distinct": len(set(idx)), "window_clipped_s": 0.0}}})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    split = {}
    for sp in ("train", "val", "test"):
        for r in csv.DictReader(open(OMNI / "splits/syn/random" / f"{sp}.csv")):
            split[r["path"]] = sp
    segs = {}
    for r in csv.DictReader(open(OMNI / "labels/of-syn.csv")):
        if (r["label"] in KEEP and r["path"].split("/")[0] in KEEP.values() and r["age_group"] in ADULT
                and r["camera_shot"] == "static_wide" and float(r["end"]) - float(r["start"]) >= MIN_DURATION):
            segs.setdefault(r["path"], []).append((float(r["start"]), float(r["end"]), r["label"],
                                                   {**r, "split": split.get(r["path"], "train")}))
    jobs = [(str(a.videos / f"{p}.mp4"), s, str(a.out / "frames32")) for p, s in segs.items()
            if (a.videos / f"{p}.mp4").exists()]
    print(f"{len(jobs)} videos, {sum(len(j[1]) for j in jobs)} clips", flush=True)
    with Pool(a.workers) as pool:
        rows = [r for rs in pool.imap_unordered(clip, jobs, chunksize=4) for r in rs]
    for sp in ("train", "val", "test"):
        sel = [r for r in rows if r["split"] == sp]
        (a.out / f"frames32_{sp}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
        n = {k: sum(r["label_name"] == k for r in sel) for k in KEEP.values()}
        print(sp, n, flush=True)


if __name__ == "__main__":
    main()
