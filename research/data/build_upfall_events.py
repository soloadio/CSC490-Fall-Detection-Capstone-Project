#!/usr/bin/env python3
"""Turn UP-Fall trials into labelled event windows (like edf/mcfd) using OmniFall's segment labels.

OmniFall labels/up_fall.csv gives, per trial and camera, segments (start, end, label) in seconds from
the first frame, with the same 10 labels as edf/mcfd. Both cameras share identical segments, so one
event = one segment, seen by cam1 and cam2. The split comes from the subject (splits/up_fall_*.jsonl).

For each event and camera, 32 frames are taken at the midpoints of 32 equal bins over [start, end]
(same convention as edf/mcfd); each target time picks the nearest real frame, so short segments repeat
frames and n_distinct < 32. PNG bytes are copied from the zip unchanged.

Outputs:
    splits/up_fall_events_{train,val,test}.jsonl   run_probs.py-compatible manifest, one row per event
    <frames-root>/SubjectN/Activity<A>_Trial<T>_seg<K>/camC_fXX.png   (only for --extract splits)

Usage:
    python3 build_upfall_events.py --extract test          # manifests for all splits, frames for test
    python3 build_upfall_events.py --extract train val     # later, for LoRA training
"""
import argparse
import csv
import json
import re
import shutil
import zipfile
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
N_FRAMES = 32
MIN_DURATION = 0.1  # seconds; OmniFall has a few zero-length segments
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})_(\d{2})_(\d{2}(?:\.\d+)?)")


def frame_times(zip_path):
    """Sorted (seconds since first frame, member name) for every PNG in a camera zip."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".png"))
    stamps = []
    for n in names:
        m = TS_RE.search(Path(n).name)
        if not m:
            raise ValueError(f"cannot parse timestamp from {n} in {zip_path}")
        d, hh, mm, ss = m.groups()
        stamps.append(datetime.fromisoformat(f"{d}T{hh}:{mm}:00").timestamp() + float(ss))
    t0 = stamps[0]
    return [(s - t0, n) for s, n in zip(stamps, names)]


def pick(times, start, end):
    """Nearest frame to each of N_FRAMES bin midpoints in [start, end]."""
    out = []
    j = 0
    for i in range(N_FRAMES):
        t = start + (i + 0.5) * (end - start) / N_FRAMES
        while j + 1 < len(times) and abs(times[j + 1][0] - t) <= abs(times[j][0] - t):
            j += 1
        out.append(j)
    return out


def process_trial(job):
    trial, segments, frames_root, extract = job
    rows = []
    cams = {}
    for c in (1, 2):
        z = Path(trial[f"cam{c}_zip"])
        if trial[f"has_cam{c}"]:
            cams[c] = (z, frame_times(z))
    if not cams:
        return rows
    for k, (start, end, label) in enumerate(segments):
        if end - start < MIN_DURATION:
            continue
        name = f"Activity{trial['activity']}_Trial{trial['trial']}_seg{k}"
        dest = Path(frames_root) / trial["scenario"] / name
        row_cams = {}
        for c, (z, times) in cams.items():
            idx = pick(times, start, end)
            video_end = times[-1][0]
            row_cams[str(c)] = {
                "frames": [str(dest / f"cam{c}_f{i:02d}.png") for i in range(N_FRAMES)],
                "source_frames": [times[i][1] for i in idx],
                "n_distinct": len(set(idx)),
                "window_clipped_s": round(max(0.0, end - video_end - 0.1), 3),
            }
        if extract:
            expected = {f"cam{c}_f{i:02d}.png" for c in cams for i in range(N_FRAMES)}
            if not (dest.is_dir() and {p.name for p in dest.iterdir()} == expected):
                tmp = dest.with_name(dest.name + ".tmp")
                if tmp.exists():
                    shutil.rmtree(tmp)
                tmp.mkdir(parents=True)
                for c, (z, _) in cams.items():
                    with zipfile.ZipFile(z) as zf:
                        for i, src in enumerate(row_cams[str(c)]["source_frames"]):
                            (tmp / f"cam{c}_f{i:02d}.png").write_bytes(zf.read(src))
                if dest.exists():
                    shutil.rmtree(dest)
                tmp.rename(dest)
        rows.append({
            "event_id": f"up_fall/{trial['scenario']}/{name}",
            "dataset": "up_fall",
            "scenario": trial["scenario"],
            "subject": trial["subject"],
            "activity": trial["activity"],
            "trial": trial["trial"],
            "label": label[0],
            "label_name": label[1],
            "start": start,
            "end": end,
            "usable_multicam": len(row_cams) == 2,
            "split": trial["split"],
            "split_cs": trial["split"],
            "cams": row_cams,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("/scratch/wuhungma/omnifall/labels/up_fall.csv"))
    ap.add_argument("--label-ids", type=Path, default=Path("/scratch/wuhungma/omnifall/labels/label2id.csv"))
    ap.add_argument("--splits-dir", type=Path, default=HERE.parent / "splits")
    ap.add_argument("--frames-root", type=Path, default=Path("/scratch/wuhungma/frames32/up_fall_events"))
    ap.add_argument("--extract", nargs="*", default=[], choices=SPLITS, help="splits to extract frames for")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    id2name = {r["id"]: r["label"] for r in csv.DictReader(open(args.label_ids))}
    segs = defaultdict(list)  # "SubjectN/ActivityA/TrialT" -> [(start, end, (id, name))]
    for r in csv.DictReader(open(args.labels)):
        if r["cam"] != "1":  # cam2 segments are identical
            continue
        trial = r["path"].rsplit("/", 1)[0]
        segs[trial].append((float(r["start"]), float(r["end"]), (int(r["label"]), id2name[r["label"]])))
    for v in segs.values():
        v.sort()

    trials = []
    for s in SPLITS:
        with open(args.splits_dir / f"up_fall_{s}.jsonl") as f:
            trials += [json.loads(l) for l in f]
    jobs = []
    for t in trials:
        key = f"{t['scenario']}/Activity{t['activity']}/Trial{t['trial']}"
        if key in segs:
            jobs.append((t, segs[key], str(args.frames_root), t["split"] in args.extract))

    with Pool(args.workers) as pool:
        results = pool.map(process_trial, jobs, chunksize=4)

    by_split = defaultdict(list)
    for rows in results:
        for r in rows:
            by_split[r["split"]].append(r)
    for s in SPLITS:
        rows = sorted(by_split[s], key=lambda r: (r["subject"], r["activity"], r["trial"], r["start"]))
        with open(args.splits_dir / f"up_fall_events_{s}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n_fall = sum(r["label_name"] in ("fall", "fallen") for r in rows)
        tag = "frames extracted" if s in args.extract else "manifest only"
        print(f"{s:5s} {len(rows):4d} events ({n_fall} fall/fallen), "
              f"{sum(len(r['cams']) for r in rows)} camera-segments  [{tag}]")
    print(f"trials without OmniFall labels or video skipped: {len(trials) - len(jobs)}")


if __name__ == "__main__":
    main()
