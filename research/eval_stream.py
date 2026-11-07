#!/usr/bin/env python3
"""Score whole recordings with a sliding window, as a live camera would see them (VideoMAE fall / no_fall).

Every camera recording of a split is replayed from start to end. A window of W seconds ends every
--stride seconds; its P(fall) is stamped at the window's END time (the moment a live system would know).
Several window lengths are scored in one pass, because decoding is the slow part.

Windows end at stride, 2*stride, ... from the very start: a live camera would already have been running,
so the part of an early window before the recording began is filled with the first frame. (UP-Fall
recordings are ~10 s with the fall often starting ~2 s in, so waiting for a full first window would
miss most falls.)

Frames per window follow train_videomae.py exactly: 16 frames at the even midpoints of 32 equal bins
over [t - W, t] (nearest real frame), short side resized to 224 (bilinear), centre crop, ImageNet norm.

Recordings (one per camera), restricted to the split's scenarios / subjects in splits/:
    edf, mcfd   OmniFall MP4s rebuilt by prepare_stream_videos.sbatch (correct timing)
    up_fall     the timestamped PNGs inside the camera zips (variable ~18 fps), read directly
Label timelines come from OmniFall labels/<dataset>.csv; stream_metrics.py turns scores into alarms.

Output: <out>/scores_<split>.jsonl, one line per recording:
    {"video", "dataset", "scenario", "cam", "duration", "n_frames", "windows": {"2.0": {"t": [...], "p": [...]}, ...}}
Re-running appends only recordings not yet in the file.

Usage:
    python3 eval_stream.py --split val
    python3 eval_stream.py --split test --model /scratch/wuhungma/videomae/holdout_up_fall/best --out ...
    python3 eval_stream.py --split val --limit 3          # quick check
"""
import argparse
import csv
import json
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_upfall_events import frame_times as upfall_frame_times  # noqa: E402

LABELS = Path("/scratch/wuhungma/omnifall/labels")
VIDEOS = Path("/scratch/wuhungma/stream_eval/videos")
MODEL = Path("/scratch/wuhungma/videomae/base_lr5e-5/best")
SIZE = 224
N_BINS, FRAME_IDX = 32, list(range(0, 32, 2))


def recordings(split, datasets):
    """One dict per camera recording of the split, from the team split files + OmniFall label paths."""
    label_paths = {d: sorted({r["path"] for r in csv.DictReader(open(LABELS / f"{d}.csv"))}) for d in datasets}
    out = []
    scen = defaultdict(set)
    with open(HERE / "splits" / f"{split}.jsonl") as f:
        for line in f:
            r = json.loads(line)
            scen[r["dataset"]].add(r["scenario"])
    for d in ("edf", "mcfd"):
        if d not in datasets:
            continue
        for p in label_paths[d]:
            if p.split("/")[0] in scen[d]:
                out.append({"video": f"{d}/{p}", "dataset": d, "scenario": p.split("/")[0], "path": p,
                            "cam": p.split("/")[1].replace("view", "").replace("cam", ""),
                            "file": str(VIDEOS / d / f"{p}.mp4")})
    if "up_fall" in datasets:
        known = set(label_paths["up_fall"])
        with open(HERE / "splits" / f"up_fall_{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                for c in (1, 2):
                    p = (f"{r['scenario']}/Activity{r['activity']}/Trial{r['trial']}/"
                         f"{r['scenario']}Activity{r['activity']}Trial{r['trial']}Camera{c}")
                    if r[f"has_cam{c}"] and p in known:
                        out.append({"video": f"up_fall/{p}", "dataset": "up_fall", "scenario": r["scenario"],
                                    "path": p, "cam": str(c), "file": r[f"cam{c}_zip"]})
    return out


def window_plan(times, windows, stride):
    """For each window length: end times and the frame indices (16 per window) it uses."""
    times = np.asarray(times)
    plan = {}
    for w in windows:
        ends = np.arange(stride, times[-1] + 1e-6, stride)  # early windows repeat frame 0 for t < 0
        offs = (np.array(FRAME_IDX) + 0.5) * w / N_BINS - w  # relative to the window end
        target = ends[:, None] + offs[None, :]
        j = np.clip(np.searchsorted(times, target), 1, len(times) - 1)
        idx = np.where(np.abs(times[j - 1] - target) <= np.abs(times[j] - target), j - 1, j)
        plan[w] = (ends, idx)
    return plan


def to_224(im):
    w, h = im.size
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    x0, y0 = (nw - SIZE) // 2, (nh - SIZE) // 2
    return np.asarray(im.convert("RGB").resize((nw, nh), Image.BILINEAR).crop((x0, y0, x0 + SIZE, y0 + SIZE)))


def mp4_frames(path):
    """(frame times in s from the first frame, width, height) for an MP4, from packet timestamps."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:packet=pts_time", "-of", "json", path],
                       capture_output=True, text=True, check=True)
    info = json.loads(r.stdout)
    t = sorted(float(p["pts_time"]) for p in info["packets"] if p.get("pts_time") not in (None, "N/A"))
    st = info["streams"][0]
    return [x - t[0] for x in t], st["width"], st["height"]


def decode_mp4(path, needed, n, w, h):
    """Decode every frame once (ffmpeg pipe) and keep only the needed indices, resized to 224."""
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-vsync", "passthrough", "-"], stdout=subprocess.PIPE)
    size, keep, i, buf = w * h * 3, {}, 0, None
    while True:
        nxt = proc.stdout.read(size)
        if len(nxt) < size:
            break
        buf = nxt
        if i in needed:
            keep[i] = to_224(Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", 0, 1))
        i += 1
    proc.wait()
    if i == 0:
        raise RuntimeError("ffmpeg decoded no frames")
    if i - 1 not in keep:  # windows past the last decoded frame are clipped to it
        keep[i - 1] = to_224(Image.frombuffer("RGB", (w, h), buf, "raw", "RGB", 0, 1))
    if i != n:
        print(f"  note: {path}: decoded {i} frames, {n} packets", flush=True)
    return keep


def as_is(item):
    """DataLoader collate that leaves numpy arrays alone (the default turns them into tensors)."""
    return item


class Recordings:
    def __init__(self, recs, windows, stride):
        self.recs, self.windows, self.stride = recs, windows, stride

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, k):
        rec = dict(self.recs[k])
        try:
            if rec["dataset"] == "up_fall":
                tn = upfall_frame_times(rec["file"])
                times = [t for t, _ in tn]
                plan = window_plan(times, self.windows, self.stride)
                needed = sorted({int(i) for _, idx in plan.values() for i in idx.ravel()})
                with zipfile.ZipFile(rec["file"]) as z:
                    keep = {i: to_224(Image.open(z.open(tn[i][1]))) for i in needed}
            else:
                times, w, h = mp4_frames(rec["file"])
                plan = window_plan(times, self.windows, self.stride)
                needed = {int(i) for _, idx in plan.values() for i in idx.ravel()}
                keep = decode_mp4(rec["file"], needed, len(times), w, h)
                last = max(keep)
                plan = {wl: (e, np.minimum(idx, last)) for wl, (e, idx) in plan.items()}  # packets > frames
        except Exception as e:  # noqa: BLE001 - reported per recording, the run continues
            return rec, None, None, f"{type(e).__name__}: {e}"
        order = sorted(keep)
        lut = np.zeros(order[-1] + 1, dtype=np.int64)
        lut[order] = np.arange(len(order))
        frames = np.stack([keep[i] for i in order])
        plan = {wl: (e, lut[idx]) for wl, (e, idx) in plan.items()}  # frame number -> row of `frames`
        rec.update(duration=round(times[-1], 3), n_frames=len(times))
        return rec, frames, plan, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["edf", "mcfd", "up_fall"])
    ap.add_argument("--windows", nargs="+", type=float, default=[2.0, 3.0, 4.0], help="window lengths, seconds")
    ap.add_argument("--stride", type=float, default=0.5, help="seconds between window ends")
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/stream_eval/videomae_base"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="only the first N recordings (for testing)")
    args = ap.parse_args()

    import torch
    from transformers import VideoMAEForVideoClassification

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / f"scores_{args.split}.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(l)["video"] for l in open(out_file)}
    recs = recordings(args.split, args.datasets)
    todo = [r for r in recs if r["video"] not in done][: args.limit]
    print(f"split {args.split}: {len(recs)} recordings, {len(done)} already scored, {len(todo)} to do | "
          f"windows {args.windows} s, stride {args.stride} s | model {args.model}", flush=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    device = torch.device("cuda")
    model = VideoMAEForVideoClassification.from_pretrained(args.model).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    loader = torch.utils.data.DataLoader(Recordings(todo, args.windows, args.stride), batch_size=None,
                                         num_workers=args.workers, collate_fn=as_is)
    t0, n_win, errors = time.time(), 0, 0
    with open(out_file, "a") as f, torch.no_grad():
        for k, (rec, frames, plan, err) in enumerate(loader, 1):
            if err:
                errors += 1
                print(f"  ERROR {rec['video']}: {err}", flush=True)
                continue
            frames = torch.from_numpy(frames).to(device)  # N, H, W, C uint8
            row = {key: rec[key] for key in ("video", "dataset", "scenario", "cam", "duration", "n_frames")}
            row["windows"] = {}
            for wl, (ends, idx) in plan.items():
                probs = []
                for b in range(0, len(idx), args.batch):
                    x = frames[torch.from_numpy(idx[b:b + args.batch]).to(device)]  # B, T, H, W, C
                    x = (x.permute(0, 1, 4, 2, 3).float() / 255 - mean) / std  # B, T, C, H, W
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(pixel_values=x).logits.float()
                    probs += torch.softmax(logits, -1)[:, 1].tolist()
                row["windows"][str(wl)] = {"t": [round(float(e), 3) for e in ends], "p": [round(p, 5) for p in probs]}
                n_win += len(ends)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f"  {k}/{len(todo)} recordings | {n_win} windows | {el / 60:.1f} min | "
                      f"eta {el / k * (len(todo) - k) / 60:.0f} min | errors {errors}", flush=True)
    print(f"done: {out_file} ({errors} errors)", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
