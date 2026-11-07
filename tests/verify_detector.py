"""(1) live path vs offline GPU scores, (2) int8 vs fp32 on real windows. CPU only."""
import json, subprocess, sys, time
import numpy as np
sys.path.insert(0, "/scratch/wuhungma/deploy/space"); sys.path.insert(0, "/project/6010122/group12")
from detector import FallModel, FrameBuffer, AlarmRule
import eval_stream as es

MODEL = "/scratch/wuhungma/videomae/fallonly_lr5e-5/best"
GPU = {json.loads(l)["video"]: json.loads(l) for l in open("/scratch/wuhungma/stream_eval/videomae_fallonly/scores_test.jsonl")}
W, STRIDE, TH = 3.0, 1.0, 0.30
fp32, int8 = FallModel(MODEL, int8=False), FallModel(MODEL, int8=True)

def gpu_at(video, ends):
    g = GPU[video]["windows"]["3.0"]; lut = {round(t, 3): p for t, p in zip(g["t"], g["p"])}
    return np.array([lut[round(e, 3)] for e in ends])

# (1) live path: push every decoded frame in time order, score when a window end is reached
recs = [r for r in es.recordings("test", ["mcfd"])][:3]
print("== (1) live replay vs offline GPU scores (fp32, window 3 s, every 1 s)")
for r in recs:
    times, w, h = es.mp4_frames(r["file"])
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", r["file"], "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-vsync", "passthrough", "-"], stdout=subprocess.PIPE)
    buf, ends, live = FrameBuffer(W), list(np.arange(STRIDE, times[-1] + 1e-6, STRIDE)), []
    i, e = 0, 0
    while e < len(ends):
        raw = proc.stdout.read(w * h * 3)
        if len(raw) < w * h * 3 or i >= len(times):
            break
        while e < len(ends) and times[i] > ends[e]:          # window end reached before this frame
            live.append(fp32.p_fall(buf.window_frames(ends[e]))); e += 1
        buf.push(times[i], np.frombuffer(raw, np.uint8).reshape(h, w, 3)); i += 1
    while e < len(ends):
        live.append(fp32.p_fall(buf.window_frames(ends[e]))); e += 1
    proc.kill()
    g = gpu_at(r["video"], ends[:len(live)]); live = np.array(live)
    print(f"  {r['video']}: {len(live)} windows | mean |diff| {np.abs(live-g).mean():.4f} | max {np.abs(live-g).max():.3f} | "
          f"alarm-threshold decisions differ in {int(((live>=TH)!=(g>=TH)).sum())} windows")

# (2) int8 vs fp32 on real windows (offline frame selection), MCFD all + UP-Fall subset incl. laying
recs = es.recordings("test", ["mcfd", "up_fall"])
recs = [r for r in recs if r["dataset"] == "mcfd"] + [r for r in recs if r["dataset"] == "up_fall"][::4]
ds = es.Recordings(recs, [W], STRIDE)
P32, P8, PG, tf = [], [], [], [0.0, 0.0]
t0 = time.time()
for k in range(len(ds)):
    rec, frames, plan, err = ds[k]
    if err: print("  skip", rec["video"], err); continue
    ends, idx = plan[W]
    g = gpu_at(rec["video"], ends)
    for j in range(len(ends)):
        fr = [frames[q] for q in idx[j]]
        a = time.perf_counter(); P32.append(fp32.p_fall(fr)); b = time.perf_counter(); P8.append(int8.p_fall(fr)); c = time.perf_counter()
        tf[0] += b - a; tf[1] += c - b
    PG += list(g)
P32, P8, PG = map(np.array, (P32, P8, PG))
print(f"== (2) {len(ds)} recordings, {len(P32)} windows ({(time.time()-t0)/60:.1f} min)")
print(f"  fp32 CPU vs GPU : mean |diff| {np.abs(P32-PG).mean():.4f}, threshold decisions differ {int(((P32>=TH)!=(PG>=TH)).sum())}")
print(f"  int8 vs fp32    : mean |diff| {np.abs(P8-P32).mean():.4f}, max {np.abs(P8-P32).max():.3f}, threshold decisions differ {int(((P8>=TH)!=(P32>=TH)).sum())} of {len(P32)}")
print(f"  windows >= {TH}: fp32 {int((P32>=TH).sum())}, int8 {int((P8>=TH).sum())}, GPU {int((PG>=TH).sum())}")
print(f"  avg time per window on {__import__('torch').get_num_threads()} threads: fp32 {tf[0]/len(P32):.3f} s, int8 {tf[1]/len(P8):.3f} s")
