"""Does feeding 5 frames/s (what the app now receives) change the scores vs the 30 fps offline evaluation?"""
import json, subprocess, sys
import numpy as np
sys.path.insert(0, "/scratch/wuhungma/deploy/space"); sys.path.insert(0, "/project/6010122/group12")
from detector import FallModel, FrameBuffer
import eval_stream as es
MODEL = "/scratch/wuhungma/videomae/fallonly_lr5e-5/best"
GPU = {json.loads(l)["video"]: json.loads(l) for l in open("/scratch/wuhungma/stream_eval/videomae_fallonly/scores_test.jsonl")}
W, STRIDE, TH, FEED = 3.0, 0.5, 0.30, 0.2
m = FallModel(MODEL, int8=True)
tot = diff = 0; maxd = []; hi_gpu = hi_live = 0
for r in es.recordings("test", ["mcfd"]):
    times, w, h = es.mp4_frames(r["file"])
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", r["file"], "-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "passthrough", "-"], stdout=subprocess.PIPE)
    buf, ends, live, nxt = FrameBuffer(W), list(np.arange(STRIDE, times[-1] + 1e-6, STRIDE)), [], 0.0
    i = e = 0
    while True:
        raw = proc.stdout.read(w * h * 3)
        if len(raw) < w * h * 3 or i >= len(times): break
        while e < len(ends) and times[i] > ends[e]:
            fr = buf.window_frames(ends[e]); live.append(m.p_fall(fr) if fr else 0.0); e += 1
        if times[i] >= nxt:                                   # keep one frame every 0.2 s, like the phone
            buf.push(times[i], np.frombuffer(raw, np.uint8).reshape(h, w, 3)); nxt += FEED
        i += 1
    while e < len(ends):
        live.append(m.p_fall(buf.window_frames(ends[e]))); e += 1
    proc.kill()
    g = GPU[r["video"]]["windows"]["3.0"]; lut = {round(t, 3): p for t, p in zip(g["t"], g["p"])}
    gp = np.array([lut[round(x, 3)] for x in ends[:len(live)]]); lv = np.array(live)
    tot += len(lv); diff += int(((lv >= TH) != (gp >= TH)).sum()); maxd.append(float(np.abs(lv - gp).max()))
    hi_gpu += int((gp >= TH).sum()); hi_live += int((lv >= TH).sum())
print(f"MCFD test: {tot} windows | threshold decisions that differ (5 fps int8 vs 30 fps GPU): {diff} ({100*diff/tot:.1f}%)")
print(f"windows above threshold: 30 fps {hi_gpu}, 5 fps {hi_live} | median of per-recording max |diff| {np.median(maxd):.3f}")
