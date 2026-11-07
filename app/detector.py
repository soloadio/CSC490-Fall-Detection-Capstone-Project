"""Live fall detection with the fine-tuned VideoMAE-base (fall / no_fall), CPU.

Mirrors the offline evaluation (eval_stream.py, stream_metrics.py, fusion_metrics.py) so the live app behaves like
the measured system:
  * a window of WINDOW seconds ends "now"; 16 frames are taken at the even midpoints of 32 equal bins
    over [now - WINDOW, now] (nearest buffered frame; before the first frame, the first frame is repeated);
  * each frame: short side resized to 224 (bilinear), centre crop 224x224, ImageNet mean/std;
  * alarm: P(fall) >= threshold for k windows in a row, then silent until the score has dropped below the
    threshold AND `cooldown` seconds have passed (stream_metrics.py).
"""
import os
import threading
from collections import deque

import numpy as np
import torch
from PIL import Image

SIZE = 224
N_BINS, FRAME_IDX = 32, list(range(0, 32, 2))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_224(rgb):
    """HxWx3 uint8 (any size) -> 224x224x3 uint8, exactly as eval_stream.to_224."""
    im = Image.fromarray(rgb) if isinstance(rgb, np.ndarray) else rgb
    w, h = im.size
    s = SIZE / min(w, h)
    nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
    x0, y0 = (nw - SIZE) // 2, (nh - SIZE) // 2
    return np.asarray(im.convert("RGB").resize((nw, nh), Image.BILINEAR).crop((x0, y0, x0 + SIZE, y0 + SIZE)))


class FallModel:
    """The network. One instance is shared by all sessions; calls are serialised with a lock."""

    def __init__(self, model_dir, int8=True, threads=None):
        from transformers import VideoMAEForVideoClassification
        if threads:
            torch.set_num_threads(int(threads))
        model = VideoMAEForVideoClassification.from_pretrained(model_dir).eval()
        if int8:  # dynamic int8 on the Linear layers: ~4x smaller in memory, ~1.5x faster on CPU
            try:
                model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
            except Exception as e:  # API deprecated in recent PyTorch; keep running in fp32 rather than crash
                print(f"int8 quantization unavailable ({type(e).__name__}: {e}); using fp32", flush=True)
                int8 = False
        self.model, self.int8, self.lock = model, int8, threading.Lock()
        self.fall_index = int(model.config.label2id.get("fall", 1))

    @torch.inference_mode()
    def p_fall(self, frames224, wait=True):
        """frames224: list of 16 arrays 224x224x3 uint8 in time order -> P(fall).
        With wait=False, returns None instead of queueing when another session is using the model."""
        x = (np.stack(frames224).astype(np.float32) / 255 - MEAN) / STD  # T, H, W, C
        x = torch.from_numpy(x).permute(0, 3, 1, 2).unsqueeze(0)  # 1, T, C, H, W
        if not self.lock.acquire(blocking=wait):
            return None
        try:
            logits = self.model(pixel_values=x).logits
        finally:
            self.lock.release()
        return float(torch.softmax(logits, -1)[0, self.fall_index])


class FrameBuffer:
    """Recent (time, 224x224 frame) pairs of one camera; picks the 16 frames of a window."""

    def __init__(self, window=3.0, keep_extra=1.0):
        self.window, self.keep = window, window + keep_extra
        self.items = deque()

    def push(self, t, rgb):
        self.items.append((t, to_224(rgb)))
        while self.items and self.items[0][0] < t - self.keep:
            self.items.popleft()

    def last_time(self):
        return self.items[-1][0] if self.items else None

    def stats(self, now, span=3.0):
        """(frames per second received over the last `span` s, largest gap between frames in that span)."""
        ts = [t for t, _ in self.items if t >= now - span]
        if len(ts) < 2:
            return float(len(ts)) / span, None
        gaps = [b - a for a, b in zip(ts, ts[1:])]
        return len(ts) / span, max(gaps)

    def window_frames(self, t_end):
        if not self.items:
            return None
        times = np.array([t for t, _ in self.items])
        targets = t_end - self.window + (np.array(FRAME_IDX) + 0.5) * self.window / N_BINS
        j = np.clip(np.searchsorted(times, targets), 1, max(1, len(times) - 1))
        if len(times) == 1:
            idx = np.zeros(len(targets), dtype=int)
        else:
            idx = np.where(np.abs(times[j - 1] - targets) <= np.abs(times[j] - targets), j - 1, j)
        return [self.items[i][1] for i in idx]


class AlarmRule:
    def __init__(self, threshold=0.30, k=1, cooldown=10.0):
        self.th, self.k, self.cooldown = threshold, k, cooldown
        self.run, self.armed, self.last_alarm = 0, True, -1e9

    def update(self, t, p):
        """Feed one window score; True if this window raises an alarm (same rule as stream_metrics.alarms)."""
        if p >= self.th:
            self.run += 1
        else:
            self.run, self.armed = 0, True
        if self.run >= self.k and self.armed and t - self.last_alarm >= self.cooldown:
            self.armed, self.last_alarm = False, t
            return True
        return False


class Room:
    """Several cameras watching one room; each posts its window scores, the room raises one alarm.

    Two or more live cameras: room score = the second-highest of the cameras' highest P(fall) over the last
    `agree` seconds, i.e. an alarm needs at least two cameras to see a fall within that time ("two agree" in
    fusion_metrics.py: same recall as one camera, about 6x fewer false alarms on the paired test recordings).
    One live camera: its own score, i.e. the single-camera rule. Times are server arrival times (time.monotonic),
    which puts all cameras on one clock."""

    def __init__(self, threshold=0.30, k=1, cooldown=10.0, agree=1.0, stale=3.0):
        self.rule, self.agree, self.stale = AlarmRule(threshold, k, cooldown), agree, stale
        self.cams = {}  # camera id -> {"name", "scores": deque of (t, p), "seen": t of the last score}
        self.lock = threading.Lock()
        self.t, self.score, self.last_alarm, self.alarm_cams, self.log = -1e9, None, None, [], []
        self.checking, self.confirmed, self.reports = False, None, []  # second-opinion check (verifier.py)

    def live(self, now):
        return {i: c for i, c in self.cams.items() if now - c["seen"] <= self.stale}

    def recent(self, c, t):
        ps = [p for s, p in c["scores"] if s >= t - self.agree]
        return max(ps) if ps else c["scores"][-1][1]

    def post(self, cam_id, name, t, p):
        """One camera's score for the window ending at t. True if the room raises an alarm now."""
        with self.lock:
            c = self.cams.setdefault(cam_id, {"scores": deque()})
            c["name"], c["seen"] = name, t
            c["scores"].append((t, p))
            while c["scores"][0][0] < t - self.stale:
                c["scores"].popleft()
            self.t = max(self.t, t)  # scores of different cameras can arrive slightly out of order
            per_cam = sorted(((self.recent(x, self.t), x["name"]) for x in self.live(self.t).values()), reverse=True)
            self.score = per_cam[1][0] if len(per_cam) >= 2 else per_cam[0][0]
            if not self.rule.update(self.t, self.score):
                return False
            self.last_alarm = self.t
            self.alarm_cams = [n for q, n in per_cam if q >= self.rule.th]
            return True

    def leave(self, cam_id):
        with self.lock:
            self.cams.pop(cam_id, None)

    def snapshot(self, now):
        """(cameras as [(name, latest P, seconds since its last score)], room score, last alarm time, cameras
        that saw the last alarm), for display."""
        with self.lock:
            cams = sorted((c["name"], c["scores"][-1][1], now - c["seen"]) for c in self.live(now).values())
            return cams, self.score, self.last_alarm, list(self.alarm_cams)


def settings_from_env():
    return dict(window=float(os.environ.get("WINDOW", 3.0)), threshold=float(os.environ.get("THRESHOLD", 0.30)),
                k=int(os.environ.get("K", 1)), cooldown=float(os.environ.get("COOLDOWN", 10.0)),
                agree=float(os.environ.get("AGREE", 1.0)),
                infer_every=float(os.environ.get("INFER_EVERY", 0.5)), int8=os.environ.get("INT8", "1") == "1",
                model_dir=os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "model")),
                threads=os.environ.get("THREADS"))
