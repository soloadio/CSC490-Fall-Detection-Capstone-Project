"""CPU speed of VideoMAE-base fall/no_fall: one window = 16 frames x 224x224, batch 1.
Measures PyTorch fp32 and dynamic int8 (Linear layers) at several thread counts, plus model size.
Preprocessing (decode + resize) is timed separately on 16 real PNG frames."""
import glob, io, os, statistics, time, torch
import numpy as np
from PIL import Image
from transformers import VideoMAEForVideoClassification

MODEL = "/scratch/wuhungma/videomae/base_lr5e-5/best"
print("CPU:", open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": "), "| cores given:", os.environ.get("SLURM_CPUS_PER_TASK"))
m32 = VideoMAEForVideoClassification.from_pretrained(MODEL).eval()
m8 = torch.ao.quantization.quantize_dynamic(m32, {torch.nn.Linear}, dtype=torch.qint8).eval()
def size_mb(m):
    b = io.BytesIO(); torch.save(m.state_dict(), b); return b.tell() / 2**20
print(f"size: fp32 {size_mb(m32):.0f} MB | int8 {size_mb(m8):.0f} MB")

x = torch.randn(1, 16, 3, 224, 224)
with torch.inference_mode():
    a, b = torch.softmax(m32(pixel_values=x).logits, -1), torch.softmax(m8(pixel_values=x).logits, -1)
print(f"int8 vs fp32 on a random input: P(fall) {b[0,1]:.4f} vs {a[0,1]:.4f}")

def bench(m, threads, n=8):
    torch.set_num_threads(threads)
    with torch.inference_mode():
        for _ in range(2): m(pixel_values=x)
        ts = []
        for _ in range(n):
            t = time.perf_counter(); m(pixel_values=x); ts.append(time.perf_counter() - t)
    return statistics.median(ts)

print("\nthreads | fp32 s/window | int8 s/window")
for th in (1, 2, 4, 8):
    print(f"{th:7d} | {bench(m32, th):13.3f} | {bench(m8, th):13.3f}", flush=True)

frames = sorted(glob.glob("/scratch/wuhungma/frames32/mcfd/chute01/*/cam1_f*.png"))[:16]
t = time.perf_counter()
for _ in range(5):
    arr = [np.asarray(Image.open(p).convert("RGB").resize((336, 224)).crop((56, 0, 280, 224))) for p in frames]
print(f"\npreprocessing 16 frames (720x480 PNG decode + resize + crop): {(time.perf_counter()-t)/5:.3f} s "
      "(a live camera gives raw frames, so decode is mostly not needed)")
