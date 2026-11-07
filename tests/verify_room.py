"""Replay every test camera pair through detector.Room (scores posted camera by camera, as the live app does) and
compare its alarms with the offline "two agree" rule of fusion_metrics.py at the deployed setting."""
import sys
sys.path[:0] = ["/scratch/wuhungma/deploy/space", "/project/6010122/group12"]
import fusion_metrics as F
from detector import Room
from stream_metrics import alarms, load_scores, load_truth

W, TH, K, CD = "3.0", 0.30, 1, 10.0
recs = load_scores("/scratch/wuhungma/stream_eval/videomae_fallonly/scores_test.jsonl")
out, lab, _ = F.build(recs, load_truth(recs), "pairs", 0.5, 1.0)
same = diff = n_off = n_live = 0
examples = []
for g, fused in enumerate(out["two agree"]):
    a, b = out["single"][2 * g]["windows"][W], out["single"][2 * g + 1]["windows"][W]
    offline = alarms(fused["windows"][W]["t"], fused["windows"][W]["p"], TH, K, CD)
    room, live = Room(TH, K, CD, agree=1.0), []
    for t, pa, pb in zip(a["t"], a["p"], b["p"]):
        for cam, p in (("A", pa), ("B", pb)):
            if room.post(cam, cam, t, p):
                live.append(t)
    n_off += len(offline); n_live += len(live)
    if offline == live:
        same += 1
    else:
        diff += 1
        examples.append((fused["video"], offline, live))
print(f"{same + diff} camera pairs: identical alarms in {same}, different in {diff}; alarms offline {n_off}, live {n_live}")
for e in examples[:10]:
    print("  ", e)
