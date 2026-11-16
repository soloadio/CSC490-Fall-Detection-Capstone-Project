#!/usr/bin/env python3
"""Read video_probe.py results and print make_reasons_v3.py options for the best writer:
the condition with the most correct speed classes; --speed only if that condition gets >= 80% right."""
import json, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from video_probe import speed_of
rows = [json.loads(l) for l in open(sys.argv[1])]
conds = [c for c in ("q2_images", "q2_video", "q3_video") if all(c in r for r in rows)]
score = {c: sum((speed_of(r[c]) == "fast") == (r["kind"] == "fall") for r in rows) / len(rows) for c in conds}
best = max(conds, key=score.get)
model = "/scratch/wuhungma/models/" + ("qwen3-vl-8b-instruct" if best.startswith("q3") else "qwen2-vl-7b-instruct")
mode = "images" if best.endswith("images") else "video"
print(f"--model {model} --mode {mode}" + (" --speed" if score[best] >= 0.8 else ""))
print(json.dumps(score), file=sys.stderr)
