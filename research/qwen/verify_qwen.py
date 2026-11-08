#!/usr/bin/env python3
"""Qwen2-VL-7B + LoRA (Solomon) on the same VideoMAE alarm clips as the Gemma check (verify_alarms.py / verify_mv.py).

Reads the clips that verify_alarms.py frames already cut (<clips-dir>/clips.jsonl, clips_mv.jsonl and
<clips-dir>/clips/<clip_id>/fNN.jpg) and writes <out>/gemma_lora{,_clips_mv}.jsonl in the same format, so
verify_alarms.py report / verify_mv.py report work unchanged on <out> (the "lora" model in those reports is Qwen here).
Resumable; records with an error are retried on the next run.

    python3 verify_qwen.py --out /scratch/wuhungma/verify_qwen
"""
import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + [str(p) for p in HERE.parent.iterdir() if p.is_dir()]  # sibling folders of research/
sys.path.insert(0, str(HERE.parent.parent / "app"))  # qwen_scoring.py (shared with the Modal checker)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips-dir", type=Path, default=Path("/scratch/wuhungma/verify_gemma"))
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/verify_qwen"))
    ap.add_argument("--base", default="/scratch/wuhungma/models/qwen2-vl-7b-instruct")
    ap.add_argument("--adapter", default="/scratch/wuhungma/models/qwen2vl-7b-fall-lora")
    ap.add_argument("--lists", nargs="+", default=["clips.jsonl", "clips_mv.jsonl"])
    ap.add_argument("--limit", type=int, default=0, help="at most N new clips per list (testing)")
    a = ap.parse_args()

    from qwen_scoring import QwenScorer
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.lists + ["events_mv.jsonl"]:  # the reports read these from --out
        link = a.out / name
        if not link.exists() and (a.clips_dir / name).exists():
            link.symlink_to(a.clips_dir / name)
    qs = QwenScorer(a.base, a.adapter).load()
    print("Qwen + LoRA loaded", flush=True)
    for name in a.lists:
        suffix = "" if name == "clips.jsonl" else "_" + Path(name).stem
        out_file = a.out / f"gemma_lora{suffix}.jsonl"
        good = [r for r in map(json.loads, open(out_file)) if "error" not in r] if out_file.exists() else []
        out_file.write_text("".join(json.dumps(r) + "\n" for r in good))  # drop failed records so they are redone
        done = {r["clip_id"] for r in good}
        clips = [json.loads(l) for l in open(a.clips_dir / name)]
        todo = [c for c in clips if c["clip_id"] not in done
                and (a.clips_dir / "clips" / c["clip_id"] / "f31.jpg").exists()]
        if a.limit:
            todo = todo[:a.limit]
        print(f"[{name}] {len(done)} done, {len(todo)} to do", flush=True)
        t0, n_err = time.time(), 0
        with open(out_file, "a") as f:
            for n, c in enumerate(todo, 1):
                rec = {"clip_id": c["clip_id"]}
                try:
                    d = a.clips_dir / "clips" / c["clip_id"]
                    r = qs.check([Image.open(d / f"f{i:02d}.jpg").convert("RGB") for i in range(32)])
                    rec.update(p_fall=round(r["p_fall"], 5), mass=round(r["mass"], 4), text=r["text"])
                except Exception as e:  # noqa: BLE001 - recorded; a rerun retries it
                    n_err += 1
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    if n_err == 1:
                        import traceback
                        traceback.print_exc()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)} | {(time.time() - t0) / n:.1f} s/clip | errors {n_err}", flush=True)


if __name__ == "__main__":
    main()
