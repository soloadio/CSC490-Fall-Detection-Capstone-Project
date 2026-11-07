#!/usr/bin/env python3
"""VideoMAE wakes Gemma 3: does a Gemma verifier remove VideoMAE's false alarms and keep its real ones?

Stages (all write to --out, default /scratch/wuhungma/verify_gemma):
  alarms   (CPU, light) VideoMAE alarms of the deployed setting (and a more sensitive one) on the val/test streaming
           scores; each alarm is labelled hit / fa / neutral with the rules of stream_metrics.py.
  frames   (CPU job)    cut a clip of --clip-len s around every alarm: 32 frames at the native resolution, evenly
           spaced, ending at the alarm (variant "now") and 1 s after it (variant "wait1").
  score    (GPU job)    Gemma 3 12B-it, zero-shot and with the LoRA adapter: P(fall) with the forced-prefix scoring of
           run_probs.py (gemma_binary_scoring prompt) plus one sentence of evidence.
  report   (CPU, light) cascade results: an alarm is confirmed if Gemma's P(fall) >= tau.

Usage:
    python3 verify_alarms.py alarms
    sbatch verify_alarms.sbatch frames        # then: sbatch verify_alarms.sbatch score ; python3 verify_alarms.py report
"""
import argparse
import csv
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + [str(p) for p in HERE.parent.iterdir() if p.is_dir()]  # sibling folders of research/
import eval_stream as es  # noqa: E402
from build_upfall_events import frame_times as upfall_frame_times  # noqa: E402
from stream_metrics import alarms, episodes, load_scores, load_truth  # noqa: E402

SCORES = Path("/scratch/wuhungma/stream_eval/videomae_fallonly")
SETTINGS = {"deployed": (3.0, 1, 0.30), "sensitive": (3.0, 1, 0.10)}  # (window, k, threshold), cooldown 10 s
COOLDOWN, PRE_TOL, POST_TOL, MERGE_GAP = 10.0, 0.5, 2.0, 1.0  # stream_metrics.py defaults used for the model
VARIANTS = {"now": 0.0, "wait1": 1.0}  # clip ends this many seconds after the alarm
N_FRAMES = 32
MODEL = "/home/wuhungma/scratch/models/gemma-3-12b-it"
ADAPTER = "/scratch/wuhungma/lora/r16_bf16_ep1/adapter"


def classify(recs, truth, name, w, k, th):
    """Alarms of one setting with their kind: hit (first alarm of a fall), fa (false alarm), neutral (rest)."""
    out = []
    for r in recs:
        eps = episodes(truth.get(r["video"], []), MERGE_GAP)
        segs = truth.get(r["video"], [])
        falls = [e for e in eps if e["onset"] is not None]
        win = r["windows"][str(w)]
        hit = set()
        for t in alarms(win["t"], win["p"], th, k, COOLDOWN):
            match = [i for i, e in enumerate(falls) if e["onset"] - PRE_TOL <= t <= e["end"] + POST_TOL]
            onset = None
            if match:
                i = match[0]
                kind = "hit" if i not in hit else "neutral"
                hit.add(i)
                onset = falls[i]["onset"]
            elif any(e["start"] - PRE_TOL <= t <= e["end"] + POST_TOL for e in eps):
                kind = "neutral"
            elif not any(s <= t <= e for s, e, _ in segs):
                kind = "neutral"
            else:
                kind = "fa"
            out.append({"video": r["video"], "dataset": r["dataset"], "scenario": r["scenario"], "cam": r["cam"],
                        "t": t, "kind": kind, "onset": onset, "setting": name})
    return out


def activity(video, truth, t):
    """Label of the segment that contains time t (OmniFall label id), for looking at what fooled the models."""
    for s, e, lab in truth.get(video, []):
        if s <= t <= e:
            return lab
    return None


def stage_alarms(a):
    a.out.mkdir(parents=True, exist_ok=True)
    clips, n = {}, 0
    for split in ("val", "test"):
        recs = load_scores(SCORES / f"scores_{split}.jsonl")
        truth = load_truth(recs)
        files = {r["video"]: r["file"] for r in es.recordings(split, ["edf", "mcfd", "up_fall"])}
        for name, (w, k, th) in SETTINGS.items():
            for al in classify(recs, truth, name, w, k, th):
                al["label_at_alarm"] = activity(al["video"], truth, al["t"])
                for var, extra in VARIANTS.items():
                    key = (split, al["video"], round(al["t"], 3), var)
                    if key not in clips:
                        clips[key] = {"clip_id": f"{split}_{n:05d}_{var}", "split": split, "variant": var,
                                      "file": files[al["video"]], "t_end": al["t"] + extra, "alarm": {}, **{
                                          k_: al[k_] for k_ in ("video", "dataset", "scenario", "cam", "t")}}
                        n += 1
                    clips[key]["alarm"][name] = {"kind": al["kind"], "onset": al["onset"],
                                                 "label_at_alarm": al["label_at_alarm"]}
    with open(a.out / "clips.jsonl", "w") as f:
        for c in clips.values():
            f.write(json.dumps(c) + "\n")
    by = {}
    for c in clips.values():
        if c["variant"] != "now":
            continue
        for name, v in c["alarm"].items():
            by[(c["split"], name, v["kind"])] = by.get((c["split"], name, v["kind"]), 0) + 1
    print(f"{len(clips)} clips ({len(clips) // len(VARIANTS)} alarms x {len(VARIANTS)} variants) -> {a.out / 'clips.jsonl'}")
    for key in sorted(by):
        print("  ", *key, by[key])


def clip_frames(file, dataset, t_ends, clip_len):
    """{t_end: list of 32 RGB arrays} for several clips of one recording (native resolution)."""
    if dataset == "up_fall":
        tn = upfall_frame_times(file)
        times = np.array([t for t, _ in tn])
    else:
        times, w, h = es.mp4_frames(file)
        times = np.array(times)
    plans = {}
    for te in t_ends:
        tg = te - clip_len + (np.arange(N_FRAMES) + 0.5) * clip_len / N_FRAMES
        j = np.clip(np.searchsorted(times, tg), 1, len(times) - 1)
        idx = np.where(np.abs(times[j - 1] - tg) <= np.abs(times[j] - tg), j - 1, j)
        plans[te] = np.where(tg < times[0], 0, idx)  # before the recording started: first frame
    needed = sorted({int(i) for p in plans.values() for i in p})
    keep = {}
    if dataset == "up_fall":
        with zipfile.ZipFile(file) as z:
            for i in needed:
                keep[i] = np.asarray(Image.open(z.open(tn[i][1])).convert("RGB"))
    else:
        lo = needed[0]
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", file, "-f", "rawvideo", "-pix_fmt", "rgb24",
                                 "-vsync", "passthrough", "-"], stdout=subprocess.PIPE)
        want, i = set(needed), 0
        while True:
            raw = proc.stdout.read(w * h * 3)
            if len(raw) < w * h * 3:
                break
            if i in want:
                keep[i] = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            i += 1
            if i > needed[-1]:
                break
        proc.kill()
        last = max(keep)  # packets can outnumber decoded frames
        plans = {te: np.minimum(p, last) for te, p in plans.items()}
        keep.update({int(x): keep[last] for p in plans.values() for x in p if int(x) not in keep})
    return {te: [keep[int(i)] for i in p] for te, p in plans.items()}


def stage_frames(a):
    clips = [json.loads(l) for l in open(a.out / a.clips)]
    root = a.out / "clips"
    todo = {}
    for c in clips:
        if not (root / c["clip_id"] / "f31.jpg").exists():
            todo.setdefault((c["file"], c["dataset"]), []).append(c)
    print(f"{sum(len(v) for v in todo.values())} clips to cut from {len(todo)} recordings", flush=True)
    for n, ((file, ds), cs) in enumerate(todo.items(), 1):
        try:
            fr = clip_frames(file, ds, sorted({c["t_end"] for c in cs}), a.clip_len)
        except Exception as e:  # noqa: BLE001 - reported, the run continues
            print("FAILED", file, f"{type(e).__name__}: {e}", flush=True)
            continue
        for c in cs:
            d = root / c["clip_id"]
            d.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(fr[c["t_end"]]):
                Image.fromarray(im).save(d / f"f{i:02d}.jpg", quality=95)
        if n % 20 == 0:
            print(f"  {n}/{len(todo)} recordings", flush=True)
    print("frames done")


def stage_score(a):
    import time

    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    import gemma_binary_scoring as gs

    clips = [json.loads(l) for l in open(a.out / a.clips)]
    suffix = "" if a.clips == "clips.jsonl" else "_" + Path(a.clips).stem  # e.g. gemma_lora_clips_mv.jsonl
    processor = AutoProcessor.from_pretrained(MODEL)
    pre, plan, bad = gs.label_plan(processor.tokenizer)
    assert not bad, bad
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
    for tag in a.models:
        if tag == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
            print("LoRA adapter merged", flush=True)
        out_file = a.out / f"gemma_{tag}{suffix}.jsonl"
        done = {json.loads(l)["clip_id"] for l in open(out_file)} if out_file.exists() else set()
        todo = [c for c in clips if c["clip_id"] not in done and (a.out / "clips" / c["clip_id"] / "f31.jpg").exists()]
        print(f"[{tag}] {len(done)} done, {len(todo)} to do", flush=True)
        t0 = time.time()
        with open(out_file, "a") as f:
            for n, c in enumerate(todo, 1):
                rec = {"clip_id": c["clip_id"]}
                try:
                    paths = [str(a.out / "clips" / c["clip_id"] / f"f{i:02d}.jpg") for i in range(N_FRAMES)]
                    inp = processor.apply_chat_template(gs.messages_for(paths), add_generation_prompt=True, tokenize=True,
                                                        return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
                    logp = gs.score_labels(model, inp, pre, plan)
                    probs, mass, ent = gs.to_probs(logp)
                    with torch.inference_mode():
                        gen = model.generate(**inp, max_new_tokens=100, do_sample=False)
                    rec.update(p_fall=round(probs["fall"], 5), mass=round(mass, 4),
                               text=processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
                except Exception as e:  # noqa: BLE001 - recorded; a rerun retries it
                    rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                    if isinstance(e, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  [{tag}] {n}/{len(todo)} | {(time.time() - t0) / n:.1f} s/clip", flush=True)


def stage_report(a):
    clips = {c["clip_id"]: c for c in map(json.loads, open(a.out / "clips.jsonl"))}
    res = {}
    for tag in a.models:
        p = a.out / f"gemma_{tag}.jsonl"
        if p.exists():
            res[tag] = {r["clip_id"]: r for r in map(json.loads, open(p)) if "error" not in r}
    lines = []
    for setting in SETTINGS:
        for variant in VARIANTS:
            for tag, rs in res.items():
                lines.append(f"\n## VideoMAE setting '{setting}' | Gemma {tag} | clip {variant}")
                lines.append(f"{'split':5} {'tau':>5} {'alarms':>6} {'hits':>5} {'FA':>4} -> kept hits  removed FA  (hits kept / FA kept)")
                for split in ("val", "test"):
                    cs = [c for c in clips.values() if c["split"] == split and c["variant"] == variant
                          and setting in c["alarm"] and c["clip_id"] in rs]
                    hit = [c for c in cs if c["alarm"][setting]["kind"] == "hit"]
                    fa = [c for c in cs if c["alarm"][setting]["kind"] == "fa"]
                    for tau in (0.1, 0.3, 0.5, 0.7, 0.9):
                        kh = sum(rs[c["clip_id"]]["p_fall"] >= tau for c in hit)
                        kf = sum(rs[c["clip_id"]]["p_fall"] >= tau for c in fa)
                        lines.append(f"{split:5} {tau:5.1f} {len(cs):6d} {len(hit):5d} {len(fa):4d} -> "
                                     f"{kh:4d}/{len(hit):<4d} {len(fa) - kf:4d}/{len(fa):<4d}")
    text = "\n".join(lines)
    (a.out / "report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["alarms", "frames", "score", "report"])
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/verify_gemma"))
    ap.add_argument("--clips", default="clips.jsonl", help="clip list in --out (clips_mv.jsonl: multi-camera clips)")
    ap.add_argument("--clip-len", type=float, default=4.0, help="seconds of video in a clip")
    ap.add_argument("--models", nargs="+", default=["zeroshot", "lora"], choices=["zeroshot", "lora"])
    a = ap.parse_args()
    {"alarms": stage_alarms, "frames": stage_frames, "score": stage_score, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
