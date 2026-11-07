"""Joint multi-view pass: Gemma 3 12B-it sees k frames from EACH camera of one event in a single prompt.

One record per event (not per camera). Same forced prefix {"verdict": " and the same 10-label probabilities as run_probs.py,
so single-camera, voting and joint multi-view results are directly comparable. Events: usable_multicam only.
Resumable, shardable (--shard I/N). Output: <out-root>/<tag>/results_shardIofN.jsonl and config.json.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=32, help="frames per camera (evenly strided from the 32 extracted)")
ap.add_argument("--max-cams", type=int, default=0, help="use at most N cameras per event (first N by camera id); 0 = all")
ap.add_argument("--datasets", default="edf,mcfd")
ap.add_argument("--manifest", default="/scratch/wuhungma/frames32/manifest_frames32.jsonl")
ap.add_argument("--out-root", default="/scratch/wuhungma/runs")
ap.add_argument("--tag", default=None)
ap.add_argument("--shard", default="0/1")
ap.add_argument("--limit", type=int, default=0, help="process at most N NEW events (for testing)")
ap.add_argument("--scoring", default="gemma_scoring", help="prompt/scoring module: gemma_scoring (10 classes) or gemma_binary_scoring (fall / no_fall)")
ap.add_argument("--dry-run", action="store_true", help="no model: check manifest, shard and that frame files exist")
a = ap.parse_args()

MODEL = "/home/wuhungma/scratch/models/gemma-3-12b-it"
sid, sn = map(int, a.shard.split("/"))
tag = a.tag or ("bin_" if "binary" in a.scoring else "") + f"mv_k{a.k}" + (f"_c{a.max_cams}" if a.max_cams else "")
run_dir = Path(a.out_root) / tag; run_dir.mkdir(parents=True, exist_ok=True)
out_file = run_dir / f"results_shard{sid}of{sn}.jsonl"
idx = [round(j * 31 / (a.k - 1)) for j in range(a.k)] if a.k > 1 else [15]
dsets = set(a.datasets.split(","))

items = []
for line in open(a.manifest):
    e = json.loads(line)
    if e["dataset"] not in dsets or not e["usable_multicam"]:
        continue
    cams = sorted(c for c, v in e["cams"].items() if v.get("frames"))
    if a.max_cams: cams = cams[: a.max_cams]
    if len(cams) < 2:
        continue
    items.append(dict(event_id=e["event_id"], dataset=e["dataset"], scenario=e["scenario"], gold=e["label_name"],
                      split_cs=e["split_cs"], cams=cams, frames={c: e["cams"][c]["frames"] for c in cams}))
total = len(items)
items = items[sid::sn]

done = set()
if out_file.exists():
    for l in open(out_file):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue
        if "error" not in r: done.add(r["event_id"])
todo = [it for it in items if it["event_id"] not in done]
if a.limit: todo = todo[: a.limit]
print(f"tag={tag} k/cam={a.k} idx={idx} | events {total} | shard {sid}/{sn}: {len(items)} | already done {len(done)} | to do now {len(todo)}", flush=True)

if a.dry_run:
    missing = [p for it in todo for c in it["cams"] for i in idx for p in [it["frames"][c][i]] if not os.path.exists(p)]
    print("frame files missing:", len(missing), missing[:3])
    ncam = sorted({len(it["cams"]) for it in todo}); print("cameras per event:", ncam, "| images per prompt:", sorted({len(it["cams"]) * a.k for it in todo}))
    print("approx input tokens (256/image):", max(len(it["cams"]) for it in todo) * a.k * 256)
    sys.exit(0)

import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
sys.path.insert(0, str(Path(__file__).parent))
import importlib
gs = importlib.import_module(a.scoring)

def instruction_mv_10(n_cams, k):
    return f"""You are watching one person filmed at the same time by {n_cams} different cameras. For each camera you get {k} frames in time order.
The cameras show the same moment from different viewpoints; the person may be hidden in some views.
Decide which ONE activity the person is doing during this moment. Choose exactly one label:
{gs.DEFS}

Answer with a single JSON object and nothing else:
{{"verdict": "<one label>", "evidence": "<one short sentence about what you see>", "occlusion_flag": <true if the person is hidden or partly hidden in most views, else false>}}"""

instruction_mv = getattr(gs, "instruction_mv", instruction_mv_10)

def messages_mv(cams, frames, idx):
    content = [{"type": "text", "text": instruction_mv(len(cams), len(idx))}]
    for ci, c in enumerate(cams, 1):
        for j, i in enumerate(idx):
            tg = "earliest" if j == 0 else "latest" if j == len(idx) - 1 else "later"
            content += [{"type": "text", "text": f"Camera {ci}, frame {j+1} ({tg}):"}, {"type": "image", "image": Image.open(frames[c][i]).convert("RGB")}]
    return [{"role": "user", "content": content}]

ex = items[0]
cfg = dict(model=MODEL, k_per_cam=a.k, frame_idx=idx, max_cams=a.max_cams, datasets=sorted(dsets), manifest=a.manifest,
           prefix=gs.PREFIX, classes=gs.CLASSES, instruction_example=instruction_mv(len(ex["cams"]), a.k),
           instruction_sha1=hashlib.sha1(instruction_mv(len(ex["cams"]), a.k).encode()).hexdigest()[:12],
           scoring_code_sha1=hashlib.sha1(open(gs.__file__, "rb").read()).hexdigest()[:12])
(run_dir / "config.json").write_text(json.dumps(cfg, indent=1))

processor = AutoProcessor.from_pretrained(MODEL); tok = processor.tokenizer
pre, plan, bad = gs.label_plan(tok); assert not bad, bad
model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
print("model loaded on", torch.cuda.get_device_name(0), flush=True)

n_err, t0 = 0, time.time()
with open(out_file, "a") as f:
    for n, it in enumerate(todo, 1):
        rec = {k: v for k, v in it.items() if k != "frames"}; rec["k_per_cam"] = a.k
        try:
            inp = processor.apply_chat_template(messages_mv(it["cams"], it["frames"], idx), add_generation_prompt=True,
                                                tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            t = time.time(); logp = gs.score_labels(model, inp, pre, plan)
            probs, mass, ent = gs.to_probs(logp)
            rec.update(logp={k: round(v, 5) for k, v in logp.items()}, probs={k: round(v, 5) for k, v in probs.items()},
                       mass=round(mass, 4), entropy=round(ent, 4), argmax=max(probs, key=probs.get),
                       input_tokens=int(inp["input_ids"].shape[-1]), sec=round(time.time() - t, 2))
        except Exception as ex_:
            n_err += 1; rec["error"] = f"{type(ex_).__name__}: {str(ex_)[:200]}"
            if isinstance(ex_, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        f.write(json.dumps(rec) + "\n"); f.flush()
        if n % 10 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)} | {el/n:.1f}s/event | eta {(len(todo)-n)*el/n/60:.0f} min | errors {n_err}", flush=True)
print(f"finished shard {sid}/{sn}: {len(todo)-n_err} ok, {n_err} errors, {(time.time()-t0)/60:.1f} min")
sys.exit(1 if n_err else 0)
