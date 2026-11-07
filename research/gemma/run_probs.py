"""Full single-camera pass: Gemma 3 12B-it label probabilities for every ok camera-segment.

For each camera-segment we pick k of the 32 extracted frames (evenly strided), force the answer prefix
{"verdict": "  and store the log-probability of each of the 10 labels (plus normalized probs, mass, entropy).
Raw logp are kept so a temperature / calibration can be fitted later without rerunning.

Resumable (skips finished camera-segments), shardable (--shard I/N), errors are recorded not fatal.
Output: <out-root>/<tag>/results_shardIofN.jsonl and config.json.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=32, choices=[4, 8, 16, 32], help="frames per camera-segment")
ap.add_argument("--manifest", required=True, help="e.g. splits/frames32_val.jsonl (make_frames_manifest.py)")
ap.add_argument("--adapter", default=None, help="LoRA adapter dir (train_lora.py output); merged into the bf16 model")
ap.add_argument("--out-root", default="/scratch/wuhungma/runs")
ap.add_argument("--tag", default=None, help="run name (default k<k>)")
ap.add_argument("--shard", default="0/1", help="I/N: process items i where index %% N == I")
ap.add_argument("--limit", type=int, default=0, help="process at most N NEW items (already-finished ones do not count; for testing)")
ap.add_argument("--with-text", action="store_true", help="also generate the free-text JSON (evidence, occlusion_flag)")
ap.add_argument("--scoring", default="gemma_scoring", help="prompt/scoring module: gemma_scoring (10 classes) or gemma_binary_scoring (fall / no_fall)")
ap.add_argument("--event-list", default=None, help="jsonl with an event_id per line; only these events are processed")
ap.add_argument("--dry-run", action="store_true", help="no model: check manifest, shard and that frame files exist")
a = ap.parse_args()

MODEL = "/home/wuhungma/scratch/models/gemma-3-12b-it"
sid, sn = map(int, a.shard.split("/"))
tag = a.tag or (f"bin_k{a.k}" if "binary" in a.scoring else f"k{a.k}")
run_dir = Path(a.out_root) / tag; run_dir.mkdir(parents=True, exist_ok=True)
out_file = run_dir / f"results_shard{sid}of{sn}.jsonl"
idx = [round(j * 31 / (a.k - 1)) for j in range(a.k)]          # strided subset of the 32 extracted frames

keep = {json.loads(l)["event_id"] for l in open(a.event_list)} if a.event_list else None
items = []
for line in open(a.manifest):
    e = json.loads(line)
    if keep is not None and e["event_id"] not in keep:
        continue
    for cam, v in e["cams"].items():
        if v.get("frames"):
            items.append(dict(event_id=e["event_id"], dataset=e["dataset"], scenario=e["scenario"], cam=cam,
                              gold=e["label_name"], usable_multicam=e["usable_multicam"], split_cs=e["split_cs"],
                              n_distinct=v["n_distinct"], window_clipped_s=v["window_clipped_s"], frames=v["frames"]))
total = len(items)
items = items[sid::sn]

done = set()
if out_file.exists():
    for l in open(out_file):
        try: r = json.loads(l)
        except json.JSONDecodeError: continue                  # a half-written last line from an interrupted job
        if "error" not in r: done.add((r["event_id"], r["cam"]))
todo = [it for it in items if (it["event_id"], it["cam"]) not in done]
if a.limit: todo = todo[: a.limit]
print(f"tag={tag} k={a.k} frame idx={idx if a.k <= 8 else str(idx[:4]) + '...'} | all items {total} | shard {sid}/{sn}: {len(items)} | "
      f"already done {len(done)} | to do now {len(todo)}", flush=True)

if a.dry_run:
    missing = [p for it in todo for i in idx for p in [it["frames"][i]] if not os.path.exists(p)]
    print("frame files missing:", len(missing), missing[:3])
    print("example:", {k: v for k, v in todo[0].items() if k != "frames"}); sys.exit(0)

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
sys.path.insert(0, str(Path(__file__).parent))
import importlib
gs = importlib.import_module(a.scoring)

cfg = dict(model=MODEL, k=a.k, frame_idx=idx, manifest=a.manifest, prefix=gs.PREFIX, classes=gs.CLASSES,
           instruction_example=gs.instruction(a.k), instruction_sha1=hashlib.sha1(gs.instruction(a.k).encode()).hexdigest()[:12],
           scoring_code_sha1=hashlib.sha1(open(gs.__file__, "rb").read()).hexdigest()[:12], with_text=a.with_text, adapter=a.adapter)
(run_dir / "config.json").write_text(json.dumps(cfg, indent=1))

processor = AutoProcessor.from_pretrained(MODEL); tok = processor.tokenizer
pre, plan, bad = gs.label_plan(tok); assert not bad, bad
model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda").eval()
if a.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().eval()
    print("LoRA adapter merged:", a.adapter, flush=True)
print("model loaded on", torch.cuda.get_device_name(0), flush=True)

n_err, t0 = 0, time.time()
with open(out_file, "a") as f:
    for n, it in enumerate(todo, 1):
        rec = {k: v for k, v in it.items() if k != "frames"}; rec["k"] = a.k
        try:
            inp = processor.apply_chat_template(gs.messages_for([it["frames"][i] for i in idx]), add_generation_prompt=True,
                                                tokenize=True, return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            t = time.time(); logp = gs.score_labels(model, inp, pre, plan)
            probs, mass, ent = gs.to_probs(logp)
            rec.update(logp={k: round(v, 5) for k, v in logp.items()}, probs={k: round(v, 5) for k, v in probs.items()},
                       mass=round(mass, 4), entropy=round(ent, 4), argmax=max(probs, key=probs.get), input_tokens=int(inp["input_ids"].shape[-1]))
            if a.with_text:
                with torch.inference_mode():
                    out = model.generate(**inp, max_new_tokens=120, do_sample=False)
                rec["text"] = processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            rec["sec"] = round(time.time() - t, 2)
        except Exception as ex:                                # record and carry on; rerun retries these
            n_err += 1; rec["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
            if isinstance(ex, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        f.write(json.dumps(rec) + "\n"); f.flush()
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            print(f"  {n}/{len(todo)} | {el/n:.1f}s/item | eta {(len(todo)-n)*el/n/60:.0f} min | errors {n_err}", flush=True)
print(f"finished shard {sid}/{sn}: {len(todo)-n_err} ok, {n_err} errors, {(time.time()-t0)/60:.1f} min")
sys.exit(1 if n_err else 0)
