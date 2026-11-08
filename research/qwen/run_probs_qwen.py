"""Qwen2-VL-7B-Instruct version of group12/run_probs.py: same prompt, same clips, same output format.

Prompt text, frame tags, forced prefix {"verdict": "  and the label set come unchanged from group12's
gemma_scoring / gemma_binary_scoring (imported read-only), so results line up 1:1 with the Gemma runs.
Only the model call is Qwen-specific (score_labels below mirrors gemma_scoring.score_labels).

Clips: every camera-segment with frames in group12/frames32/manifest_frames32.jsonl (edf, mcfd, occu).
Frame paths there point at /scratch/wuhungma/frames32 (not readable for us); they are mapped to group12/frames32.
Resumable, shardable (--shard I/N), errors recorded not fatal.
Output: <out-root>/<tag>/results_shardIofN.jsonl and config.json.
"""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path

GROUP = "/project/def-ajbonner/group12"
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=32, choices=[4, 8, 16, 32], help="frames per camera-segment")
ap.add_argument("--manifest", default=f"{GROUP}/frames32/manifest_frames32.jsonl")
ap.add_argument("--frames-from", default="/scratch/wuhungma/frames32", help="frame path prefix in the manifest ...")
ap.add_argument("--frames-to", default=f"{GROUP}/frames32", help="... replaced by this readable copy")
ap.add_argument("--out-root", default=str(HERE / "runs"))
ap.add_argument("--tag", default=None, help="run name (default qwen_k<k> / qwen_bin_k<k>)")
ap.add_argument("--shard", default="0/1", help="I/N: process items i where index %% N == I")
ap.add_argument("--limit", type=int, default=0, help="process at most N NEW items (for testing)")
ap.add_argument("--with-text", action="store_true", help="also generate the free-text JSON (evidence, occlusion_flag)")
ap.add_argument("--scoring", default="gemma_scoring", help="gemma_scoring (10 classes) or gemma_binary_scoring (fall / no_fall)")
ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct", help="HF id (from the local cache) or a local path")
ap.add_argument("--adapter", default=None, help="LoRA adapter dir (train_lora_qwen.py output); merged in memory, base weights untouched")
ap.add_argument("--datasets", default="", help="comma list, e.g. edf,mcfd (default: all in the manifest)")
ap.add_argument("--max-pixels", type=int, default=78000, help="cap on pixels per frame (edf/mcfd 320x240 frames are unchanged; up_fall 640x480 is scaled to about the same size)")
ap.add_argument("--check", action="store_true", help="also score the first item without the KV cache and print the difference")
ap.add_argument("--dry-run", action="store_true", help="no model: check manifest, shard and that frame files exist")
a = ap.parse_args()

sid, sn = map(int, a.shard.split("/"))
tag = a.tag or ("qwen_bin_k" if "binary" in a.scoring else "qwen_k") + str(a.k)
run_dir = Path(a.out_root) / tag; run_dir.mkdir(parents=True, exist_ok=True)
out_file = run_dir / f"results_shard{sid}of{sn}.jsonl"
idx = [round(j * 31 / (a.k - 1)) for j in range(a.k)]          # identical to run_probs.py

items = []
for line in open(a.manifest):
    e = json.loads(line)
    if a.datasets and e["dataset"] not in a.datasets.split(","):
        continue
    for cam, v in e["cams"].items():
        if v.get("frames"):
            items.append(dict(event_id=e["event_id"], dataset=e["dataset"], scenario=e["scenario"], cam=cam,
                              gold=e["label_name"], usable_multicam=e["usable_multicam"], split_cs=e["split_cs"], split=e.get("split"),
                              n_distinct=v["n_distinct"], window_clipped_s=v["window_clipped_s"],
                              frames=v["frames"] if os.access(v["frames"][0], os.R_OK)   # e.g. up_fall: only in his scratch
                              else [p.replace(a.frames_from, a.frames_to, 1) for p in v["frames"]]))
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")                   # compute nodes have no internet
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
sys.dont_write_bytecode = True                                 # never create __pycache__ inside group12
sys.path.insert(0, GROUP)                                      # read-only import of the groupmate's prompt modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gemma"))  # in the repo: research/gemma/ (same prompt files)
import importlib
gs = importlib.import_module(a.scoring)
from gemma_scoring import to_probs


def vision_kw(inputs, n_extra):
    """pixel_values / image_grid_thw, plus per-token type ids (mm_token_type_ids, needed for M-RoPE) padded with
    text (0) for the n_extra tokens appended after the prompt."""
    kw = dict(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
    for k, v in inputs.items():
        if "token_type_ids" in k:
            kw[k] = torch.cat([v, torch.zeros(v.shape[0], n_extra, dtype=v.dtype, device=v.device)], dim=1)
    return kw


@torch.inference_mode()
def score_labels(model, inputs, pre, plan):
    """As gemma_scoring.score_labels: prefill prompt + forced prefix once, then teacher-force each label on a cache copy."""
    dev = inputs["input_ids"].device
    ids = torch.cat([inputs["input_ids"], torch.tensor([pre], device=dev)], dim=1)
    L = ids.shape[1]
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), **vision_kw(inputs, len(pre)),
                use_cache=True, logits_to_keep=1)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    cache, logp = out.past_key_values, {}
    delta = model.model.rope_deltas                            # M-RoPE offset of text after the images, set by the prefill
    for lab, cont in plan.items():
        lp = first[cont[0]].item()
        if len(cont) > 1:
            c = copy.deepcopy(cache)
            n = len(cont) - 1                                  # explicit positions: transformers would build them from the full mask
            pos = (torch.arange(L, L + n, device=dev) + delta.to(dev)).view(1, 1, n).expand(3, 1, n)
            o = model(input_ids=torch.tensor([cont[:-1]], device=dev),
                      attention_mask=torch.ones(1, L + n, dtype=torch.long, device=dev),
                      position_ids=pos, cache_position=torch.arange(L, L + n, device=dev),
                      past_key_values=c, use_cache=True)
            lps = torch.log_softmax(o.logits[0].float(), dim=-1)
            lp += sum(lps[j - 1, cont[j]].item() for j in range(1, len(cont)))
            del c, o
        logp[lab] = lp
    return logp


@torch.inference_mode()
def score_labels_nocache(model, inputs, pre, plan):
    """Reference: one full forward per label. Slow; only used by --check to validate score_labels."""
    dev = inputs["input_ids"].device
    base = torch.cat([inputs["input_ids"], torch.tensor([pre], device=dev)], dim=1)
    L, logp = base.shape[1], {}
    for lab, cont in plan.items():
        ids = torch.cat([base, torch.tensor([cont], device=dev)], dim=1)
        lg = model(input_ids=ids, attention_mask=torch.ones_like(ids), **vision_kw(inputs, len(pre) + len(cont))).logits[0]
        lps = torch.log_softmax(lg[L - 1: L - 1 + len(cont)].float(), dim=-1)
        logp[lab] = sum(lps[j, cont[j]].item() for j in range(len(cont)))
    return logp


processor = AutoProcessor.from_pretrained(a.model); tok = processor.tokenizer
processor.image_processor.size["longest_edge"] = a.max_pixels
pre, plan, bad = gs.label_plan(tok); assert not bad, bad
ip = processor.image_processor
cfg = dict(model=a.model, k=a.k, frame_idx=idx, manifest=a.manifest, frames_to=a.frames_to, prefix=gs.PREFIX, classes=gs.CLASSES,
           label_tokens=plan, image_pixels=dict(min=ip.size["shortest_edge"], max=ip.size["longest_edge"]),
           instruction_example=gs.instruction(a.k), instruction_sha1=hashlib.sha1(gs.instruction(a.k).encode()).hexdigest()[:12],
           scoring_code_sha1=hashlib.sha1(open(gs.__file__, "rb").read()).hexdigest()[:12], with_text=a.with_text, adapter=a.adapter)
(run_dir / "config.json").write_text(json.dumps(cfg, indent=1))
print("label tokens:", plan, flush=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
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
                                                tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            t = time.time(); logp = score_labels(model, inp, pre, plan)
            probs, mass, ent = to_probs(logp)
            rec.update(logp={k: round(v, 5) for k, v in logp.items()}, probs={k: round(v, 5) for k, v in probs.items()},
                       mass=round(mass, 4), entropy=round(ent, 4), argmax=max(probs, key=probs.get), input_tokens=int(inp["input_ids"].shape[-1]))
            if a.check and n == 1:
                ref = score_labels_nocache(model, inp, pre, plan)
                print("  --check max |logp cached - full| =", max(abs(logp[k] - ref[k]) for k in logp), flush=True)
            if a.with_text:
                with torch.inference_mode():
                    out = model.generate(**inp, max_new_tokens=120, do_sample=False)
                rec["text"] = processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            rec["sec"] = round(time.time() - t, 2)
        except Exception as ex:                                # record and carry on; rerun retries these
            n_err += 1; rec["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
            if n_err == 1:
                import traceback; traceback.print_exc()
            if isinstance(ex, torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
        f.write(json.dumps(rec) + "\n"); f.flush()
        if n % 25 == 0 or n == len(todo) or a.limit:
            el = time.time() - t0
            print(f"  {n}/{len(todo)} | {it['event_id']} cam{it['cam']} gold={it['gold']} -> {rec.get('argmax', 'ERR')} "
                  f"| {el/n:.1f}s/item | eta {(len(todo)-n)*el/n/60:.0f} min | errors {n_err}", flush=True)
print(f"finished shard {sid}/{sn}: {len(todo)-n_err} ok, {n_err} errors, {(time.time()-t0)/60:.1f} min")
sys.exit(1 if n_err else 0)
