#!/usr/bin/env python3
"""LoRA / QLoRA fine-tuning of Gemma 3 12B-it for fall / no_fall, on single-camera clips.

Uses exactly the zero-shot prompt (gemma_binary_scoring): the assistant turn starts with
{"verdict": "  and the loss is cross-entropy on the label tokens ("fall" or "no_fall") that
follow. That is the same quantity run_probs.py scores, so an adapter trained here is evaluated
with  run_probs.py --adapter <out>/adapter.

Only the language model gets LoRA (attention + MLP projections); the vision tower and projector
stay frozen. --qlora loads the frozen model in 4-bit NF4 so it fits a 40 GB GPU slice.

--reasons <jsonl> (make_reasons.py): the answer continues with the clip's evidence sentence and occlusion flag,
{"verdict": "fall", "evidence": "...", "occlusion_flag": false}<end_of_turn>, and the loss adds
--reason-weight x (mean cross-entropy over those extra tokens) to the unchanged label loss. Clips without a reason
keep the label-only loss.

Resumable: every --save-every optimizer steps the adapter, optimizer and position are saved to
<out>/checkpoint; re-running the same command continues from there.

Usage:
    python3 train_lora.py --out /scratch/wuhungma/lora/r16 --max-steps 5      # feasibility test
    python3 train_lora.py --out /scratch/wuhungma/lora/r16 --qlora
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + [str(p) for p in HERE.parent.iterdir() if p.is_dir()]  # sibling folders of research/
import gemma_binary_scoring as gs  # noqa: E402

MODEL = "/home/wuhungma/scratch/models/gemma-3-12b-it"
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_clips(manifest, k, reasons=None):
    idx = [round(j * 31 / (k - 1)) for j in range(k)]
    clips = []
    with open(manifest) as f:
        for line in f:
            e = json.loads(line)
            target = "fall" if e["label_name"] in gs.POSITIVE_GOLD else "no_fall"
            for cam, v in e["cams"].items():
                r = (reasons or {}).get(f"{e['event_id']}#cam{cam}")
                clips.append({"event_id": e["event_id"], "dataset": e["dataset"], "cam": cam,
                              "frames": [v["frames"][i] for i in idx], "target": target,
                              "reason": r if r and r.get("evidence") else None})
    return clips


class ClipDataset(torch.utils.data.Dataset):
    """Builds the full token sequence prompt + prefix + label, on CPU workers."""

    def __init__(self, clips, processor, plan, pre):
        self.clips, self.processor, self.plan, self.pre = clips, processor, plan, pre
        self.tok = processor.tokenizer
        self.eot = self.tok.convert_tokens_to_ids("<end_of_turn>")

    def rest_tokens(self, c):
        """Tokens of the answer after the label: '", "evidence": "...", "occlusion_flag": x}<end_of_turn>'.
        The label tokens stay exactly those that run_probs.py scores; None if the tokenizer would merge them."""
        r = c["reason"]
        if not r:
            return []
        ev = json.dumps(r["evidence"])[1:-1]  # JSON-escaped, without the outer quotes
        tail = f'", "evidence": "{ev}", "occlusion_flag": {"true" if r["occlusion_flag"] else "false"}}}'
        full = self.tok.encode(gs.PREFIX + c["target"] + tail, add_special_tokens=False)
        head = self.pre + self.plan[c["target"]]
        if full[:len(head)] != head:
            return []
        return full[len(head):] + [self.eot]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        inp = self.processor.apply_chat_template(gs.messages_for(c["frames"]), add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt")
        cont = self.plan[c["target"]]
        rest = self.rest_tokens(c)
        extra = torch.tensor([self.pre + cont + rest])
        out = {"input_ids": torch.cat([inp["input_ids"], extra], 1), "pixel_values": inp["pixel_values"],
               "n_label": len(cont), "n_rest": len(rest)}
        if "token_type_ids" in inp:
            out["token_type_ids"] = torch.cat([inp["token_type_ids"], torch.zeros_like(extra)], 1)
        return out


def label_loss(model, batch, device, reason_weight=0.5):
    """Cross-entropy on the label tokens, plus reason_weight x mean cross-entropy on the reason tokens (if any)."""
    ids = batch["input_ids"].to(device)
    n, m = batch["n_label"], batch.get("n_rest", 0)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
              pixel_values=batch["pixel_values"].to(device, torch.bfloat16), logits_to_keep=n + m + 1)
    if "token_type_ids" in batch:
        kw["token_type_ids"] = batch["token_type_ids"].to(device)
    logits = model(**kw).logits[0, :n + m].float()  # position t predicts token t+1
    targets = ids[0, -(n + m):]
    loss = torch.nn.functional.cross_entropy(logits[:n], targets[:n])
    if m:
        loss = loss + reason_weight * torch.nn.functional.cross_entropy(logits[n:], targets[n:])
    return loss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=HERE.parent / "splits" / "frames32_train.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=32, help="frames per clip (must match evaluation)")
    ap.add_argument("--qlora", action="store_true", help="4-bit frozen base model")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (feasibility test)")
    ap.add_argument("--save-every", type=int, default=25, help="optimizer steps between checkpoints")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reasons", type=Path, help="make_reasons.py output: also train the evidence sentence")
    ap.add_argument("--reason-weight", type=float, default=0.5)
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    # multi-GPU: launched by torchrun, one process per GPU. Each process runs grad_accum / world
    # clips of every mini-batch; LoRA gradients are summed across GPUs before each update, so the
    # math is the same as one GPU running all grad_accum clips.
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    if args.grad_accum % world:
        raise SystemExit(f"--grad-accum {args.grad_accum} must be a multiple of the GPU count {world}")
    per_rank = args.grad_accum // world
    log = print if rank == 0 else (lambda *a, **k: None)

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoint"
    torch.manual_seed(args.seed)  # same LoRA init on every GPU

    reasons = {json.loads(l)["clip_id"]: json.loads(l) for l in open(args.reasons)} if args.reasons else None
    clips = load_clips(args.train, args.k, reasons)
    random.Random(args.seed).shuffle(clips)
    total_steps = math.ceil(len(clips) * args.epochs / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    n_micro = total_steps * args.grad_accum  # whole mini-batches only, so every GPU gets the same count
    n_pos = sum(c["target"] == "fall" for c in clips)
    log(f"train clips {len(clips)} ({n_pos} fall, {len(clips) - n_pos} no_fall), "
        f"{total_steps} optimizer steps x {args.grad_accum} clips, {world} GPU(s)"
        + (f", {sum(c['reason'] is not None for c in clips)} with a reason" if reasons else ""), flush=True)

    processor = AutoProcessor.from_pretrained(MODEL)
    pre, plan, bad = gs.label_plan(processor.tokenizer)
    assert not bad, bad

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if args.qlora else None
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                           device_map={"": local_rank}, quantization_config=quant)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if (ckpt / "adapter").exists():
        model = PeftModel.from_pretrained(model, ckpt / "adapter", is_trainable=True)
        state = json.loads((ckpt / "state.json").read_text())
    else:
        lcfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                          target_modules=TARGETS, task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        state = {"step": 0, "micro": 0, "losses": []}
    if rank == 0:
        model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    if (ckpt / "optimizer.pt").exists():
        sd = torch.load(ckpt / "optimizer.pt", weights_only=False, map_location=f"cuda:{local_rank}")
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        log(f"resumed at step {state['step']}", flush=True)

    if rank == 0:
        (args.out / "config.json").write_text(json.dumps(
            {**vars(args), "out": str(args.out), "train": str(args.train), "model": MODEL, "targets": TARGETS,
             "n_clips": len(clips), "total_steps": total_steps, "gpus": world,
             "reasons": str(args.reasons) if args.reasons else None}, indent=1))

    # global micro-batch i goes to GPU i % world
    order = [clips[i % len(clips)] for i in range(n_micro)][state["micro"]:][rank::world]
    loader = torch.utils.data.DataLoader(ClipDataset(order, processor, plan, pre), batch_size=None,
                                         shuffle=False, num_workers=args.workers, prefetch_factor=2)

    def save(dest):
        if rank == 0:
            model.save_pretrained(dest / "adapter")
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, dest / "optimizer.pt")
            (dest / "state.json").write_text(json.dumps(state))
        if world > 1:
            dist.barrier()

    model.train()
    device = torch.device("cuda", local_rank)
    t0, acc_loss, run_steps, n_local = time.time(), 0.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        if state["step"] >= total_steps:
            break
        loss = label_loss(model, batch, device, args.reason_weight) / args.grad_accum
        loss.backward()
        acc_loss += loss.item()
        n_local += 1
        if n_local % per_rank:
            continue
        state["micro"] += args.grad_accum
        if world > 1:
            for p in params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            t = torch.tensor(acc_loss, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            acc_loss = t.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["losses"].append(round(acc_loss, 4))
        run_steps += 1
        log(f"step {state['step']}/{total_steps} loss {acc_loss:.4f} lr {sched.get_last_lr()[0]:.2e} | "
            f"{(time.time() - t0) / run_steps:.1f}s/step | "
            f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB/GPU", flush=True)
        acc_loss = 0.0
        if state["step"] % args.save_every == 0:
            ckpt.mkdir(exist_ok=True)
            save(ckpt)

    if rank == 0:
        model.save_pretrained(args.out / "adapter")
        (args.out / "losses.json").write_text(json.dumps(state["losses"]))
    log(f"done: {state['step']} steps, adapter saved to {args.out / 'adapter'}", flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
