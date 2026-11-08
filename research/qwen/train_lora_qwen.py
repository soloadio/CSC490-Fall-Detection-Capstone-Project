"""LoRA fine-tuning of Qwen2-VL-7B-Instruct for fall / no_fall: a port of group12/train_lora.py.

Same prompt (gemma_binary_scoring, imported read-only), same target: the assistant turn starts with
{"verdict": "  and the loss is cross-entropy on the label tokens ("fall" or "no_fall") that follow, i.e.
exactly what run_probs_qwen.py scores. Evaluate with  run_probs_qwen.py --adapter <out>/adapter.

Only the language model gets LoRA (attention + MLP projections); the vision tower stays frozen.
The base model is never written to: only the small adapter is saved under --out.
Resumable: every --save-every optimizer steps the adapter, optimizer and position go to <out>/checkpoint.
Single GPU.

    python train_lora_qwen.py --out ~/qwen/lora/r16 --max-steps 5      # feasibility test
    python train_lora_qwen.py --out ~/qwen/lora/r16
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

import torch

GROUP = "/project/def-ajbonner/group12"
sys.dont_write_bytecode = True                                 # never create __pycache__ inside group12
sys.path.insert(0, GROUP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gemma"))  # in the repo: research/gemma/ (same prompt files)
import gemma_binary_scoring as gs  # noqa: E402

TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_clips(manifest, k, datasets):
    idx = [round(j * 31 / (k - 1)) for j in range(k)]
    clips = []
    for line in open(manifest):
        e = json.loads(line)
        if e["dataset"] not in datasets:
            continue
        target = "fall" if e["label_name"] in gs.POSITIVE_GOLD else "no_fall"
        for cam, v in e["cams"].items():
            clips.append({"event_id": e["event_id"], "dataset": e["dataset"], "cam": cam,
                          "frames": [v["frames"][i] for i in idx], "target": target})
    return clips


class ClipDataset(torch.utils.data.Dataset):
    """Builds the full token sequence prompt + prefix + label, on CPU workers."""

    def __init__(self, clips, processor, plan, pre):
        self.clips, self.processor, self.plan, self.pre = clips, processor, plan, pre

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        inp = self.processor.apply_chat_template(gs.messages_for(c["frames"]), add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt")
        extra = torch.tensor([self.pre + self.plan[c["target"]]])
        out = {"input_ids": torch.cat([inp["input_ids"], extra], 1), "pixel_values": inp["pixel_values"],
               "image_grid_thw": inp["image_grid_thw"], "n_label": len(self.plan[c["target"]])}
        for k, v in inp.items():                               # mm_token_type_ids (M-RoPE): text = 0 for appended tokens
            if "token_type_ids" in k:
                out[k] = torch.cat([v, torch.zeros_like(extra, dtype=v.dtype)], 1)
        return out


def label_loss(model, batch, device):
    ids = batch["input_ids"].to(device)
    n = batch["n_label"]
    kw = {k: v.to(device) for k, v in batch.items() if "token_type_ids" in k}
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=batch["pixel_values"].to(device, torch.bfloat16),
                   image_grid_thw=batch["image_grid_thw"].to(device), logits_to_keep=n + 1, **kw).logits[0, :n].float()
    return torch.nn.functional.cross_entropy(logits, ids[0, -n:])   # position t predicts token t+1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, default=Path(GROUP) / "splits" / "frames32_train.jsonl")
    ap.add_argument("--datasets", default="edf,mcfd,up_fall")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--k", type=int, default=32, help="frames per clip (must match evaluation)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (feasibility test)")
    ap.add_argument("--save-every", type=int, default=25, help="optimizer steps between checkpoints")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-pixels", type=int, default=78000, help="cap on pixels per frame (must match evaluation)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "checkpoint"
    torch.manual_seed(args.seed)

    clips = load_clips(args.train, args.k, set(args.datasets.split(",")))
    random.Random(args.seed).shuffle(clips)
    total_steps = math.ceil(len(clips) * args.epochs / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    n_micro = total_steps * args.grad_accum
    n_pos = sum(c["target"] == "fall" for c in clips)
    print(f"train clips {len(clips)} ({n_pos} fall, {len(clips) - n_pos} no_fall) from {args.datasets}, "
          f"{total_steps} optimizer steps x {args.grad_accum} clips", flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.size["longest_edge"] = args.max_pixels
    pre, plan, bad = gs.label_plan(processor.tokenizer)
    assert not bad, bad

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    if (ckpt / "adapter").exists():
        model = PeftModel.from_pretrained(model, ckpt / "adapter", is_trainable=True)
        state = json.loads((ckpt / "state.json").read_text())
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
        state = {"step": 0, "micro": 0, "losses": []}
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    if (ckpt / "optimizer.pt").exists():
        sd = torch.load(ckpt / "optimizer.pt", weights_only=False, map_location="cuda:0")
        opt.load_state_dict(sd["opt"]); sched.load_state_dict(sd["sched"])
        print(f"resumed at step {state['step']}", flush=True)

    (args.out / "config.json").write_text(json.dumps(
        {**vars(args), "out": str(args.out), "train": str(args.train), "targets": TARGETS,
         "n_clips": len(clips), "total_steps": total_steps}, indent=1))

    order = [clips[i % len(clips)] for i in range(n_micro)][state["micro"]:]
    loader = torch.utils.data.DataLoader(ClipDataset(order, processor, plan, pre), batch_size=None,
                                         shuffle=False, num_workers=args.workers, prefetch_factor=2)

    def save(dest):
        model.save_pretrained(dest / "adapter")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict()}, dest / "optimizer.pt")
        (dest / "state.json").write_text(json.dumps(state))

    model.train()
    device = torch.device("cuda", 0)
    t0, acc_loss, run_steps, n_local = time.time(), 0.0, 0, 0
    torch.cuda.reset_peak_memory_stats()
    for batch in loader:
        if state["step"] >= total_steps:
            break
        loss = label_loss(model, batch, device) / args.grad_accum
        loss.backward()
        acc_loss += loss.item()
        n_local += 1
        if n_local % args.grad_accum:
            continue
        state["micro"] += args.grad_accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        state["step"] += 1
        state["losses"].append(round(acc_loss, 4))
        run_steps += 1
        print(f"step {state['step']}/{total_steps} loss {acc_loss:.4f} lr {sched.get_last_lr()[0]:.2e} | "
              f"{(time.time() - t0) / run_steps:.1f}s/step | peak mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        acc_loss = 0.0
        if state["step"] % args.save_every == 0:
            ckpt.mkdir(exist_ok=True)
            save(ckpt)

    model.save_pretrained(args.out / "adapter")
    (args.out / "losses.json").write_text(json.dumps(state["losses"]))
    print(f"done: {state['step']} steps, adapter saved to {args.out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
