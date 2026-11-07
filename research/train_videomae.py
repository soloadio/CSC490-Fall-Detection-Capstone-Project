#!/usr/bin/env python3
"""Fine-tune VideoMAE-base (Kinetics-400 pretrained) for fall / no_fall on single-camera clips.

Same data as the Gemma runs: splits/frames32_{train,val,test}.jsonl (make_frames_manifest.py),
one example per camera clip, gold fall = OmniFall labels {fall, fallen} by default
(--pos fall counts only the falling motion; `fallen`, i.e. lying after a fall, is then no_fall). VideoMAE takes 16 frames,
so every other one of the 32 extracted frames is used (even indices 0, 2, ..., 30).

Preprocessing: resize the short side to 224 and take a 224x224 crop (random crop + horizontal flip
when training, centre crop otherwise), ImageNet mean/std normalisation.

After every epoch the model is scored on val; the epoch with the best val balanced accuracy
(single camera, threshold 0.5) is kept. Predictions are written in run_probs.py format so that
eval_binary.py scores them exactly like the Gemma results:
    <out>/val_predictions.jsonl     best epoch on val
    <out>/test_predictions.jsonl    best epoch on test (only with --test)

Usage:
    python3 train_videomae.py --out /scratch/wuhungma/videomae/base_lr5e-5
    python3 train_videomae.py --out ... --test       # also score test with the chosen epoch
    python3 train_videomae.py --out ... --datasets edf mcfd --test   # held-out-dataset test
    python3 train_videomae.py --out ... --predict-only   # test predictions from an already-trained <out>/best
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL = "/scratch/wuhungma/models/videomae-base-finetuned-kinetics"
POS = {"fall", "fallen"}
FRAME_IDX = list(range(0, 32, 2))
SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)


def load_clips(split, pos=POS):
    clips = []
    with open(HERE / "splits" / f"frames32_{split}.jsonl") as f:
        for line in f:
            e = json.loads(line)
            for cam, v in e["cams"].items():
                clips.append({"event_id": e["event_id"], "dataset": e["dataset"], "scenario": e["scenario"],
                              "cam": cam, "gold": e["label_name"], "y": int(e["label_name"] in pos),
                              "frames": [v["frames"][i] for i in FRAME_IDX]})
    return clips


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, clips, train):
        self.clips, self.train = clips, train

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        c = self.clips[i]
        imgs = [Image.open(p).convert("RGB") for p in c["frames"]]
        w, h = imgs[0].size
        s = SIZE / min(w, h)
        nw, nh = max(SIZE, round(w * s)), max(SIZE, round(h * s))
        if self.train:
            x0, y0 = random.randint(0, nw - SIZE), random.randint(0, nh - SIZE)
            flip = random.random() < 0.5
        else:
            x0, y0, flip = (nw - SIZE) // 2, (nh - SIZE) // 2, False
        out = []
        for im in imgs:
            im = im.resize((nw, nh), Image.BILINEAR).crop((x0, y0, x0 + SIZE, y0 + SIZE))
            if flip:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            out.append(np.asarray(im))
        x = torch.from_numpy(np.stack(out)).permute(3, 0, 1, 2).float() / 255  # C, T, H, W
        x = (x - MEAN) / STD
        return x.permute(1, 0, 2, 3), c["y"], i  # VideoMAE wants T, C, H, W


@torch.no_grad()
def predict(model, clips, args, device):
    model.eval()
    loader = torch.utils.data.DataLoader(ClipDataset(clips, train=False), batch_size=args.batch * 2,
                                         num_workers=args.workers)
    p_fall = [0.0] * len(clips)
    for x, _, idx in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pixel_values=x.to(device, non_blocking=True)).logits.float()
        for i, p in zip(idx.tolist(), torch.softmax(logits, -1)[:, 1].tolist()):
            p_fall[i] = p
    model.train()
    return p_fall


def balanced_acc(p, clips):
    tp = sum(q > 0.5 and c["y"] for q, c in zip(p, clips))
    tn = sum(q <= 0.5 and not c["y"] for q, c in zip(p, clips))
    n1 = sum(c["y"] for c in clips)
    return 0.5 * (tp / n1 + tn / (len(clips) - n1))


def write_predictions(path, clips, p):
    with open(path, "w") as f:
        for c, q in zip(clips, p):
            f.write(json.dumps({"event_id": c["event_id"], "dataset": c["dataset"], "scenario": c["scenario"],
                                "cam": c["cam"], "gold": c["gold"],
                                "probs": {"fall": round(q, 6), "no_fall": round(1 - q, 6)}}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", action="store_true", help="also predict the test split with the best epoch")
    ap.add_argument("--datasets", nargs="+", default=["edf", "mcfd", "up_fall"],
                    help="train AND choose the epoch on these datasets only (test predictions cover all three)")
    ap.add_argument("--pos", nargs="+", default=sorted(POS),
                    help="OmniFall labels counted as fall (default: fall fallen)")
    ap.add_argument("--predict-only", action="store_true", help="skip training; predict test with <out>/best")
    args = ap.parse_args()

    from transformers import VideoMAEForVideoClassification

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    if args.predict_only:
        args.test = True
        args.epochs = 0
    pos = set(args.pos)
    train = [c for c in load_clips("train", pos) if c["dataset"] in args.datasets]
    val = [c for c in load_clips("val", pos) if c["dataset"] in args.datasets]
    print(f"train {len(train)} clips ({sum(c['y'] for c in train)} fall), val {len(val)} clips", flush=True)

    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL, num_labels=2, id2label={0: "no_fall", 1: "fall"}, label2id={"no_fall": 0, "fall": 1},
        ignore_mismatched_sizes=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(ClipDataset(train, train=True), batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=True, persistent_workers=True)
    total = max(1, args.epochs * len(loader))
    warmup = len(loader)  # one epoch of linear warm-up, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))

    history, best = [], (-1.0, None)
    for epoch in range(1, args.epochs + 1):
        t0, losses = time.time(), []
        for x, y, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(pixel_values=x.to(device, non_blocking=True)).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())
        p_val = predict(model, val, args, device)
        ba = balanced_acc(p_val, val)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val_bal_acc": round(ba, 4)})
        print(f"epoch {epoch}/{args.epochs} train loss {np.mean(losses):.4f} | val balanced acc {ba:.1%} | "
              f"{time.time() - t0:.0f}s", flush=True)
        if ba > best[0]:
            best = (ba, epoch)
            model.save_pretrained(args.out / "best")
            write_predictions(args.out / "val_predictions.jsonl", val, p_val)

    if not args.predict_only:
        (args.out / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                           "best_epoch": best[1], "epochs": history}, indent=1))
        print(f"best epoch {best[1]} (val balanced acc {best[0]:.1%}); model in {args.out / 'best'}", flush=True)

    if args.test:
        model = VideoMAEForVideoClassification.from_pretrained(args.out / "best").to(device)
        test = load_clips("test", pos)
        write_predictions(args.out / "test_predictions.jsonl", test, predict(model, test, args, device))
        print(f"test predictions: {args.out / 'test_predictions.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
