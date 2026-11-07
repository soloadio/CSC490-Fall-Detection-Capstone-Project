#!/usr/bin/env python3
"""Fall / no_fall classifier on body keypoints (output of extract_pose.py).

Input per camera clip: 32 frames x 17 COCO keypoints. Per-frame features (53 numbers):
    17 x (x, y)  keypoints, centred on the clip's mean body position and divided by the clip's
                 median person-box height, so camera distance and position do not matter but
                 movement (e.g. dropping to the floor) does
    17 x score   keypoint confidence
    box shape    width / height, and height / median height (a fallen body is wide and short)
Model: a small 1-D temporal CNN over the 32 frames (~0.2 M weights), trained from scratch.
Training augmentation: left-right flip, random scale, small noise.

The epoch with the best val balanced accuracy (single camera, threshold 0.5, on --datasets) is
kept. Predictions are written in run_probs.py format for eval_binary.py:
    <out>/val_predictions.jsonl, <out>/test_predictions.jsonl

Usage:
    python3 train_pose.py --out /scratch/wuhungma/pose_clf/all
    python3 train_pose.py --out /scratch/wuhungma/pose_clf/holdout_edf --datasets mcfd up_fall
"""
import argparse
import glob
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

POS = {"fall", "fallen"}
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO left <-> right


def load(pose_dir, split):
    kps, boxes, meta = [], [], []
    for f in sorted(glob.glob(str(pose_dir / f"pose_{split}_shard*of*.npz"))):
        z = np.load(f)
        kps.append(z["keypoints"])
        boxes.append(z["boxes"])
        meta += json.loads(str(z["meta"]))
    if not meta:
        raise SystemExit(f"no pose files for {split} in {pose_dir}")
    return np.concatenate(kps), np.concatenate(boxes), meta


def features(kps, boxes, flip=None, scale=None, noise=0.0, rng=None):
    """kps [N,32,17,3], boxes [N,32,5] -> [N,53,32] float32."""
    xy = kps[..., :2].copy()
    sc = kps[..., 2].copy()
    h = np.maximum(boxes[..., 3], 1.0)
    w = np.maximum(boxes[..., 2], 1.0)
    med_h = np.median(h, axis=1)                                    # [N]
    wts = (sc > 0.3).astype(np.float32) + 1e-3
    centre = (xy * wts[..., None]).sum((1, 2)) / wts.sum((1, 2))[..., None]   # [N,2]
    xy = (xy - centre[:, None, None]) / med_h[:, None, None, None]
    if flip is not None:
        xy[flip] = xy[flip][:, :, FLIP]
        xy[flip, ..., 0] *= -1
        sc[flip] = sc[flip][:, :, FLIP]
    if scale is not None:
        xy *= scale[:, None, None, None]
    if noise:
        xy += rng.normal(0, noise, xy.shape).astype(np.float32)
    shape = np.stack([np.log(w / h), np.log(h / med_h[:, None])], -1)  # [N,32,2]
    f = np.concatenate([xy.reshape(*xy.shape[:2], -1), sc, shape], -1)  # [N,32,53]
    return np.clip(f, -10, 10).transpose(0, 2, 1).astype(np.float32)


class TCN(nn.Module):
    def __init__(self, d_in=53, d=128, p_drop=0.3):
        super().__init__()
        layers = []
        for i, dil in enumerate((1, 2, 4)):
            layers += [nn.Conv1d(d_in if i == 0 else d, d, 5, padding=2 * dil, dilation=dil),
                       nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(p_drop)]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(2 * d, 2)

    def forward(self, x):
        h = self.body(x)
        return self.head(torch.cat([h.mean(-1), h.amax(-1)], -1))


def balanced_acc(p, y):
    pred = p > 0.5
    return 0.5 * ((pred & (y == 1)).sum() / max(1, (y == 1).sum()) + (~pred & (y == 0)).sum() / max(1, (y == 0).sum()))


@torch.no_grad()
def predict(model, x, dev):
    model.eval()
    p = torch.softmax(model(torch.from_numpy(x).to(dev)), -1)[:, 1].cpu().numpy()
    model.train()
    return p


def write_predictions(path, meta, p):
    with open(path, "w") as f:
        for m, q in zip(meta, p):
            f.write(json.dumps({**m, "probs": {"fall": round(float(q), 6), "no_fall": round(1 - float(q), 6)}}) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose-dir", type=Path, default=Path("/scratch/wuhungma/pose"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["edf", "mcfd", "up_fall"],
                    help="train AND choose the epoch on these datasets only (test predictions cover all three)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    data = {}
    for split in ("train", "val", "test"):
        k, b, m = load(args.pose_dir, split)
        if split != "test":
            keep = np.array([x["dataset"] in args.datasets for x in m])
            k, b, m = k[keep], b[keep], [x for x, t in zip(m, keep) if t]
        y = np.array([x["gold"] in POS for x in m], np.int64)
        data[split] = (k, b, m, y)
    ktr, btr, _, ytr = data["train"]
    xval = features(*data["val"][:2])
    yval = data["val"][3]
    print(f"train {len(ytr)} clips ({ytr.sum()} fall), val {len(yval)}, datasets {args.datasets}", flush=True)

    model = TCN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, history = (-1.0, 0), []
    for epoch in range(1, args.epochs + 1):
        n = len(ytr)
        x = features(ktr, btr, flip=rng.random(n) < 0.5, scale=rng.uniform(0.85, 1.15, n).astype(np.float32),
                     noise=0.02, rng=rng)
        order = rng.permutation(n)
        losses = []
        for i in range(0, n, args.batch):
            j = order[i:i + args.batch]
            loss = nn.functional.cross_entropy(model(torch.from_numpy(x[j]).to(dev)), torch.from_numpy(ytr[j]).to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        p_val = predict(model, xval, dev)
        ba = float(balanced_acc(p_val, yval))
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4), "val_bal_acc": round(ba, 4)})
        if ba > best[0]:
            best = (ba, epoch)
            torch.save(model.state_dict(), args.out / "best.pt")
            write_predictions(args.out / "val_predictions.jsonl", data["val"][2], p_val)
        if epoch % 10 == 0:
            print(f"epoch {epoch}/{args.epochs} train loss {np.mean(losses):.4f} | val balanced acc {ba:.1%}", flush=True)

    model.load_state_dict(torch.load(args.out / "best.pt"))
    write_predictions(args.out / "test_predictions.jsonl", data["test"][2], predict(model, features(*data["test"][:2]), dev))
    (args.out / "history.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                       "best_epoch": best[1], "epochs": history}, indent=1))
    print(f"best epoch {best[1]} (val balanced acc {best[0]:.1%}); predictions in {args.out}", flush=True)


if __name__ == "__main__":
    main()
