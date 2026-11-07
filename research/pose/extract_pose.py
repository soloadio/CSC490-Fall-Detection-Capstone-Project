#!/usr/bin/env python3
"""Extract 17 COCO body keypoints for every frame of every camera clip (all splits, all datasets).

For each of the 32 frames of a clip: RT-DETR finds people, the highest-scoring person box is
kept, and ViTPose-base estimates 17 keypoints inside it. If no person is found in a frame, the
previous frame's box is reused (or the whole image, with box score 0).

Output per shard: <out>/pose_<split>_shardIofN.npz with
    keypoints  float32 [clips, 32, 17, 3]   x, y in pixels, keypoint score
    boxes      float32 [clips, 32, 5]       x, y, w, h in pixels, detector score (0 = no person)
    image_hw   int32   [clips, 2]
    meta       json list of {event_id, dataset, scenario, cam, gold}

Usage:
    python3 extract_pose.py --shard 0/4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
DET = "/scratch/wuhungma/models/rtdetr_r50vd_coco_o365"
POSE = "/scratch/wuhungma/models/vitpose-base-simple"


class Clips(torch.utils.data.Dataset):
    def __init__(self, clips):
        self.clips = clips

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        return i, [Image.open(p).convert("RGB") for p in self.clips[i]["frames"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/pose"))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--det-threshold", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    sid, sn = map(int, args.shard.split("/"))

    from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation

    dev = torch.device("cuda")
    det_proc = AutoProcessor.from_pretrained(DET)
    det = RTDetrForObjectDetection.from_pretrained(DET).to(dev).eval()
    person = [k for k, v in det.config.id2label.items() if v == "person"]
    assert person, det.config.id2label
    person = int(person[0])
    pose_proc = AutoProcessor.from_pretrained(POSE)
    pose = VitPoseForPoseEstimation.from_pretrained(POSE).to(dev).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        out_file = args.out / f"pose_{split}_shard{sid}of{sn}.npz"
        if out_file.exists():
            print(f"{out_file} exists, skipping", flush=True)
            continue
        clips = []
        with open(HERE.parent / "splits" / f"frames32_{split}.jsonl") as f:
            for line in f:
                e = json.loads(line)
                for cam, v in e["cams"].items():
                    clips.append({"event_id": e["event_id"], "dataset": e["dataset"], "scenario": e["scenario"],
                                  "cam": cam, "gold": e["label_name"], "frames": v["frames"]})
        clips = clips[sid::sn]
        n = len(clips)
        kps = np.zeros((n, 32, 17, 3), np.float32)
        boxes = np.zeros((n, 32, 5), np.float32)
        hw = np.zeros((n, 2), np.int32)
        loader = torch.utils.data.DataLoader(Clips(clips), batch_size=None, num_workers=args.workers)
        with torch.inference_mode():
            for k, (i, imgs) in enumerate(loader, 1):
                W, H = imgs[0].size
                hw[i] = (H, W)
                d_in = det_proc(images=imgs, return_tensors="pt").to(dev)
                d_out = det(**d_in)
                res = det_proc.post_process_object_detection(
                    d_out, target_sizes=torch.tensor([(H, W)] * len(imgs)), threshold=args.det_threshold)
                prev = np.array([0, 0, W, H, 0], np.float32)
                frame_boxes = []
                for t, r in enumerate(res):
                    m = r["labels"] == person
                    if m.any():
                        j = int(r["scores"][m].argmax())
                        x1, y1, x2, y2 = r["boxes"][m][j].tolist()
                        prev = np.array([x1, y1, x2 - x1, y2 - y1, float(r["scores"][m][j])], np.float32)
                    else:
                        prev = prev.copy()
                        prev[4] = 0.0
                    boxes[i, t] = prev
                    frame_boxes.append(prev[None, :4])
                p_in = pose_proc(imgs, boxes=frame_boxes, return_tensors="pt").to(dev)
                p_out = pose(**p_in)
                p_res = pose_proc.post_process_pose_estimation(p_out, boxes=frame_boxes)
                for t, pr in enumerate(p_res):
                    kps[i, t, :, :2] = pr[0]["keypoints"].cpu().numpy()
                    kps[i, t, :, 2] = pr[0]["scores"].cpu().numpy()
                if k % 100 == 0 or k == n:
                    print(f"{split} shard {sid}/{sn}: {k}/{n} clips", flush=True)
        meta = [{k: c[k] for k in ("event_id", "dataset", "scenario", "cam", "gold")} for c in clips]
        np.savez_compressed(out_file, keypoints=kps, boxes=boxes, image_hw=hw, meta=json.dumps(meta))
        print(f"wrote {out_file}", flush=True)


if __name__ == "__main__":
    main()
