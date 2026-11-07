# Fall detection results (fall vs no fall)

All methods use the official splits in `splits/`, the same camera clips, and say "fall" when P(fall) > 0.5.

- **Single camera:** each camera's clip is judged on its own. **Camera vote:** average P(fall) over an event's cameras, one decision per event.
- **Balanced accuracy:** average of the fall hit rate and the no-fall hit rate (50% = useless).
- **Recall:** share of real falls caught. **Precision:** share of "fall" alarms that were real.
- **Val** was used to choose settings; **test** was used once per final model and is the number to trust.
- **New room:** trained on the other two datasets only, so the test rooms were never seen. Closest to a real deployment.

## Summary (test, single camera, balanced accuracy)

| method | edf, room seen | edf, new room | mcfd, room seen | mcfd, new room | up_fall, room seen | up_fall, new room |
|---|---|---|---|---|---|---|
| Gemma 3 zero-shot | – | 79.8% | – | 73.0% | – | 62.3% |
| **VideoMAE** | **95.5%** | **87.5%** | **95.0%** | **90.0%** | **97.8%** | **94.4%** |
| Pose model | 86.5% | 70.9% | 83.7% | 77.6% | 86.7% | 74.3% |
| Gemma 3 + LoRA | *running* | | | | | |

Zero-shot Gemma was never trained, so every room is new to it.

**VideoMAE is best everywhere.** In new rooms it loses 3–8 points, mostly as extra false alarms.

## Methods

| method | what it is | training |
|---|---|---|
| Gemma 3 zero-shot | Gemma 3 12B-it, 32 frames, no training | none |
| VideoMAE | VideoMAE-base (Kinetics-400 pretrained), 16 frames at 224x224, all weights fine-tuned | 10 epochs, ~15 min on a 40 GB slice; lr 5e-5, best val epoch |
| Pose model | RT-DETR finds the person, ViTPose gives 17 keypoints per frame; a small temporal CNN (~0.2 M weights) classifies the 32-frame keypoint sequence | 80 epochs from scratch, ~1 min; best val epoch |
| Gemma 3 + LoRA | LoRA rank 16 on Gemma's language layers, same prompt as zero-shot | 1 epoch, ~8 h on one H100 (*running*) |

## Test results (camera vote, rooms seen in training)

| dataset | method | balanced acc. | fall recall | precision |
|---|---|---|---|---|
| edf | Gemma 3 zero-shot | 87.2% | 77.8% | 87.5% |
| edf | VideoMAE | 95.0% | 100% | 75.0% |
| edf | Pose model | 96.7% | 100% | 81.8% |
| mcfd | Gemma 3 zero-shot | 78.8% | 88.9% | 61.5% |
| mcfd | VideoMAE | 96.9% | 100% | 90.0% |
| mcfd | Pose model | 87.5% | 100% | 69.2% |
| up_fall | Gemma 3 zero-shot | 57.7% | 15.5% | 100% |
| up_fall | VideoMAE | 97.6% | 100% | 93.3% |
| up_fall | Pose model | 94.8% | 97.6% | 89.1% |

With camera vote the pose model gains a lot (it is weak per camera but the cameras' errors differ).

The camera-vote test sets are small (25–39 events for edf and mcfd), so one event moves a number by 3–4 points.

## Val results (camera vote, balanced accuracy)

| dataset | Gemma 3 zero-shot | VideoMAE | Pose model |
|---|---|---|---|
| edf | 78.5% | 93.8% | 93.8% |
| mcfd | 74.2% | 97.1% | 97.1% |
| up_fall | 52.9% | 97.8% | 93.1% |

## Caveats

- mcfd has one actor in one room, so its splits share the same person; edf (5 people) and up_fall (17 people) are split by person and say more about new people.
- Every clip is cut exactly at a labelled event's start and end. A live app sees fixed-length windows instead, so real-world accuracy still needs a test on continuous video.
- With this little data, small differences between runs are not meaningful.

## Where the files are

- Gemma zero-shot: `/scratch/wuhungma/runs/bin_{val,test}_upfall_ev_k32`, `bin_trainval_k32`, `bin_test_k32`
- VideoMAE: `/scratch/wuhungma/videomae/base_lr5e-5/` (model in `best/`); new-room runs in `holdout_{edf,mcfd,up_fall}/`
- Pose: keypoints in `/scratch/wuhungma/pose/`; classifiers in `/scratch/wuhungma/pose_clf/{all,holdout_*}/`
- Re-compute any table: `python3 eval_binary.py --split {val,test} <predictions.jsonl ...>`
