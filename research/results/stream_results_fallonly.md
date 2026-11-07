# VideoMAE "fall only" on continuous video

Run 2026-10-20. Same evaluation as `stream_results.md` (whole recordings, sliding windows every 0.5 s, settings
chosen on val, applied once to test, cooldown 10 s, `--max-fa-per-hour 30`), same splits.

**Only change:** the training label. The original model counts OmniFall `fall` **and `fallen`** (lying on the floor
after a fall) as "fall". A person lying after a fall looks the same as one who lay down on purpose, so the model
learned "person on the floor = fall" and alarmed on every intentional lie-down. The new model counts only `fall`
(the falling motion) as positive; `fallen`, `lie_down`, `lying` and everything else are "no fall".

- Training: `train_videomae.py --pos fall --lr 5e-5 --test` (job 66135893), 580 positive of 2,967 train clips, best epoch 6
  (val clip balanced accuracy 97.6%). Model: `/scratch/wuhungma/videomae/fallonly_lr5e-5/best`.
- Streaming scores: jobs 66135894 (val), 66135895 (test), in `/scratch/wuhungma/stream_eval/videomae_fallonly/`.
- Chosen on val: **3 s window, k = 1, threshold 0.30** (old model: 2 s, k = 2, 0.35).

## Test results (run once)

| model | falls caught | recall | false alarms | false alarms / hour | delay median | delay p90 |
|---|---|---|---|---|---|---|
| old (fall + fallen) | 136 / 146 | 93.2% | 59 | 38 | 1.6 s | 2.1 s |
| **new (fall only)** | 136 / 146 | 93.2% | **7** | **4.5** | **1.2 s** | 2.1 s |

By dataset (new model): EDF 10/16 caught, 2 false alarms; MCFD 36/40, 3; UP-Fall 90/90, 2 (old model: 53).

UP-Fall "laying" (Activity 11) recordings with at least one alarm: old model 12/12 val, 18/18 test; **new model 0/12 val, 0/18 test.**

## Caveats
- Non-fall video is only about 1.5 h per split and scripted; EDF and MCFD false-alarm rates rest on a few minutes.
- EDF misses are mostly the cooldown artifact described in `stream_results.md` (back-to-back scripted falls).
- A person who is already on the floor when the camera starts (`fallen` without the fall) is no longer flagged by design.
- Single camera; same rooms as training for EDF/MCFD/UP-Fall (split by person, not by room), no new-room test yet.

## Reproduce
```
sbatch --export=ALL,ARGS="--out /scratch/wuhungma/videomae/fallonly_lr5e-5 --lr 5e-5 --pos fall --test" train_videomae.sbatch
sbatch --export=ALL,ARGS="--split val --model /scratch/wuhungma/videomae/fallonly_lr5e-5/best --out /scratch/wuhungma/stream_eval/videomae_fallonly" eval_stream.sbatch
sbatch --export=ALL,ARGS="--split test --model /scratch/wuhungma/videomae/fallonly_lr5e-5/best --out /scratch/wuhungma/stream_eval/videomae_fallonly" eval_stream.sbatch
python3 stream_metrics.py --select /scratch/wuhungma/stream_eval/videomae_fallonly/scores_val.jsonl \
    --report /scratch/wuhungma/stream_eval/videomae_fallonly/scores_test.jsonl --cooldown 10 --max-fa-per-hour 30
```
