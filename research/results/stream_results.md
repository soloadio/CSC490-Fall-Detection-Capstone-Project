# VideoMAE on continuous video (sliding window)

Run 2026-10-20. Model: VideoMAE-base `/scratch/wuhungma/videomae/base_lr5e-5/best` (the model in `results.md`),
team splits in `splits/`. Unlike every other table so far, nothing is cut at event boundaries: each whole
camera recording is replayed and scored like a live camera.

## How it works

- **Windows:** 2, 3 or 4 s long, one every 0.5 s. P(fall) is stamped at the window's end (when a live system would know).
  Before a recording starts, windows repeat the first frame.
- **Alarm:** P(fall) >= threshold for k windows in a row; then silent until the score drops below the threshold
  and the cooldown has passed.
- **Fall caught:** an alarm between 0.5 s before the labelled `fall` onset and 2 s after the fall/fallen period ends.
  **Delay** = alarm time minus onset. **False alarm:** an alarm outside any fall period, in labelled time.
- **Settings chosen on val, run once on test.** Chosen: 2 s window, k = 2, threshold 0.35, cooldown 10 s.

## Results

| split | dataset | recordings | falls | caught | recall | false alarms | non-fall hours | false alarms / hour | delay median | delay p90 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | EDF | 2 | 18 | 16 | 88.9% | 2 | 0.04 | (45) | 2.4 s | 3.1 s |
| val | MCFD | 32 | 40 | 39 | 97.5% | 5 | 0.09 | (58) | 1.7 s | 3.6 s |
| val | UP-Fall | 132 | 60 | 60 | 100% | 22 | 0.95 | 23 | 1.5 s | 2.0 s |
| val | **all** | 166 | 118 | 115 | **97.5%** | 29 | 1.08 | **27** | **1.6 s** | 2.9 s |
| test | EDF | 2 | 16 | 9 | 56.2%* | 3 | 0.04 | (77) | 1.8 s | 2.8 s |
| test | MCFD | 32 | 40 | 37 | 92.5% | 3 | 0.10 | (31) | 1.9 s | 2.2 s |
| test | UP-Fall | 198 | 90 | 90 | 100% | 53 | 1.41 | 38 | 1.5 s | 1.9 s |
| test | **all** | 232 | 146 | 136 | **93.2%** | 59 | 1.54 | **38** | **1.6 s** | 2.1 s |

False alarms per hour in brackets rest on 2–6 minutes of non-fall video and mean little.

\* **EDF test misses are an evaluation artifact, not model misses.** The actor falls every ~8.5 s; VideoMAE scores
P(fall) = 1.00 on all 16 falls, but 7 came while the alarm was still in its 10 s cooldown from the previous fall.
Scripted datasets with back-to-back falls make recall depend on the cooldown (with 30 s, val recall drops to 83%).

## Findings

1. **Falls are detected, fast.** Practically every fall reaches P(fall) ~ 1; the alarm comes a median 1.6 s after onset.
2. **False alarms are the problem, and they have one cause: lying down on purpose.** In UP-Fall:

   | activity | val: recordings with a false alarm | test |
   |---|---|---|
   | laying (Activity 11) | **12 / 12** | **18 / 18** |
   | walking | 0 / 12 | 3 / 18 |
   | standing, sitting, picking up an object, jumping | 0 / 48 | 0 / 72 |

   Every intentional lie-down triggers an alarm. Raising the threshold does not help (no setting reached 1 false
   alarm/hour on val, even at 0.995): the model is fully confident about lying people.
3. **The clip-level numbers hid this.** In `results.md`, "laying" clips are a small share of non-fall clips; on a
   timeline, a person lying down for 30 s produces alarm after alarm.

## Caveats

- Only ~1.5 h of non-fall video per split, all scripted. Real homes have hours of quiet footage; real false alarm
  rates need recordings of ordinary days.
- Lying on a bed or couch is common at home, so finding 2 matters more in deployment than here.
- Single camera only; no multi-camera fusion yet.

## Next steps

- Teach the model the fall vs lying-down difference: more `lie_down`/`lying` negatives in training (OmniFall has
  them), training on sliding windows rather than cut events, or a second stage that looks at the seconds *before*
  the person is on the floor (fast drop vs controlled lowering).
- Check whether camera fusion or Gemma + LoRA as a verifier rejects the lie-down alarms.

## Reproduce

```
sbatch prepare_stream_videos.sbatch                          # EDF/MCFD MP4s -> /scratch/wuhungma/stream_eval/videos
sbatch --export=ALL,ARGS="--split val" eval_stream.sbatch    # and --split test (~3 min each)
python3 stream_metrics.py --select /scratch/wuhungma/stream_eval/videomae_base/scores_val.jsonl \
    --report /scratch/wuhungma/stream_eval/videomae_base/scores_test.jsonl --cooldown 10 --max-fa-per-hour 30
```
