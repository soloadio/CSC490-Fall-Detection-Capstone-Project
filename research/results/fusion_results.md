# Two cameras, one room: does combining views help?

Run 2026-10-20. Model: VideoMAE "fall only" (`stream_results_fallonly.md`), same sliding-window scores
(`/scratch/wuhungma/stream_eval/videomae_fallonly/`), no new model runs. Script: `fusion_metrics.py`.

## Setup

Every test recording was filmed by several cameras at once (EDF 2 views, MCFD 8, UP-Fall 2). For each **pair of
cameras of the same take** (what two phones in a room would see; MCFD gives 28 pairs per take) the two P(fall) series
are put on one clock and combined:

| rule | room score at time t | meaning |
|---|---|---|
| single | each camera alone | baseline (both cameras of every pair, so all rules see the same video) |
| any | max of the two | alarm if either camera sees a fall |
| mean | average | |
| **two agree** | the lower of the two cameras' highest P(fall) in the last 1 s | alarm only if both cameras saw P >= threshold within 1 s |

The combined score goes through the same alarm rule and the same fall / false-alarm scoring as `stream_metrics.py`
(cooldown 10 s). Clocks: UP-Fall cameras are synchronised; EDF views are 5 to 6 s apart and MCFD cameras up to 3.6 s
(median 0.5 s), so each camera's offset was estimated from the labels (a live app timestamps frames on arrival, which
gives the same shared clock). Only time recorded by both cameras is scored.

## Test results, deployed setting (3 s window, k = 1, threshold 0.30)

| rule | falls caught | recall | false alarms | false alarms / hour | delay median |
|---|---|---|---|---|---|
| single camera | 353 / 386 | 91.5% | 24 | 11.3 | 1.5 s |
| any | 172 / 193 | 89.1% | 22 | 20.8 | 1.5 s |
| mean | 178 / 193 | 92.2% | 16 | 15.1 | 1.5 s |
| **two agree** | **182 / 193** | **94.3%** | **2** | **1.9** | 1.8 s |

(Single counts each camera of a pair separately, hence twice the falls and twice the non-fall time.)

By dataset, two agree vs single: MCFD 94.3% vs 90.4% recall, 1 vs 20 false alarms; UP-Fall 100% vs 100%, 0 vs 2;
EDF 5/8 vs 10/16 (its misses are the back-to-back-falls cooldown artifact in `stream_results.md`).

With settings chosen on val per rule (target <= 30 or <= 5 false alarms/hour), two agree picks a 2 s window and
threshold 0.75: 94.3% recall, 1 false alarm (0.9/h), median delay 1.5 s. Single camera at its best val setting for
<= 5/h: 93.0%, 8.97/h on test.

## Findings

1. **"Two agree" is the rule to use.** It keeps (slightly raises) recall and cuts false alarms about 6x: a false
   alarm needs both views to be fooled at the same moment, a real fall is seen by both.
2. **"Any" is worse than one camera.** It adds both cameras' false alarms, and each false alarm starts a 10 s
   cooldown that can swallow a real fall.
3. It costs about 0.3 s of extra delay at the deployed setting (the second camera has to agree).
4. With all 8 MCFD cameras (`--group all`) the numbers are similar; most homes would have 2.

## Caveats

- Non-fall time is small: about 1 h of paired camera time, mostly UP-Fall; MCFD pairs reuse the same 4 takes 28
  times. The false-alarm counts (2 vs 24) are the main evidence, not the per-hour rates.
- Lab rooms, both cameras usually see the whole person. In a real room one camera may be blocked; "two agree"
  then misses the fall unless the other view catches it alone (the app falls back to the single-camera rule when
  only one camera is connected).
- Offsets estimated from labels; a live system shares a clock but adds network jitter (~0.1 to 0.3 s), within the 1 s
  agreement window.

## Reproduce
```
sbatch --export=ALL,GROUP=pairs /scratch/wuhungma/stream_eval/fusion/fusion.sbatch   # and GROUP=all
# runs: python3 fusion_metrics.py --select <scores_val> --report <scores_test> --group pairs \
#     --cooldown 10 --max-fa-per-hour 30 5 --json /scratch/wuhungma/stream_eval/fusion/fusion_pairs.json
```
