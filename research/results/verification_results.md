# VideoMAE raises the alarm, Gemma checks it

Run 2026-11-03. Scripts: `verify_alarms.py`, `verify_mv.py`. Outputs: `/scratch/wuhungma/verify_gemma/` (`report.txt`, `report_mv.txt`).

## Setup

VideoMAE "fall only" at the deployed setting (3 s window, k = 1, threshold 0.30) raises alarms on the recordings. For
each alarm, Gemma 3 12B-it, without (zero-shot) or with the LoRA adapter, looks at 32 frames spanning 3 s and answers
fall / no_fall. The alarm stands if P(fall) >= 0.5. A "kept fall" is a real fall whose alarm survives the check.

## One camera (test: 136 real-fall alarms, 7 false alarms; val: 115 and 4)

| Gemma | clip ends | test falls kept | test false alarms removed | val falls kept | val false alarms removed |
|---|---|---|---|---|---|
| zero-shot | at the alarm | 97 / 136 | 6 / 7 | 86 / 115 | 4 / 4 |
| zero-shot | 1 s after | 94 / 136 | 6 / 7 | 75 / 115 | 4 / 4 |
| LoRA | at the alarm | 108 / 136 | 4 / 7 | 88 / 115 | 2 / 4 |
| **LoRA** | **1 s after** | **132 / 136** | **5 / 7** | **103 / 115** | **2 / 4** |

## Several cameras (up to 3 per take, clip ends 1 s after; test: 55 fall events, 4 false-alarm events; val: 43 and 1)

| rule | zero-shot: test falls kept | zero-shot: false alarms removed | LoRA: test falls kept | LoRA: false alarms removed | LoRA: val falls kept | LoRA: val false alarms removed |
|---|---|---|---|---|---|---|
| VideoMAE only | 55 / 55 | 0 / 4 | 55 / 55 | 0 / 4 | 43 / 43 | 0 / 1 |
| first camera alone | 45 / 55 | 4 / 4 | 51 / 55 | 4 / 4 | 39 / 43 | 1 / 1 |
| any camera confirms | 50 / 55 | 4 / 4 | 54 / 55 | 3 / 4 | 42 / 43 | 0 / 1 |
| majority confirms | 27 / 55 | 4 / 4 | 50 / 55 | 4 / 4 | 39 / 43 | 1 / 1 |
| all cameras confirm | 25 / 55 | 4 / 4 | 50 / 55 | 4 / 4 | 39 / 43 | 1 / 1 |

## Findings

1. LoRA is what makes Gemma usable as a checker: zero-shot cancels about 30% of real falls on one camera.
2. Letting the clip run 1 s past the alarm helps LoRA a lot (132 vs 108 falls kept on test).
3. Requiring several cameras to agree hurts zero-shot badly and costs LoRA little.

## Caveats

- Very few false alarms: 7 and 4 on test, 4 and 1 on val. One or two events change these rates, so the false-alarm
  removal is suggestive only. The val rate for one camera (2 / 4) is much lower than test (5 / 7).
- Verdicts only: the cameras did not read each other's explanations.
- Offline recordings with full video; the live app receives about 5 frames/s, so its 32 frames contain repeats.
