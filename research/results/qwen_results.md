# Qwen2-VL-7B-Instruct results (fall vs no fall)

Code: `research/qwen/`. Same prompt as the Gemma runs (`research/gemma/gemma_binary_scoring.py`), 32 frames per clip,
same official splits and clips, scored with `research/gemma/eval_binary.py`. Says "fall" when P(fall) > 0.5.

- **Zero-shot:** Qwen2-VL-7B-Instruct, no training (run 2026-10-19/20).
- **LoRA:** rank 16 on the language layers, 1 epoch on `frames32_train.jsonl` (edf + mcfd + up_fall, 2967 clips),
  about 55 min on one H100 (run 2026-10-20, evaluated 2026-11-07). Training loss: 0.21 (first 50 steps) -> 0.09 (last 50).
- **Camera vote:** average P(fall) over an event's cameras, one decision per event. **Balanced accuracy:** average of
  the fall hit rate and the no-fall hit rate. **Recall:** share of real falls caught. **Precision:** share of alarms that were real.

## Test results (camera vote)

| dataset | method | balanced acc. | fall recall | precision | AUROC |
|---|---|---|---|---|---|
| edf | Qwen2-VL zero-shot | 78.3% | 100% | 40.9% | 0.980 |
| edf | **Qwen2-VL + LoRA** | **96.7%** | **100%** | **81.8%** | **0.993** |
| edf | Gemma 3 + LoRA | 96.7% | 100% | 81.8% | 0.993 |
| mcfd | Qwen2-VL zero-shot | 80.2% | 66.7% | 85.7% | 0.896 |
| mcfd | **Qwen2-VL + LoRA** | **93.8%** | **100%** | **81.8%** | **1.000** |
| mcfd | Gemma 3 + LoRA | 93.8% | 100% | 81.8% | 1.000 |
| up_fall | Qwen2-VL zero-shot | 95.8% | 98.8% | 90.2% | 0.985 |
| up_fall | **Qwen2-VL + LoRA** | **95.2%** | **100%** | **87.5%** | **0.990** |
| up_fall | Gemma 3 + LoRA | 96.4% | 100% | 90.3% | 0.991 |

Gemma 3 + LoRA numbers are from `lora_results.md` (same clips). For zero-shot Gemma and VideoMAE see `results.md`.

## Val results (camera vote, balanced accuracy)

| dataset | Qwen2-VL zero-shot | Qwen2-VL + LoRA |
|---|---|---|
| edf | 86.7% | 92.1% |
| mcfd | 77.5% | 100% |
| up_fall | 91.0% | 92.9% |

## Findings

1. **LoRA fixes Qwen's weak spots.** edf goes from 78.3% to 96.7%: zero-shot caught every fall but half its alarms were false.
   mcfd goes from 80.2% to 93.8%: zero-shot missed a third of the falls, LoRA catches all of them.
2. **On UP-Fall, zero-shot Qwen was already good** (95.8%, against 57.7% for zero-shot Gemma), and LoRA keeps it there.
3. **Qwen 7B + LoRA matches Gemma 12B + LoRA** on edf and mcfd (identical camera-vote numbers) and is about 1 point behind
   on UP-Fall. It does this with a smaller model and much shorter training (55 min against about 8 h).
4. **The threshold matters little after LoRA.** Tuned on val, it is 0.561: edf test 96.7% -> 98.3%, mcfd and up_fall
   unchanged. Before LoRA it was 0.545 and helped edf only (78.3% -> 81.7%).

## Caveats

- Camera-vote test sets for edf and mcfd are small (39 and 25 events), so one event moves a number by 3–4 points.
- mcfd has one actor in one room, so its splits share the same person.
- Clips are cut at labelled event boundaries; continuous-video (streaming) tests have not been run for Qwen.
- up_fall frames (640x480) are scaled to 308x224 for Qwen (`--max-pixels 78000`); edf/mcfd frames are unchanged.

## Where the files are (Alliance Fir cluster)

- Zero-shot predictions: `/home/soloadio/qwen/runs/qwen_bin_k32/` (edf, mcfd, occu), `qwen_bin_upfall_{val,test}_k32/`
- LoRA predictions: `/home/soloadio/qwen/runs/qwen_lora_r16_all3_{val,test}/`; adapter: `/home/soloadio/qwen/lora/r16_all3/adapter`
- Re-compute: `python3 research/gemma/eval_binary.py --split {val,test} <results_shard*.jsonl>`
