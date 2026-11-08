# Qwen2-VL-7B-Instruct (zero-shot and LoRA)

Same task and prompt as `research/gemma/` (fall vs no fall), with Qwen2-VL-7B-Instruct in place of Gemma 3 12B.
The prompt, label definitions and the forced answer prefix `{"verdict": "` come unchanged from
`research/gemma/gemma_binary_scoring.py` (or `gemma_scoring.py` for the 10 classes), and the output files have the
same format as `run_probs.py`, so `research/gemma/eval_binary.py` scores them exactly like the Gemma results.
Results: `research/results/qwen_results.md`.

| File | What it does |
|---|---|
| `run_probs_qwen.py` | P(label) for every camera clip: one prefill pass, then teacher-forced label tokens on a copy of the KV cache (as `gemma_scoring.score_labels`). `--adapter` adds a trained LoRA in memory |
| `train_lora_qwen.py` | LoRA on the language model only (vision tower frozen); loss = cross-entropy on the label tokens after the forced prefix, as `research/gemma/train_lora.py` |
| `tune_threshold.py` | Picks the P(fall) threshold on val (camera vote), reports val and test |
| `eval.py` | Quick metrics for one run folder (10-class and binary) |
| `run_qwen.sbatch`, `train_qwen.sbatch` | Slurm jobs (1× H100) |

## Settings

- Model `Qwen/Qwen2-VL-7B-Instruct` (Hugging Face cache, `HF_HUB_OFFLINE=1` on compute nodes), bf16.
- 32 frames per clip, frame tags as in the Gemma prompt.
- `--max-pixels 78000` per frame. edf/mcfd 320x240 frames are unchanged (99 image tokens each); up_fall 640x480 is
  scaled to 308x224 (88 tokens), about the same size.
- LoRA: rank 16, alpha 32, dropout 0.05, q/k/v/o/gate/up/down projections of the language model, lr 1e-4 with cosine
  decay, 8 clips per step, 1 epoch over `splits/frames32_train.jsonl` (edf + mcfd + up_fall, 2967 clips, 371 steps,
  about 55 min and 20 GiB on one H100).

## Run

```bash
# zero-shot, fall / no_fall, one split
sbatch run_qwen.sbatch --scoring gemma_binary_scoring --manifest <splits>/frames32_val.jsonl --tag qwen_bin_val
# LoRA training, then the same scoring with the adapter
sbatch train_qwen.sbatch --out <dir>/lora/r16_all3
sbatch run_qwen.sbatch --scoring gemma_binary_scoring --manifest <splits>/frames32_test.jsonl \
       --adapter <dir>/lora/r16_all3/adapter --tag qwen_lora_test
# metrics
python3 ../gemma/eval_binary.py --split test <out-root>/qwen_lora_test/results_shard*.jsonl
python3 tune_threshold.py <val and test result files>
```

`--check` scores the first clip a second time without the KV cache and prints the difference (about 1e-5).
Job files and default paths are from the Alliance Fir cluster (`/project/6010122/group12`, `/home/soloadio/qwen`);
change them to run elsewhere.

## Notes for transformers 5.x

Qwen2-VL needs `mm_token_type_ids` from the processor for its 3-D rotary positions, padded with 0 for any tokens
appended after the prompt. When decoding from a KV cache, transformers builds position ids from the full attention
mask, so `score_labels` passes them explicitly (cache length + the stored `rope_deltas`).

## Weights

Not in git (see `.gitignore`). The LoRA adapter (161 MB) is at `/home/soloadio/qwen/lora/r16_all3/adapter` on the cluster.
