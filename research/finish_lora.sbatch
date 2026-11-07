#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=lora_summary
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/wuhungma/runs/logs/lora_summary_%j.log
# Runs after the LoRA val/test evaluations (even if they failed) and writes results/lora_results.md.
# usage: sbatch --dependency=afterany:<val job>:<test job> --export=ALL,TRAIN_JOB=<id>,VAL_JOB=<id>,TEST_JOB=<id> finish_lora.sbatch
module load python/3.11
cd /project/6010122/group12
R=/scratch/wuhungma/runs
OUT=results/lora_results.md
VAL=$R/bin_val_lora_r16_ep1
TEST=$R/bin_test_lora_r16_ep1

for d in $VAL $TEST; do
  if ls $d/results_shard*of4.jsonl >/dev/null 2>&1; then cat $d/results_shard*of4.jsonl > $d/results.jsonl; fi
done

{
  echo "# Gemma 3 + LoRA results (written automatically $(date '+%Y-%m-%d %H:%M %Z'))"
  echo
  echo "LoRA rank 16, 1 epoch, adapter: \`/scratch/wuhungma/lora/r16_bf16_ep1/adapter\`."
  echo "Same prompt, clips and 0.5 threshold as zero-shot Gemma. Slurm jobs: train $TRAIN_JOB, val $VAL_JOB, test $TEST_JOB."
  echo
  echo "## Job status"
  echo '```'
  sacct -n -X -j $TRAIN_JOB,$VAL_JOB,$TEST_JOB -o JobID%16,JobName%14,State%12,Elapsed
  echo '```'
  grep -h "done:" $R/logs/lora_$TRAIN_JOB.log 2>/dev/null
  for split in val test; do
    d=$R/bin_${split}_lora_r16_ep1
    echo
    echo "## $split: LoRA"
    if [ -s $d/results.jsonl ]; then
      echo "Rows: $(wc -l < $d/results.jsonl), errors: $(grep -c '"error"' $d/results.jsonl)"
      echo '```'
      python3 eval_binary.py --split $split $d/results.jsonl
      echo '```'
    else
      echo "**No results** — the evaluation did not run or failed; see the job status above and $R/logs/."
    fi
  done
  echo
  echo "## For comparison: zero-shot Gemma (same clips)"
  echo '```'
  python3 eval_binary.py --split val $R/bin_trainval_k32/results_shard*of4.jsonl $R/bin_val_upfall_ev_k32/results.jsonl
  echo
  python3 eval_binary.py --split test $R/bin_test_k32/results.jsonl $R/bin_test_upfall_ev_k32/results.jsonl
  echo '```'
} > $OUT 2>&1
echo "wrote $OUT"
