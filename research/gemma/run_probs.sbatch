#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=gemma_probs
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/wuhungma/runs/logs/log_%A_%a.log
# usage: sbatch --export=ALL,EXTRA="--manifest splits/frames32_val.jsonl --scoring gemma_binary_scoring --tag NAME [--adapter DIR]" run_probs.sbatch
module load gcc python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 run_probs.py --k ${K:-32} --shard ${SLURM_ARRAY_TASK_ID}/4 ${EXTRA}
