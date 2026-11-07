#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=verify_gemma
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wuhungma/verify_gemma/%x_%j.log
# GPU stage: Gemma 3 12B (zero-shot, then LoRA) scores the clips.   usage: sbatch [--dependency=afterok:<frames job>] verify_score.sbatch
module load gcc arrow/24.0.0 python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 verify_alarms.py score
