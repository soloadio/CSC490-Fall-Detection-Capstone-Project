#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=qwen_alarms
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wuhungma/verify_qwen/%x_%j.log
# Qwen2-VL-7B + Solomon's LoRA on the VideoMAE alarm clips (single- and multi-camera), as verify_reasons_score.sbatch.
set -e
module load gcc arrow/24.0.0 python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 verify_qwen.py --out /scratch/wuhungma/verify_qwen
