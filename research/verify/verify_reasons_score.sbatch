#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=reasons_eval
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/wuhungma/verify_gemma_reasons/%x_%j.log
# The LoRA retrained with reasons, on the same alarm clips as the old adapter (single- and multi-camera).
set -e
module load gcc arrow/24.0.0 python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
A=/scratch/wuhungma/lora/r16_reasons_ep1/adapter
python3 verify_alarms.py score --out /scratch/wuhungma/verify_gemma_reasons --models lora --adapter $A
python3 verify_alarms.py score --out /scratch/wuhungma/verify_gemma_reasons --clips clips_mv.jsonl --models lora --adapter $A
