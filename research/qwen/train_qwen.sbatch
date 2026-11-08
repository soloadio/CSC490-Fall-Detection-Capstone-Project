#!/bin/bash
#SBATCH --job-name=qwen_lora
#SBATCH --account=def-ajbonner_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/soloadio/qwen/logs/%x_%j.out

# Usage (all args go to train_lora_qwen.py):
#   sbatch --time=00:30:00 train_qwen.sbatch --out /home/soloadio/qwen/lora/smoke --max-steps 5    feasibility test
#   sbatch train_qwen.sbatch --out /home/soloadio/qwen/lora/r16                                    full run (resumes)
set -e
module load python/3.11 2>/dev/null || module load python
source /home/soloadio/self/.venv/bin/activate
export HF_HOME=/home/soloadio/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
python -u /home/soloadio/qwen/train_lora_qwen.py "$@"
