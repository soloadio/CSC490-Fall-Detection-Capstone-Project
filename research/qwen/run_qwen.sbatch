#!/bin/bash
#SBATCH --job-name=qwen_probs
#SBATCH --account=def-ajbonner_gpu
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/soloadio/qwen/logs/%x_%j.out

# Usage (all args go to run_probs_qwen.py):
#   sbatch run_qwen.sbatch --limit 5 --check --with-text                         smoke test, 10 classes
#   sbatch run_qwen.sbatch --scoring gemma_binary_scoring --limit 5 --check      smoke test, fall / no_fall
#   sbatch run_qwen.sbatch                                                        full 10-class run
#   for i in 0 1 2 3; do sbatch run_qwen.sbatch --shard $i/4; done              full run on 4 GPUs
set -e
module load python/3.11 2>/dev/null || module load python
source /home/soloadio/self/.venv/bin/activate
export HF_HOME=/home/soloadio/.cache/huggingface
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
python -u /home/soloadio/qwen/run_probs_qwen.py "$@"
