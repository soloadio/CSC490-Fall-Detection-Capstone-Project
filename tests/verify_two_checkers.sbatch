#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=two_checkers
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/wuhungma/deploy/two_checkers_%j.log
module load gcc python/3.11 ffmpeg 2>/dev/null || module load gcc python/3.11
source /scratch/wuhungma/deploy/venv-app/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False THREADS=2
cd /scratch/wuhungma/deploy
python verify_two_checkers.py 2>&1 | grep -v -i -E 'warn|quantize_per_tensor|Loading weights'
