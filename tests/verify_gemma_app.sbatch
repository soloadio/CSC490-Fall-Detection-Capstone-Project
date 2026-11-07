#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=gemma_app
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/wuhungma/deploy/gemma_app_%j.log
module load gcc python/3.11 ffmpeg 2>/dev/null || module load gcc python/3.11
source /scratch/wuhungma/deploy/venv-app/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False THREADS=2
cd /scratch/wuhungma/deploy
python verify_gemma_app.py 2>&1 | grep -v -i -E 'warn|quantize_per_tensor|Loading weights'
