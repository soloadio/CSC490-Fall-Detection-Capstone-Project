#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=verify_5fps
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/wuhungma/deploy/verify_5fps_%j.log
module load gcc python/3.11
source /scratch/wuhungma/deploy/venv-app/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
python /scratch/wuhungma/deploy/verify_5fps.py
