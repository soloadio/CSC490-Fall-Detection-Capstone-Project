#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=videomae
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wuhungma/runs/logs/videomae_%j.log
# usage: sbatch --export=ALL,ARGS="--out /scratch/wuhungma/videomae/lr5e-5 --lr 5e-5" train_videomae.sbatch
module load gcc python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 train_videomae.py ${ARGS}
