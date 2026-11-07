#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=eval_stream
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/wuhungma/runs/logs/eval_stream_%j.log
# usage: sbatch --export=ALL,ARGS="--split val" eval_stream.sbatch
module load gcc python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 eval_stream.py ${ARGS}
