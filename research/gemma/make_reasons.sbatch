#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=reasons
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/wuhungma/reasons/%x_%j.log
# Evidence sentences for the LoRA training clips.   usage: sbatch [--export=ALL,ARGS="--pilot 24"] make_reasons.sbatch
set -e
module load gcc arrow/24.0.0 python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 make_reasons.py ${ARGS}
