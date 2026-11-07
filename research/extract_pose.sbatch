#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=pose
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/wuhungma/runs/logs/pose_%A_%a.log
# usage: sbatch extract_pose.sbatch      (4 shards over train, val and test)
module load gcc python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 extract_pose.py --shard ${SLURM_ARRAY_TASK_ID}/4
