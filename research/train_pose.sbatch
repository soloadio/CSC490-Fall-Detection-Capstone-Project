#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=pose_clf
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/wuhungma/runs/logs/pose_clf_%j.log
# Trains the pose classifier on all three datasets, then once per held-out dataset.
module load gcc python/3.11 cuda/13.2
cd /project/6010122/group12
O=/scratch/wuhungma/pose_clf
python3 train_pose.py --out $O/all
python3 train_pose.py --out $O/holdout_edf --datasets mcfd up_fall
python3 train_pose.py --out $O/holdout_mcfd --datasets edf up_fall
python3 train_pose.py --out $O/holdout_up_fall --datasets edf mcfd
