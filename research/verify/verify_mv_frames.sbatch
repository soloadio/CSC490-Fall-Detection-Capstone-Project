#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=mv_frames
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wuhungma/verify_gemma/%x_%j.log
# CPU stage: cut the 32-frame clips around VideoMAE's alarms.   usage: sbatch verify_alarms.sbatch
module load gcc arrow/24.0.0 python/3.11
cd /project/6010122/group12
python3 verify_alarms.py frames --clips clips_mv.jsonl
