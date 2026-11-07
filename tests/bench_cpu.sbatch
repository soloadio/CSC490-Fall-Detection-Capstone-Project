#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=bench_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/wuhungma/deploy/bench_cpu_%j.log
module load gcc python/3.11
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python3 /scratch/wuhungma/deploy/bench_cpu.py
