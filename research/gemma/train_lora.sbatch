#!/bin/bash
#SBATCH --account=def-ajbonner_gpu
#SBATCH --job-name=gemma_lora
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/wuhungma/runs/logs/lora_%j.log
# Pick the GPUs at submit time (give 4 CPUs per GPU); one training process runs per GPU, e.g.
#   1 full H100:   sbatch --gres=gpu:h100:1 --export=ALL,ARGS="--out /scratch/wuhungma/lora/r16" train_lora.sbatch
#   4 full H100s:  sbatch --gres=gpu:h100:4 --cpus-per-task=16 --mem=200G --export=ALL,ARGS="--out ..." train_lora.sbatch
#   40 GB slice:   sbatch --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --export=ALL,ARGS="--out ... --qlora" train_lora.sbatch
module load gcc python/3.11 cuda/13.2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /project/6010122/group12
python3 -m torch.distributed.run --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE:-1} train_lora.py ${ARGS}
