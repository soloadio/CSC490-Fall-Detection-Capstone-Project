#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=verify_app
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wuhungma/deploy/verify_%j.log
module load gcc python/3.11
source /scratch/wuhungma/deploy/venv-app/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False
echo "### app launch test"
cd /scratch/wuhungma/deploy/space
MODEL_DIR=/scratch/wuhungma/videomae/fallonly_lr5e-5/best python app.py > /scratch/wuhungma/deploy/app_launch_$SLURM_JOB_ID.log 2>&1 &
APP=$!
for i in $(seq 1 60); do code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:7860/ || true); [ "$code" = "200" ] && break; sleep 2; done
echo "HTTP status of the app page: $code (after ~$((i*2)) s)"; kill $APP; tail -n 5 /scratch/wuhungma/deploy/app_launch_$SLURM_JOB_ID.log
echo "### detector verification"
python /scratch/wuhungma/deploy/verify_detector.py
