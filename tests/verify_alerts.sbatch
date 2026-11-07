#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=verify_alerts
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/wuhungma/deploy/verify_alerts_%j.log
module load gcc python/3.11 ffmpeg 2>/dev/null || module load gcc python/3.11
source /scratch/wuhungma/deploy/venv-app/bin/activate
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GRADIO_ANALYTICS_ENABLED=False THREADS=2
cd /scratch/wuhungma/deploy/space
echo "### app launch test"
MODEL_DIR=/scratch/wuhungma/videomae/fallonly_lr5e-5/best python app.py > /scratch/wuhungma/deploy/app_launch_$SLURM_JOB_ID.log 2>&1 &
APP=$!
for i in $(seq 1 60); do code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:7860/?room=lab&cam=door" || true); [ "$code" = "200" ] && break; sleep 2; done
echo "HTTP status of the app page: $code (after ~$((i*2)) s)"; kill $APP; grep -i -E 'error|traceback' /scratch/wuhungma/deploy/app_launch_$SLURM_JOB_ID.log | head
echo "### two phones in one room"
python /scratch/wuhungma/deploy/verify_alerts.py "${ALERT_TOPIC:?set ALERT_TOPIC to your ntfy topic}" 2>&1 | grep -v -i -E 'warn|quantize_per_tensor|Loading weights'
