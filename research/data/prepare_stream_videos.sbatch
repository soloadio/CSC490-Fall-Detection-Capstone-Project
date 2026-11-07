#!/bin/bash
#SBATCH --account=def-ajbonner_cpu
#SBATCH --job-name=stream_videos
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/wuhungma/runs/logs/stream_videos_%j.log
# Rebuild OmniFall's EDF and MCFD MP4s (correct timing: EDF .bin frames -> 30 fps, MCFD 120 fps header
# reinterpreted as 30) from the downloads already in the cache, for eval_stream.py. UP-Fall needs no
# videos: eval_stream.py reads its timestamped PNGs straight from the camera zips.
# usage: sbatch prepare_stream_videos.sbatch
module load gcc arrow/24.0.0 python/3.11
export OMNIFALL_DOWNLOAD_DIR=/scratch/wuhungma/omnifall_cache/downloads
V=/scratch/wuhungma/stream_eval/videos
omnifall prepare edf -y --archive $OMNIFALL_DOWNLOAD_DIR/edf/unpacked --video-dir $V/edf --workers 8
omnifall prepare mcfd -y --archive $OMNIFALL_DOWNLOAD_DIR/mcfd --video-dir $V/mcfd --workers 8
omnifall verify edf --video-dir $V/edf
omnifall verify mcfd --video-dir $V/mcfd
