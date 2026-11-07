# Research code

One folder per part of the project, in the order the work was done. Each `.sbatch` file is the Slurm job that runs
the script next to it.

| Folder | What it does | Start with |
|---|---|---|
| `data/` | Make the team splits and the 32-frame clips from EDF, MCFD and UP-Fall | `split_dataset.py`, `split_upfall.py` |
| `gemma/` | Gemma 3 12B: zero-shot scoring, LoRA training and evaluation | `train_lora.py`, `run_probs.py` |
| `videomae/` | Fine-tune VideoMAE-base (fall vs no fall) | `train_videomae.py` |
| `pose/` | Pose keypoints and the small temporal pose model | `extract_pose.py`, `train_pose.py` |
| `stream/` | Replay recordings as a live stream; alarm metrics; multi-camera fusion | `eval_stream.py`, `fusion_metrics.py` |
| `verify/` | VideoMAE raises the alarm, Gemma checks it (one and several cameras) | `verify_alarms.py`, `verify_mv.py` |
| `splits/` | The team's train / val / test lists | `splits/README.md` |
| `results/` | Write-ups with all numbers | `results/results.md` |

Scripts read `splits/` (and the generated frames) from this folder, and find modules in the other folders by
themselves. Job files and default paths are from the Alliance Fir cluster, where the scripts were run from one flat
folder (`/project/6010122/group12`); change those paths to run elsewhere.
