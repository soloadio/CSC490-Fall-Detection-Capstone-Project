---
title: Fall Detector
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# Live fall detector (CSC490 project)

Point a webcam at a room; the app watches the last 3 seconds of video every half second and raises an alarm when it
sees a person fall.

- **Model:** VideoMAE-base (Kinetics-400 pretrained), fine-tuned to tell the *falling motion* from everything else,
  on OmniFall's EDF, MCFD and UP-Fall datasets. Lying down on purpose is treated as "no fall".
- **Measured offline** on held-out recordings replayed as a live stream (a 3 s window every 0.5 s, threshold 0.30):
  93% of falls caught, a median 1.2 s after the fall starts, about 4.5 false alarms per hour of scripted non-fall
  activity. Real homes, cameras and lighting will differ.
- **Runs on CPU**, int8-quantised (about 0.3 to 0.6 s per window on 2 to 8 server cores).
- **Privacy:** frames are kept in memory for a few seconds to fill the window and are never written to disk.

**Several cameras, one room:** open the page on each phone and type the same room code (or use a link like
`https://ggfggf1234560-fall-detector.hf.space/?room=lab&cam=door`). With two or more cameras the room alarms only when
two cameras see a fall within 1 s; with one camera it uses the single-camera rule. The *Room dashboard* tab shows the
room from a laptop. On the paired test recordings, "two agree" caught 94% of falls (single camera 92%) with 2 false
alarms instead of 24. Each extra camera adds one model run per check, so the server is busier with more phones.

**Alerts:** the *Room dashboard* tab beeps when the room alarms, and can push each alarm to a phone through
[ntfy](https://ntfy.sh) (free, no account): install the ntfy app, subscribe to the topic shown on the dashboard and
press *Turn on alerts*. Alerts carry text only (room, time, which cameras), never images; anyone who knows the topic
can read them. Topics set on the dashboard are kept in memory; `ALERT_TOPIC` sets one topic for every room that survives
restarts (the hosted Space uses `fall-alert`).

Tips: show the whole body, place the camera high and a few metres away (like the training cameras).

Settings can be changed with environment variables: `WINDOW` (3.0), `THRESHOLD` (0.30), `K` (1), `COOLDOWN` (10),
`AGREE` (1.0, seconds two cameras have to agree within), `INFER_EVERY` (0.5), `STREAM_EVERY` (0.2, i.e. 5 frames/s sent by the browser), `INT8` (1), `THREADS`, `NTFY_SERVER` (https://ntfy.sh),
`ALERT_TOPIC`, `VERIFIERS`, `VERIFY_CLIP` (4.0).

**Not a medical device.** Research prototype for a course project.

Licences: the VideoMAE pretrained weights and the OmniFall annotations are released for non-commercial use
(check the VideoMAE model card and the OmniFall dataset card); the source datasets belong to their authors.

## Second opinion: vision-language checkers on Modal

When VideoMAE raises an alarm the app waits 1 s, cuts a 4 s clip (32 frames) from every camera in the room and asks
each checker in `VERIFIERS` (comma separated) whether it shows a fall; all (camera, checker) pairs run at the same time.
The alarm goes out if at least half of the votes say fall (a tie keeps the alarm; a failed check never cancels one),
and each checker's reason is shown in the room log and the push alert.

| `VERIFIERS` value | Checker | Where |
|---|---|---|
| `gemma-modal` | Gemma 3 12B-it + LoRA (trained on verdict + reason) | Modal GPU, class `Verifier` in `modal_app.py` |
| `qwen-modal` | Qwen2-VL-7B-Instruct + LoRA (Solomon) | Modal GPU, class `QwenVerifier` (`qwen_scoring.py`) |
| `gemma` | Gemma on the machine running the app (needs a large GPU) | `verifier.py` |

The hosted Space uses `VERIFIERS=gemma-modal,qwen-modal` and the secrets `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`.
Setup of the Modal side is at the top of `modal_app.py`. Each Modal GPU stays on 5 minutes after its last check
(cold start about 45 s); the dashboard's *Wake up checker* button loads both before a test.

On the offline VideoMAE alarm clips (one camera, `research/verify/`), Gemma + Qwen keep 113/115 (val) and 133/136 (test)
real falls and remove 2/4 and 5/7 false alarms; Gemma alone keeps 103/115 and 131/136.

## Set up from scratch

Trained weights are not in git; they are in the Hugging Face repo
[CSC490/fall-models](https://huggingface.co/CSC490/fall-models) and are downloaded by the steps below.

**1. Accounts (once)**
- Hugging Face: an account and a [token](https://huggingface.co/settings/tokens) (read is enough); open
  [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) and accept the Gemma terms with that account.
- [Modal](https://modal.com) (free Starter plan: $30 of GPU credit a month). Set a spending limit in its billing settings.
- Optional, for push alerts: the free [ntfy](https://ntfy.sh) app on your phone.

**2. The checkers on Modal (once, from any machine with Python 3.11)**
```bash
pip install modal huggingface_hub
modal token new                                    # opens a browser to log in
modal secret create huggingface-secret HF_TOKEN=<your Hugging Face token>
cd app
modal run modal_app.py::download                   # Gemma 3 12B-it into a Modal volume (~24 GB, a few minutes)
modal run modal_app.py::download_qwen              # Qwen2-VL-7B-Instruct (~16 GB)
modal run modal_app.py::download_adapters          # our two LoRA adapters from CSC490/fall-models
modal deploy modal_app.py                          # the "fall-verifier" app: Verifier (Gemma) and QwenVerifier (Qwen)
```

**3a. Run the website on your own machine (CPU is enough)**
```bash
cd app
pip install "gradio==6.30.0" -r requirements.txt
export VERIFIERS=gemma-modal,qwen-modal            # leave unset to run VideoMAE alone, without checkers
export ALERT_TOPIC=<an ntfy topic>                 # optional
python app.py                                      # first start downloads VideoMAE from CSC490/fall-models into app/model/
```
Open the printed address (http://127.0.0.1:7860). Phones need an HTTPS address to use their camera, so for phones use
a Space (3b) or a tunnel.

**3b. Or host it as a Hugging Face Space**
Create a Gradio Space (free CPU hardware is enough), upload the files of `app/` (this README's header configures the
Space), then under *Settings* add the secrets `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` (from `~/.modal.toml`, or a new
token at modal.com/settings/tokens) and the variables `VERIFIERS=gemma-modal,qwen-modal` and, optionally, `ALERT_TOPIC`.
The Space downloads VideoMAE on its first start.

**4. Use it**: open the page on each phone with the same room code (`?room=lab&cam=door`), open the *Room dashboard*
tab on a laptop, press *Wake up checker* before testing (each Modal GPU stays on 5 minutes after its last check).
