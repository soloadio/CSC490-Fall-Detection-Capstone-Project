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
can read them. Topics are kept in memory, so turn alerts on again after the Space restarts.

Tips: show the whole body, place the camera high and a few metres away (like the training cameras).

Settings can be changed with environment variables: `WINDOW` (3.0), `THRESHOLD` (0.30), `K` (1), `COOLDOWN` (10),
`AGREE` (1.0, seconds two cameras have to agree within), `INFER_EVERY` (0.5), `STREAM_EVERY` (0.2, i.e. 5 frames/s sent by the browser), `INT8` (1), `THREADS`, `NTFY_SERVER` (https://ntfy.sh).

**Not a medical device.** Research prototype for a course project.

Licences: the VideoMAE pretrained weights and the OmniFall annotations are released for non-commercial use
(check the VideoMAE model card and the OmniFall dataset card); the source datasets belong to their authors.

## Second opinion (off by default)

Set `VERIFIERS=gemma` to have a vision-language model check each VideoMAE alarm before it goes out (`verifier.py`).
Also set `GEMMA_BASE` (hub id or folder), `GEMMA_ADAPTER` (LoRA adapter, hub id or folder) and the secret `HF_TOKEN`
(Gemma is a gated model). Tested on the cluster only, not yet on a hosted Space.
