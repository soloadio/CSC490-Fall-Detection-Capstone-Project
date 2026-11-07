"""Live fall detection demo (Gradio). The browser streams webcam frames (STREAM_EVERY s apart); the server keeps the
last few seconds and, at most once per INFER_EVERY seconds, runs VideoMAE on the latest window in a background
thread and applies the alarm rule. One request stream per phone (no separate timer), to stay under rate limits.

Several phones can watch one room: the same room code on each phone puts them in one Room (detector.py), which
raises a single alarm when two cameras agree (one camera alone: the single-camera rule). The "Room dashboard" tab
shows a room without a camera, beeps on an alarm, and can send push alerts for the room through ntfy (a free
notification service: install the ntfy app and subscribe to the room's topic). Alerts carry text only, no images.
Frames are only held in memory for a few seconds and are never written to disk."""
try:  # Hugging Face ZeroGPU Spaces: `spaces` must be imported before torch, and the Space expects at least
    import spaces  # one @spaces.GPU function. The model runs on CPU (it is called every second, which would
    # use up the daily GPU quota), so this function exists only to satisfy that check and is never called.

    @spaces.GPU(duration=5)
    def _zerogpu_placeholder():
        return None
except ImportError:  # running anywhere else
    spaces = None

import json
import os
import re
import secrets
import threading
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from detector import AlarmRule, FallModel, FrameBuffer, Room, settings_from_env
from verifier import RawBuffer, decide, load_verifiers

CFG = settings_from_env()
MODEL = FallModel(CFG["model_dir"], int8=CFG["int8"], threads=CFG["threads"])
SHOW_ALARM_FOR = 8.0  # seconds the red banner stays up after an alarm
STREAM_EVERY = float(os.environ.get("STREAM_EVERY", 0.2))  # 5 frames/s; the model uses 16 frames over 3 s
EXEC = ThreadPoolExecutor(max_workers=1)  # the model runs here, so the frame handler never blocks
PENDING = {}  # session id -> Future (kept outside gr.State so the state stays plain data)
ROOMS, ROOMS_LOCK = {}, threading.Lock()  # room code -> Room, shared by every visitor of this server
NTFY_SERVER = os.environ.get("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
ALERT_TOPICS = {}  # room code -> ntfy topic (in memory: set it again after the Space restarts)
ALERTS = ThreadPoolExecutor(max_workers=2)  # network calls never block the frame handler

# Second opinion (verifier.py): env VERIFIERS=gemma turns it on. A VideoMAE alarm then waits VERIFY_WAIT s, every camera's
# last 3 s go to the vision-language model, and the alarm (banner, beep, push alert) only goes out if the votes confirm.
VERIFIERS = load_verifiers()
VERIFY_WAIT = float(os.environ.get("VERIFY_WAIT", 1.0))
RAW = {}  # session id -> RawBuffer of full-resolution frames (only when verifiers are on)
CHECKS = ThreadPoolExecutor(max_workers=1)  # one check at a time; the model is big
if VERIFIERS:
    threading.Thread(target=lambda: [v.load() for v in VERIFIERS], daemon=True).start()


def room_code(text):
    return (text or "").strip().lower()[:40]


def get_room(code):
    with ROOMS_LOCK:
        now = time.monotonic()
        for c in [c for c, r in ROOMS.items() if c != code and not r.live(now) and now - r.t > 600]:
            del ROOMS[c]  # forget rooms nobody has used for 10 minutes
        if code not in ROOMS:
            ROOMS[code] = Room(CFG["threshold"], CFG["k"], CFG["cooldown"], agree=CFG["agree"])
        return ROOMS[code]


def ntfy_topic(text):
    """ntfy topics: letters, digits, - and _ (up to 64). Anyone who knows the topic can read the alerts."""
    t = (text or "").strip()
    return t if re.fullmatch(r"[A-Za-z0-9_-]{6,64}", t) else ""


def send_alert(topic, title, message):
    body = json.dumps({"topic": topic, "title": title, "message": message, "priority": 5,
                       "tags": ["rotating_light"]}).encode()
    req = urllib.request.Request(NTFY_SERVER, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status


def alert_room(code, message, title=None):
    """Send a push alert for room `code` in the background; the outcome goes to the room's alarm log."""
    topic = ALERT_TOPICS.get(code)
    if not topic:
        return

    def done(fut):
        err = fut.exception()
        note = f"push alert failed ({type(err).__name__})" if err else "push alert sent"
        room = get_room(code)
        room.log.insert(0, f"{time.strftime('%H:%M:%S')}  {note}")
        del room.log[10:]

    ALERTS.submit(send_alert, topic, title or f"Fall detected in room {code}", message).add_done_callback(done)


def set_alerts(room_text, topic_text):
    """Dashboard button: remember the room's ntfy topic and send a test alert."""
    code, topic = room_code(room_text), ntfy_topic(topic_text)
    if not code:
        return "Type the room code first."
    if not topic:
        ALERT_TOPICS.pop(code, None)
        return "Alerts off for this room (a topic needs 6 to 64 letters, digits, - or _)."
    ALERT_TOPICS[code] = topic
    alert_room(code, f"Test alert: room {code} will send fall alerts to this topic.", title="Fall detector test alert")
    return (f"Alerts on: room {code} sends to ntfy topic {topic}. A test alert was sent; it should arrive within a "
            "few seconds (see the room alarm log for 'push alert sent').")


def run_check(cams):
    """cams {session id: camera name} -> list of reports, one per (checker, camera). Waits VERIFY_WAIT s first, so the
    clip also shows the moments after the alarm (what the offline experiment found works best)."""
    time.sleep(VERIFY_WAIT)
    t_end, reports = time.monotonic(), []
    for sid, name in cams.items():
        raw = RAW.get(sid)
        frames = raw.clip(t_end) if raw else None
        if frames is None:
            reports.append({"camera": name, "verifier": "-", "reason": "no recent video from this camera"})
            continue
        for v in VERIFIERS:
            try:
                if not v.ready:
                    raise RuntimeError("model still loading")
                reports.append({"camera": name, **v.check(frames)})
            except Exception as e:  # noqa: BLE001 - a failed check must not hide an alarm
                reports.append({"camera": name, "verifier": v.name, "reason": f"check failed ({type(e).__name__})"})
    return reports


def report_lines(reports):
    return [f"    {r['camera']} / {r['verifier']}: " + (f"{r['verdict']} (P {r['p_fall']:.2f}) - " if "verdict" in r else "")
            + r["reason"] for r in reports]


def start_check(cams, on_done):
    """Run the second opinion in the background; on_done(confirmed, reports) when it has voted."""
    def finish(fut):
        try:
            reports = fut.result()
        except Exception as e:  # noqa: BLE001
            reports = [{"camera": "-", "verifier": "-", "reason": f"check failed ({type(e).__name__})"}]
        on_done(decide(reports), reports)
    CHECKS.submit(run_check, dict(cams)).add_done_callback(finish)


def room_alarm(code, room, seen):
    """VideoMAE alarm in room `code`: alert now, or (verifiers on) check first."""
    clock = time.strftime("%H:%M:%S")
    if not VERIFIERS:
        room.log.insert(0, f"{clock}  fall detected by {seen}")
        del room.log[10:]
        alert_room(code, f"{clock}: fall seen by {seen}. Check on the person now.")
        return
    room.checking = True
    room.log.insert(0, f"{clock}  VideoMAE alarm ({seen}); asking the vision-language model...")
    del room.log[10:]
    cams = {sid: c["name"] for sid, c in room.cams.items() if sid in RAW}

    def done(confirmed, reports):
        room.checking, room.reports = False, reports
        now, clock2 = time.monotonic(), time.strftime("%H:%M:%S")
        verdict = "CONFIRMED" if confirmed else "cancelled (the model does not see a fall)"
        room.log[0:0] = report_lines(reports)[::-1] + [f"{clock2}  alarm {verdict}"]
        del room.log[16:]
        if confirmed:
            room.confirmed = now
            alert_room(code, f"{clock2}: fall seen by {seen} and confirmed. Check on the person now.")
    start_check(cams, done)


def warm_checker():
    """Dashboard button: send a test clip (32 grey frames) through every checker, so the GPU is loaded before a real
    alarm (a cold start takes about 45 s on Modal). Returns what each checker answered and how long it took."""
    if not VERIFIERS:
        return "Second opinion is off (no VERIFIERS set)."
    from PIL import Image
    frames, out = [Image.new("RGB", (320, 240), (128, 128, 128))] * 32, []
    for v in VERIFIERS:
        try:
            if not v.ready:
                raise RuntimeError("still loading")
            r = v.check(frames)
            out.append(f"{v.name}: ready, answered in {r['seconds']:.0f} s (test clip: {r['verdict']}, \"{r['reason']}\")")
        except Exception as e:  # noqa: BLE001
            out.append(f"{v.name}: NOT working ({type(e).__name__}: {str(e)[:150]})")
    return "\n".join(out)


def score(frames, t_end):
    t = time.perf_counter()
    p = MODEL.p_fall(frames)
    return p, time.perf_counter() - t, t_end


def new_session():
    return {"buf": FrameBuffer(CFG["window"]), "rule": AlarmRule(CFG["threshold"], CFG["k"], CFG["cooldown"]),
            "t0": time.monotonic(), "last_alarm": None, "log": [], "p": None, "sec": None, "hist": [], "fps": None, "gap": None, "last_inf": -1e9,
            "id": uuid.uuid4().hex, "room": "", "name": ""}


def box(color, text, detail):
    return (f"<div style='background:{color};color:white;padding:18px;border-radius:10px;text-align:center'>"
            f"<div style='font-size:30px;font-weight:700'>{text}</div><div style='font-size:15px'>{detail}</div></div>")


def alarm_text(last_alarm, cams, now):
    seen = f" by {', '.join(cams)}" if cams else ""
    return f"FALL DETECTED <span style='font-size:18px'>({now - last_alarm:.0f} s ago{seen})</span>"


def room_lines(code, now):
    """(alarm time, cameras that saw it, html lines about the room) for room `code`."""
    room = get_room(code)
    cams, room_p, last_alarm, alarm_cams = room.snapshot(now)
    if VERIFIERS:  # the red alarm is the confirmed one; VideoMAE's own alarm only starts the check
        last_alarm = room.confirmed
    if len(cams) >= 2:
        rule = f"{len(cams)} cameras, alarm when two agree"
    elif cams:
        rule = "1 camera, single-camera rule (open the page on another phone with the same room code to add one)"
    else:
        rule = "no camera connected"
    per_cam = " &nbsp;|&nbsp; ".join(f"{n}: P {p:.2f}" for n, p, _ in cams)
    lines = [f"room <b>{code}</b>: {rule}"] + ([per_cam] if per_cam else [])
    return last_alarm, alarm_cams, lines, len(cams)


def banner(s):
    now = time.monotonic()
    if s["room"]:
        last_alarm, cams, lines, _ = room_lines(s["room"], now)
    else:
        last_alarm, cams, lines = s["last_alarm"], [], []
    checking = get_room(s["room"]).checking if s["room"] else s.get("checking")
    if last_alarm is not None and now - last_alarm < SHOW_ALARM_FOR:
        color, text = "#c62828", alarm_text(last_alarm, cams, now)
    elif checking:
        color, text = "#ef6c00", "Possible fall: checking..."
    elif s["buf"].last_time() is None or now - s["buf"].last_time() > 2.0:
        color, text = "#757575", "Waiting for camera... (click the webcam and allow access)"
    else:
        color, text = "#2e7d32", "No fall"
    if s["p"] is not None:
        lines.append(f"this camera: P(fall) = {s['p']:.2f} &nbsp;|&nbsp; model time {s['sec']:.2f} s")
    if s["fps"] is not None:
        gap = "-" if s["gap"] is None else f"{s['gap']:.2f} s"
        lines.append(f"camera: {s['fps']:.1f} frames/s received, largest gap {gap}")
    return box(color, text, "<br>".join(lines))


def alarm_log(s):
    log = get_room(s["room"]).log if s["room"] else s["log"]
    return "\n".join(log) or "No alarms yet."


def diagnostics(s):
    return "\n".join(s["hist"]) or "No scores yet."


def outputs(s):
    return s, banner(s), alarm_log(s), diagnostics(s)


def on_frame(frame, room_text, name_text, s):
    """Every streamed webcam frame: buffer it, collect a finished score if there is one, and start the next
    score (in the background) at most once per INFER_EVERY s. Returns immediately, so frames keep flowing."""
    if s is None:
        s = new_session()
    if frame is None:
        return s, gr.skip(), gr.skip(), gr.skip()
    code, s["name"] = room_code(room_text), (name_text or "").strip()[:30] or f"camera {s['id'][:4]}"
    if code != s["room"]:
        if s["room"]:
            get_room(s["room"]).leave(s["id"])
        s["room"] = code
    now = time.monotonic()
    s["buf"].push(now, frame)
    if VERIFIERS:
        RAW.setdefault(s["id"], RawBuffer()).push(now, frame)
        for k in [k for k, b in RAW.items() if now - (b.last_time() or 0) > 60]:
            del RAW[k]
    updated = False
    fut = PENDING.get(s["id"])
    if fut is not None and fut.done():
        del PENDING[s["id"]]
        p, s["sec"], t_end = fut.result()
        s["p"] = p
        s["fps"], s["gap"] = s["buf"].stats(t_end, CFG["window"])
        clock = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 10)}"
        mark = "  <-- above threshold" if p >= CFG["threshold"] else ""
        s["hist"].insert(0, f"{clock}  P(fall) {p:.2f}  | {s['fps']:.1f} fps{mark}")
        s["hist"] = s["hist"][:15]
        if s["room"]:
            room = get_room(s["room"])
            if room.post(s["id"], s["name"], t_end, p):
                room_alarm(s["room"], room, ", ".join(room.alarm_cams))
        elif s["rule"].update(t_end, p):
            clock = time.strftime("%H:%M:%S")
            if not VERIFIERS:
                s["last_alarm"] = t_end
                s["log"].insert(0, f"{clock}  fall detected (P = {p:.2f})")
                s["log"] = s["log"][:10]
            else:
                s["checking"] = True
                s["log"].insert(0, f"{clock}  VideoMAE alarm (P = {p:.2f}); asking the vision-language model...")

                def done(confirmed, reports, s=s, clock=clock):
                    s["checking"] = False
                    s["log"][0:0] = report_lines(reports)[::-1] + [
                        f"{time.strftime('%H:%M:%S')}  alarm " + ("CONFIRMED" if confirmed else "cancelled")]
                    del s["log"][16:]
                    if confirmed:
                        s["last_alarm"] = time.monotonic()
                start_check({s["id"]: s["name"]}, done)
        updated = True
    if s["id"] not in PENDING and now - s["last_inf"] >= CFG["infer_every"] and len(s["buf"].items) >= 5:
        PENDING[s["id"]] = EXEC.submit(score, s["buf"].window_frames(now), now)
        s["last_inf"] = now
    last_alarm = (get_room(s["room"]).confirmed if VERIFIERS else get_room(s["room"]).last_alarm) if s["room"] \
        else s["last_alarm"]
    if updated or s.get("checking") or (last_alarm is not None and now - last_alarm < SHOW_ALARM_FOR + 1):
        return outputs(s)  # also refreshes the "x s ago" counter while the alarm banner is up
    return s, gr.skip(), gr.skip(), gr.skip()


def dashboard(room_text):
    code, now = room_code(room_text), time.monotonic()
    if not code:
        return box("#757575", "Type a room code", "the same code the phones use"), "No alarms yet."
    last_alarm, cams, lines, n = room_lines(code, now)
    if last_alarm is not None and now - last_alarm < SHOW_ALARM_FOR:
        color, text = "#c62828", alarm_text(last_alarm, cams, now)
    elif get_room(code).checking:
        color, text = "#ef6c00", "Possible fall: checking..."
    elif n == 0:
        color, text = "#757575", "Waiting for cameras..."
    else:
        color, text = "#2e7d32", "No fall"
    return box(color, text, "<br>".join(lines)), "\n".join(get_room(code).log) or "No alarms yet."


def from_url(request: gr.Request):
    """Pre-fill room code and camera name from the link, e.g. ...hf.space/?room=lab&cam=door; suggest a hard-to-guess
    alert topic (or the room's current one)."""
    q = request.query_params if request else {}
    room = q.get("room", "")
    topic = q.get("alert", "") or ALERT_TOPICS.get(room_code(room), "") or f"fall-alert-{secrets.token_hex(4)}"
    return room, q.get("cam", ""), room, topic


# Dashboard beep: runs in the browser on every banner update and sounds when an alarm appears (no server request).
BEEP_JS = """(html) => {
  const on = (html || "").includes("FALL DETECTED");
  if (on && !window.__fallOn) {
    try {
      const ctx = window.__fallCtx = window.__fallCtx || new AudioContext();
      ctx.resume();
      [0, 0.5, 1.0].forEach((d) => {
        const o = ctx.createOscillator(), g = ctx.createGain();
        o.frequency.value = 880; g.gain.value = 0.3;
        o.connect(g); g.connect(ctx.destination);
        o.start(ctx.currentTime + d); o.stop(ctx.currentTime + d + 0.3);
      });
    } catch (e) {}
    if (navigator.vibrate) navigator.vibrate([300, 200, 300]);
  }
  window.__fallOn = on;
}"""
UNLOCK_JS = "() => { try { (window.__fallCtx = window.__fallCtx || new AudioContext()).resume(); } catch (e) {} }"


with gr.Blocks(title="Fall detector") as demo:
    gr.Markdown(
        "# Fall detector (CSC490 demo)\n"
        f"VideoMAE-base fine-tuned on OmniFall (EDF, MCFD, UP-Fall). Every {CFG['infer_every']:g} s it looks at the last "
        f"{CFG['window']:g} s of video and raises an alarm when P(fall) >= {CFG['threshold']:g}. "
        "Best results with the whole body in view and the camera placed high, as in the training videos. "
        "Video is processed in memory only and is not stored.\n\n"
        "**Several cameras, one room:** open this page on each phone and type the same room code; the room alarms "
        "when two cameras agree. Watch the room from a laptop in the *Room dashboard* tab."
    )
    state = gr.State(None)
    with gr.Tab("Camera"):
        with gr.Row():
            room_in = gr.Textbox(label="Room code (optional)", placeholder="e.g. lab: same code on every phone in the room")
            name_in = gr.Textbox(label="Camera name", placeholder="e.g. door")
        with gr.Row():
            cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Camera")
            with gr.Column():
                status = gr.HTML(banner(new_session()))
                log = gr.Textbox(label="Alarms", lines=4, value="No alarms yet.", interactive=False)
                diag = gr.Textbox(label="Diagnostics: this camera's last 15 scores (newest first)", lines=15,
                                  value="No scores yet.", interactive=False)
    with gr.Tab("Room dashboard"):
        with gr.Row():
            dash_room = gr.Textbox(label="Room code", placeholder="the code the phones use")
            watch = gr.Button("Watch room", variant="primary")
        dash_status = gr.HTML(box("#757575", "Type a room code and press Watch", ""))
        dash_log = gr.Textbox(label="Room alarms", lines=6, value="No alarms yet.", interactive=False)
        gr.Markdown("The dashboard beeps when the room alarms (keep this tab open, sound on).\n\n"
                    "**Push alerts to a phone:** install the free **ntfy** app (Android / iPhone) or open "
                    "ntfy.sh in a browser, subscribe to the topic below, then press *Turn on alerts*. Every room "
                    "alarm is then sent there as a notification (text only, no images). Anyone who knows the topic "
                    "can read the alerts, so keep it hard to guess.")
        with gr.Row():
            topic_in = gr.Textbox(label="ntfy topic for this room")
            alerts_btn = gr.Button("Turn on alerts (sends a test)")
        alerts_msg = gr.Markdown()
        with gr.Row():
            warm_btn = gr.Button("Wake up checker (before a demo)")
            warm_msg = gr.Textbox(label="Checker status", lines=2, interactive=False)
        timer = gr.Timer(1.0, active=False)  # only the dashboard polls, and only after Watch is pressed
    cam.stream(on_frame, inputs=[cam, room_in, name_in, state], outputs=[state, status, log, diag],
               stream_every=STREAM_EVERY, time_limit=None, concurrency_limit=16, show_progress="hidden")
    watch.click(lambda: gr.Timer(active=True), outputs=timer)
    watch.click(None, js=UNLOCK_JS)  # browser only: a click is what allows the page to play sound later
    dash_status.change(None, inputs=dash_status, js=BEEP_JS)
    warm_btn.click(warm_checker, outputs=warm_msg, api_name="warm_checker")
    alerts_btn.click(set_alerts, inputs=[dash_room, topic_in], outputs=alerts_msg, api_name="set_alerts")
    timer.tick(dashboard, inputs=dash_room, outputs=[dash_status, dash_log], concurrency_limit=8,
               show_progress="hidden")
    demo.load(from_url, outputs=[room_in, name_in, dash_room, topic_in])
    gr.Markdown(f"<small>Settings: window {CFG['window']} s, threshold {CFG['threshold']}, k {CFG['k']}, "
                f"cooldown {CFG['cooldown']} s, two-camera agreement within {CFG['agree']} s, int8 {CFG['int8']}, "
                f"second opinion: {', '.join(v.name for v in VERIFIERS) or 'off'}. "
                "Research prototype, not a medical device.</small>")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0")
