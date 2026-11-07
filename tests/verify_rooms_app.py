"""Run the real app handlers: two phone sessions in one room (MCFD test chute01 cam1 + cam2, fall at ~35.5 s), frames
fed in real time at 5 frames/s from 27 s to 45 s; then one camera alone; then the dashboard. Prints what happened."""
import os, subprocess, sys, threading, time
import numpy as np
os.environ.setdefault("MODEL_DIR", "/scratch/wuhungma/videomae/fallonly_lr5e-5/best")
sys.path.insert(0, "/scratch/wuhungma/deploy/space")
import app

V = "/scratch/wuhungma/stream_eval/videos/mcfd/chute01/{}.mp4"


def frames(cam, start, end, fps=5):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(start), "-to", str(end), "-i", V.format(cam), "-vf",
                          f"fps={fps},scale=320:-2", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    w = 320
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", V.format(cam)], capture_output=True, text=True).stdout.strip().split(",")
    h = round(int(probe[1]) * 320 / int(probe[0]) / 2) * 2
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def phone(cam, room, name, fr, result):
    s = None
    for f in fr:
        t = time.monotonic()
        s, *outs = app.on_frame(f, room, name, s)
        time.sleep(max(0.0, 0.2 - (time.monotonic() - t)))
    result[name] = s


def show(title, s):
    print(f"--- {title}: room {s['room']!r}")
    print("  scores (newest first):", " ".join(h.split("P(fall) ")[1].split()[0] for h in s["hist"][:15]))
    print("  alarm log:", app.alarm_log(s).replace("\n", " / "))


def main():
    fr1, fr2 = frames("cam1", 27, 45), frames("cam2", 27, 45)
    print("frames", fr1.shape, fr2.shape)
    res = {}
    th = [threading.Thread(target=phone, args=(c, "Lab ", n, f, res)) for c, n, f in (("cam1", "door", fr1), ("cam2", "corner", fr2))]
    [t.start() for t in th]; [t.join() for t in th]
    for n in ("door", "corner"):
        show(f"two phones, {n}", res[n])
    html, log = app.dashboard("lab")
    print("--- dashboard right after:", log.replace("\n", " / "))
    print("  ", html[html.find("font-weight:700'>") + 17:][:160])
    time.sleep(4)
    html, _ = app.dashboard("lab")
    print("--- dashboard 4 s after the phones stopped:", html[html.find("font-size:15px'>") + 16:][:120])
    res = {}
    phone("cam1", "solo-room", "only", fr1, res)
    show("one phone in its own room", res["only"])
    res = {}
    phone("cam1", "", "nameless", fr1, res)
    show("no room code (single camera, as before)", res["nameless"])



if __name__ == "__main__":
    main()
