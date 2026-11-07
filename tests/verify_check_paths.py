"""App alarm logic with a stand-in checker (no Gemma, CPU): confirm, cancel, checker error, tie. Also reports when the alarm
fired relative to the fall (video time, fall at ~35.5 s) and which cameras were live then."""
import threading, time
from verify_rooms_app import app, frames, phone  # noqa: E402


class Stub:
    name, ready = "stub", True

    def __init__(self, answers):
        self.answers, self.n = list(answers), 0

    def check(self, frames_):
        a = self.answers[self.n % len(self.answers)]
        self.n += 1
        time.sleep(3)
        if a == "error":
            raise RuntimeError("boom")
        return {"verifier": self.name, "verdict": a, "p_fall": 0.9 if a == "fall" else 0.1, "reason": f"stub says {a}",
                "seconds": 3.0, "n_frames": len(frames_), "distinct": len({id(f) for f in frames_})}


fr1, fr2 = frames("cam1", 27, 45), frames("cam2", 27, 45)
for title, answers in (("both no_fall", ["no_fall"]), ("both fall", ["fall"]), ("tie", ["fall", "no_fall"]), ("checker error", ["error"])):
    app.VERIFIERS = [Stub(answers)]
    app.CHECKS = app.ThreadPoolExecutor(max_workers=1)
    code = title.replace(" ", "-")
    res, seen, t0 = {}, [], time.monotonic()
    th = [threading.Thread(target=phone, args=(c, code, n, f, res)) for c, n, f in (("cam1", "door", fr1), ("cam2", "corner", fr2))]
    [t.start() for t in th]
    room = app.get_room(code)
    while any(t.is_alive() for t in th):  # what a viewer of the dashboard sees
        html, _ = app.dashboard(code)
        for word in ("checking...", "FALL DETECTED"):
            if word in html and word not in seen:
                seen.append(word)
        time.sleep(0.2)
    [t.join() for t in th]
    app.CHECKS.shutdown(wait=True)
    html, _ = app.dashboard(code)
    print(f"--- {title}: dashboard showed {seen}; confirmed={room.confirmed is not None}; checking={room.checking}")
    if room.last_alarm:
        print(f"    alarm at video time {27 + room.last_alarm - t0:.1f} s; alarm_cams={room.alarm_cams}; "
              f"live then={[c['name'] for c in room.live(room.last_alarm).values()]}")
    print("    " + "\n    ".join(room.log))
    print("    scores door:", " ".join(h.split("P(fall) ")[1].split()[0] for h in res["door"]["hist"][::-1]))
    print("    scores corner:", " ".join(h.split("P(fall) ")[1].split()[0] for h in res["corner"]["hist"][::-1]))
    for r in room.reports[:1]:
        print("    clip frames:", r.get("n_frames"), "distinct:", r.get("distinct"))
