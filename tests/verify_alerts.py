"""Two phone sessions in room 'lab' (MCFD chute01 cam1 + cam2, real time, 5 frames/s) with push alerts on; checks the
room log for the alarm and the alert outcome. Topic from argv."""
import sys, threading, time
sys.path.insert(0, "/scratch/wuhungma/deploy")
from verify_rooms_app import app, frames, phone  # noqa: E402  (loads the app and model)

topic = sys.argv[1]
print("set_alerts:", app.set_alerts("lab", topic))
print("bad topic:", app.set_alerts("lab2", "a b"))
fr1, fr2 = frames("cam1", 27, 45), frames("cam2", 27, 45)
res = {}
th = [threading.Thread(target=phone, args=(c, "lab", n, f, res)) for c, n, f in (("cam1", "door", fr1), ("cam2", "corner", fr2))]
[t.start() for t in th]; [t.join() for t in th]
time.sleep(12)
print("room log:", " / ".join(app.get_room("lab").log))
