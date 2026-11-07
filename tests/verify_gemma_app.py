"""Second opinion inside the real app handlers: Gemma 3 12B + LoRA loaded as the verifier, two phones in one room replaying
MCFD chute01 (fall at ~35.5 s), then one phone without a room code. Prints the alarm logs and how long the check took."""
import os, sys, threading, time
os.environ.update(VERIFIERS="gemma", GEMMA_BASE="/home/wuhungma/scratch/models/gemma-3-12b-it",
                  GEMMA_ADAPTER="/scratch/wuhungma/lora/r16_bf16_ep1/adapter")
from verify_rooms_app import app, frames, phone  # noqa: E402

t0 = time.time()
while not all(v.ready for v in app.VERIFIERS):
    time.sleep(2)
    assert time.time() - t0 < 900, "model did not load"
print(f"verifier ready after {time.time() - t0:.0f} s", flush=True)
fr1, fr2 = frames("cam1", 27, 45), frames("cam2", 27, 45)
res = {}
th = [threading.Thread(target=phone, args=(c, "lab", n, f, res)) for c, n, f in (("cam1", "door", fr1), ("cam2", "corner", fr2))]
[t.start() for t in th]; [t.join() for t in th]
app.CHECKS.shutdown(wait=True)  # let the check finish
room = app.get_room("lab")
print("--- room log (newest first):"); print("\n".join(room.log))
print("checking:", room.checking, "| confirmed:", room.confirmed is not None)
app.CHECKS = app.ThreadPoolExecutor(max_workers=1)
res = {}
phone("cam1", "", "nameless", fr1, res)
app.CHECKS.shutdown(wait=True)
print("--- single camera, no room (newest first):"); print(app.alarm_log(res["nameless"]))
print("last_alarm set:", res["nameless"]["last_alarm"] is not None)
