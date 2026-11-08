"""Two checkers (stand-ins for Gemma and Qwen) on two cameras: run_check must ask all four (camera, checker) pairs at
once (about 3 s, not 12 s), and decide() must keep the alarm on a 2-2 tie and cancel on 0-4."""
import time
import numpy as np
from verify_rooms_app import app
from verifier import RawBuffer, decide


class Stub:
    ready = True
    def __init__(self, name, verdict):
        self.name, self.verdict = name, verdict
    def check(self, frames):
        time.sleep(3)
        return {"verifier": self.name, "verdict": self.verdict, "p_fall": 0.9 if self.verdict == "fall" else 0.1,
                "reason": f"{self.name} says {self.verdict}", "seconds": 3.0}


def fill():  # 8 s of fake video ending now on both cameras
    now = time.monotonic()
    for sid in ("s1", "s2"):
        app.RAW[sid] = RawBuffer()
        for i in range(80):
            app.RAW[sid].push(now - 8 + i * 0.1, np.zeros((240, 320, 3), np.uint8))
app.VERIFY_WAIT = 0.0
for title, a, b in (("Gemma fall, Qwen no_fall", "fall", "no_fall"), ("both no_fall", "no_fall", "no_fall"),
                    ("both fall", "fall", "fall")):
    fill()
    app.VERIFIERS = [Stub("Gemma", a), Stub("Qwen", b)]
    t = time.perf_counter()
    reports = app.run_check({"s1": "door", "s2": "corner"})
    print(f"--- {title}: {len(reports)} reports in {time.perf_counter() - t:.1f} s; alarm stands = {decide(reports)}")
    print("\n".join(app.report_lines(reports)))
app.VERIFIERS = [Stub("Gemma", "fall"), Stub("Qwen", "fall")]
app.VERIFIERS[1].check = lambda f: (_ for _ in ()).throw(RuntimeError("boom"))
fill()
r = app.run_check({"s1": "door"})
print("--- Qwen errors:", decide(r), app.report_lines(r))
