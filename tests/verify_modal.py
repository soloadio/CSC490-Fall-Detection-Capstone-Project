"""Send three stored test clips (32 JPEGs each, the exact files the offline run used) to the deployed Modal checker and
compare P(fall) with the offline Gemma + LoRA scores. Second round shows the warm-container time."""
import json, sys, time
from pathlib import Path
import modal

OFFLINE = {"test_00257_wait1": ("real fall, kept", 0.99331), "test_00317_wait1": ("false alarm, removed", 0.0028),
           "test_00261_wait1": ("real fall, missed", 0.06755)}
check = modal.Cls.from_name("fall-verifier", "Verifier")().check
for rnd in (1, 2):
    for cid, (what, p_off) in OFFLINE.items():
        jpegs = [(Path("/scratch/wuhungma/verify_gemma/clips") / cid / f"f{i:02d}.jpg").read_bytes() for i in range(32)]
        t = time.perf_counter()
        r = check.remote(jpegs)
        print(f"round {rnd} {cid} ({what}): Modal P {r['p_fall']:.4f} vs offline {p_off:.4f} | GPU {r['seconds']:.1f} s, "
              f"round trip {time.perf_counter() - t:.1f} s", flush=True)
        if rnd == 1:
            print("   ", r["text"].replace("\n", " ")[:200], flush=True)
