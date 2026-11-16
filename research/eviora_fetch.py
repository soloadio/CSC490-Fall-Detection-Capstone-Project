#!/usr/bin/env python3
"""Fetch selected EVIORA-HAR trimmed clips (Zenodo record 22239523, CC BY-NC 4.0) without downloading the 131 GB
archives: read each archive's central directory, then HTTP-range-fetch only the wanted members.

Wanted: both static cameras (CAM 1 = full view, CAM 2 = chair-occluded view), classes falling_* (480 falls) and the
deliberate downward movements sit-lay and stand-sit (for false alarms). Resumable.

    python3 eviora_fetch.py --out /scratch/wuhungma/eviora/clips
"""
import argparse, struct, subprocess, zlib
from pathlib import Path

REC = "https://zenodo.org/api/records/22239523/files/{}/content"
PARTS = {"CAM 1": "EVIORA_Trimmed_Part04.zip", "CAM 2": "EVIORA_Trimmed_Part05.zip"}
CLASSES = ("falling_forward", "falling_sideward", "falling_backward", "sit-lay", "stand-sit")


def curl(url, rng):
    return subprocess.run(["curl", "-s", "-L", "--retry", "5", "-r", rng, "--max-time", "600", url],
                          capture_output=True, check=True).stdout


def members(part):
    d = curl(REC.format(part), "-3000000")
    i, out = d.find(b"PK\x01\x02"), []
    while i != -1 and i + 46 <= len(d):
        method = struct.unpack("<H", d[i+10:i+12])[0]
        csize, usize = struct.unpack("<II", d[i+20:i+28]); n, m, k = struct.unpack("<HHH", d[i+28:i+34])
        off = struct.unpack("<I", d[i+42:i+46])[0]
        name = d[i+46:i+46+n].decode("utf-8", "replace"); extra, j = d[i+46+n:i+46+n+m], 0
        while j + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[j:j+4]); body, q = extra[j+4:j+4+hsz], 0
            if hid == 1:
                if usize == 0xFFFFFFFF: usize = struct.unpack("<Q", body[q:q+8])[0]; q += 8
                if csize == 0xFFFFFFFF: csize = struct.unpack("<Q", body[q:q+8])[0]; q += 8
                if off == 0xFFFFFFFF: off = struct.unpack("<Q", body[q:q+8])[0]; q += 8
            j += 4 + hsz
        out.append((name, off, csize, method))
        i = d.find(b"PK\x01\x02", i + 46 + n + m + k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for cam, part in PARTS.items():
        todo = [x for x in members(part) if x[0].endswith(".mp4") and x[0].split("/")[3] in CLASSES]
        print(f"{cam}: {len(todo)} clips, {sum(x[2] for x in todo) / 1e9:.1f} GB", flush=True)
        for n_done, (name, off, csize, method) in enumerate(todo, 1):
            dest = a.out / cam.replace(" ", "") / name.split("/")[3] / Path(name).name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            d = curl(REC.format(part), f"{off}-{off + 30 + 1024 + csize}")
            assert d[:4] == b"PK\x03\x04", name
            n, m = struct.unpack("<HH", d[26:30]); body = d[30+n+m:30+n+m+csize]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(zlib.decompress(body, -15) if method == 8 else body); tmp.rename(dest)
            if n_done % 100 == 0:
                print(f"  {cam} {n_done}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
