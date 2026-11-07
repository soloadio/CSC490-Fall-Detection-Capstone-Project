#!/usr/bin/env python3
"""Multi-camera fusion of sliding-window scores (eval_stream.py), scored like stream_metrics.py.

A "take" is one recording session seen by several cameras (EDF: 2 views, MCFD: 8, UP-Fall: 2). For every group of
cameras of a take (default: every pair, i.e. what two phones in a room would see), the per-camera P(fall) series
are put on one clock and combined into one series per rule:
    single     each camera alone (baseline; both cameras of every pair, so each rule sees the same video)
    any        max over cameras: alarm when any camera sees a fall
    mean       average over cameras
    two agree  second-highest of each camera's max over the last --agree-window s: alarm when at least two
               cameras saw P(fall) >= threshold within that time
The fused series then goes through the same alarm rule and fall / false-alarm scoring as stream_metrics.py
(its functions are reused), with the labels of the group's first camera.

Clocks: UP-Fall cameras are synchronised, EDF and MCFD cameras are not (EDF views ~6 s apart, MCFD up to a few s).
A live system timestamps frames on arrival, so the cameras share a clock; here each camera's offset to the reference
camera is estimated from the labels (median shift of matching label boundaries) and only the time covered by all
cameras of the group is scored.

Settings: per rule, (W, k, threshold) are chosen on --select (val) exactly as in stream_metrics.py and applied
once to --report (test); every rule is also reported at the deployed setting (--fixed).

Usage:
    python3 fusion_metrics.py --select <dir>/scores_val.jsonl --report <dir>/scores_test.jsonl --cooldown 10 \
        --max-fa-per-hour 30 --json <out>.json
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from stream_metrics import THRESHOLDS, better, evaluate, load_scores, load_truth, summarize, table

GRID = sorted(set(THRESHOLDS) | {0.10, 0.15, 0.20, 0.25})  # fusion may want lower thresholds than one camera
RULES = ["single", "any", "mean", "two agree"]


def take_key(r):
    return r["video"].rsplit("Camera", 1)[0] if r["dataset"] == "up_fall" else f"{r['dataset']}/{r['scenario']}"


def clock_offset(ref, cam):
    """Seconds to subtract from `cam`'s clock to put it on `ref`'s clock (both: sorted (start, end, label))."""
    if len(ref) > 1 and [l for *_, l in ref] == [l for *_, l in cam]:
        return float(np.median([c[0] - r[0] for r, c in zip(ref[1:], cam[1:])])), "boundaries"
    fr, fc = [s for s, _, l in ref if l == "1"], [s for s, _, l in cam if l == "1"]
    if fr and len(fr) == len(fc):
        return float(np.median(np.subtract(fc, fr))), "falls"
    return 0.0, "none"


def on_grid(t_ref, t_cam, p_cam, off):
    """Camera scores at the reference window ends (nearest window end, camera clock shifted by `off`)."""
    tc = np.asarray(t_cam) - off
    j = np.clip(np.searchsorted(tc, t_ref), 1, len(tc) - 1)
    j = np.where(np.abs(tc[j - 1] - t_ref) <= np.abs(tc[j] - t_ref), j - 1, j)
    return np.asarray(p_cam)[j]


def recent_max(p, n):
    out = p.copy()
    for i in range(1, n):
        out[i:] = np.maximum(out[i:], p[:-i])
    return out


def build(recs, truth, group, stride, agree_window):
    """Fused pseudo-recordings per rule, plus their (clipped, reference-clock) labels."""
    takes = defaultdict(list)
    for r in recs:
        takes[take_key(r)].append(r)
    out, lab, offsets = {rule: [] for rule in RULES}, {}, []
    n_agree = int(round(agree_window / stride)) + 1
    for key, cams in sorted(takes.items()):
        cams = sorted(cams, key=lambda r: int(r["cam"]))
        groups = combinations(cams, 2) if group == "pairs" else [cams]
        for g in groups:
            if len(g) < 2:
                continue
            ref = truth.get(g[0]["video"], [])
            offs = [0.0] + [clock_offset(ref, truth.get(c["video"], []))[0] for c in g[1:]]
            offsets += [(c["dataset"], clock_offset(ref, truth.get(c["video"], []))[1], o) for c, o in zip(g[1:], offs[1:])]
            gid = key + "|" + "+".join(c["cam"] for c in g)
            lo = max(-o for o in offs)  # time recorded by every camera, on the reference clock
            hi = min(c["duration"] - o for c, o in zip(g, offs))
            fused = {rule: {} for rule in RULES}
            for w in g[0]["windows"]:
                t = np.asarray(g[0]["windows"][w]["t"])
                t = t[(t >= lo) & (t <= hi)]
                P = np.stack([on_grid(t, c["windows"][w]["t"], c["windows"][w]["p"], o) for c, o in zip(g, offs)])
                for i, c in enumerate(g):
                    fused["single"].setdefault(i, {})[w] = {"t": t.tolist(), "p": P[i].tolist()}
                fused["any"][w] = {"t": t.tolist(), "p": P.max(0).tolist()}
                fused["mean"][w] = {"t": t.tolist(), "p": P.mean(0).tolist()}
                agree = np.sort(np.stack([recent_max(p, n_agree) for p in P]), 0)[-2]
                fused["two agree"][w] = {"t": t.tolist(), "p": agree.tolist()}
            # labels clipped to that time; a fall that starts before every camera is recording is dropped
            # (its `fallen` part then counts as neutral, as in stream_metrics.py)
            lab[gid] = [(max(s, lo), min(e, hi), l) for s, e, l in ref
                        if e > lo and s < hi and not (l == "1" and s < lo)]
            base = {"dataset": g[0]["dataset"], "video": gid}
            for rule in ("any", "mean", "two agree"):
                out[rule].append({**base, "windows": fused[rule]})
            for i in fused["single"]:
                out["single"].append({**base, "windows": fused["single"][i]})
    return out, lab, offsets


def choose(recs, truth, a, target):
    best = None
    for w in sorted(recs[0]["windows"], key=float):
        for k in a.k:
            for th in GRID:
                s = summarize(evaluate(recs, truth, w, th, k, a))
                if s["all"]["fa_per_hour"] <= target and better(s["all"], best and best[3]["all"]):
                    best = (w, k, th, s)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--group", choices=["pairs", "all"], default="pairs", help="every 2-camera pair, or all cameras")
    ap.add_argument("--max-fa-per-hour", type=float, nargs="+", default=[30.0])
    ap.add_argument("--fixed", default="3.0,1,0.30", help="W,k,threshold reported for every rule (deployed setting)")
    ap.add_argument("--agree-window", type=float, default=1.0)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--cooldown", type=float, default=10.0)
    ap.add_argument("--pre-tol", type=float, default=0.5)
    ap.add_argument("--post-tol", type=float, default=2.0)
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    a.t0 = 0.0
    a.stride = float(json.loads((a.select.parent / "config.json").read_text())["stride"])

    data = {}
    for split, path in (("val", a.select), ("test", a.report)):
        recs = load_scores(path)
        data[split] = build(recs, load_truth(recs), a.group, a.stride, a.agree_window)
        offs = data[split][2]
        by = defaultdict(list)
        for d, how, o in offs:
            by[d, how].append(abs(o))
        print(f"{split}: {len(data[split][0]['any'])} camera groups ({a.group}); clock offsets |s| by dataset/method: "
              + "; ".join(f"{d} {how} n={len(v)} median {np.median(v):.2f} max {max(v):.2f}" for (d, how), v in sorted(by.items())))

    result = {"group": a.group, "agree_window": a.agree_window, "cooldown": a.cooldown, "fixed": {}, "chosen": {}}
    w, k, th = a.fixed.split(",")
    w, k, th = str(float(w)), int(k), float(th)
    print(f"\n===== fixed setting: window {w} s, k = {k}, threshold {th} (deployed) =====")
    for rule in RULES:
        recs, lab, _ = data["test"]
        s = summarize(evaluate(recs[rule], lab, w, th, k, a))
        result["fixed"][rule] = s
        print(table(s, f"[test] {rule}"))
    for target in a.max_fa_per_hour:
        print(f"\n===== chosen on val per rule, target <= {target} false alarms/hour =====")
        for rule in RULES:
            c = choose(data["val"][0][rule], data["val"][1], a, target)
            if c is None:
                print(f"\n[{rule}] no setting meets the target on val")
                continue
            cw, ck, cth, s_val = c
            s_test = summarize(evaluate(data["test"][0][rule], data["test"][1], cw, cth, ck, a))
            result["chosen"].setdefault(str(target), {})[rule] = {"window": cw, "k": ck, "threshold": cth,
                                                                 "val": s_val, "test": s_test}
            print(f"\n[{rule}] chosen on val: window {cw} s, k = {ck}, threshold {cth} "
                  f"(val recall {s_val['all']['recall']:.1%}, {s_val['all']['fa_per_hour']:.1f} FA/h)")
            print(table(s_test, f"[test] {rule}"))
    if a.json:
        a.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
