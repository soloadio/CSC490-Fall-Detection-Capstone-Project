#!/usr/bin/env python3
"""Turn sliding-window scores (eval_stream.py) into alarms and score them like a live fall detector.

Alarm rule, per camera recording, for a window length W, threshold th and k:
    raise an alarm when P(fall) >= th for k consecutive windows; then stay silent until the score has
    dropped below th again AND --cooldown seconds have passed.

Ground truth, from OmniFall labels/<dataset>.csv: consecutive `fall` / `fallen` segments (gaps up to
--merge-gap s) form one fall episode. An episode containing a `fall` segment is a fall to detect; its
onset is the start of the first `fall` segment.
    hit          first alarm in [onset - pre_tol, episode end + post_tol]; delay = alarm - onset
    neutral      later alarms of the same episode, alarms near a `fallen`-only episode
                 (recording starts on the floor), alarms in unlabelled time
    false alarm  every other alarm
False alarms per hour = false alarms / monitored non-fall time (window ends outside any episode
+ tolerance and inside labelled time; each window end stands for one stride of eval_stream.py).

eval_stream.py pads early windows with the first frame, so every W covers the same span from the start;
--t0 can still exclude the first seconds (falls with an earlier onset are then reported as skipped).

Choosing settings: on --select (val), for each (W, k) take the th with the highest fall recall whose
false alarms per hour <= --max-fa-per-hour (ties: fewer false alarms, then shorter delay), then the
best (W, k) the same way. Those settings are applied once to --report (test).

Usage:
    python3 stream_metrics.py --select <dir>/scores_val.jsonl --report <dir>/scores_test.jsonl
    python3 stream_metrics.py --select <dir>/scores_val.jsonl --max-fa-per-hour 2 --sweep
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = Path("/scratch/wuhungma/omnifall/labels")
FALL, FALLEN = "1", "2"
THRESHOLDS = [round(x, 3) for x in list(np.arange(0.30, 0.95, 0.05)) + [0.95, 0.97, 0.98, 0.99, 0.995]]


def load_scores(path):
    return [json.loads(l) for l in open(path)]


def load_truth(recs):
    """Per recording: labelled segments and fall episodes."""
    want = defaultdict(set)
    for r in recs:
        want[r["dataset"]].add(r["video"].split("/", 1)[1])
    segs = defaultdict(list)
    for d, paths in want.items():
        for row in csv.DictReader(open(LABELS / f"{d}.csv")):
            if row["path"] in paths:
                segs[f"{d}/{row['path']}"].append((float(row["start"]), float(row["end"]), row["label"]))
    return {v: sorted(s) for v, s in segs.items()}


def episodes(segs, merge_gap):
    eps = []
    for s, e, lab in segs:
        if lab not in (FALL, FALLEN):
            continue
        if eps and s <= eps[-1]["end"] + merge_gap:
            ep = eps[-1]
            ep["end"] = max(ep["end"], e)
        else:
            ep = {"start": s, "end": e, "onset": None}
            eps.append(ep)
        if lab == FALL and ep["onset"] is None:
            ep["onset"] = s
    return eps


def alarms(t, p, th, k, cooldown):
    out, run, armed, last = [], 0, True, -1e9
    for ti, pi in zip(t, p):
        if pi >= th:
            run += 1
        else:
            run, armed = 0, True
        if run >= k and armed and ti - last >= cooldown:
            out.append(ti)
            armed, last = False, ti
    return out


def evaluate(recs, truth, w, th, k, a):
    """Per-dataset counts for one setting."""
    stats = defaultdict(lambda: {"falls": 0, "hits": 0, "fa": 0, "neg_s": 0.0, "delays": [], "skipped": 0,
                                 "recordings": 0})
    for r in recs:
        segs = truth.get(r["video"], [])
        eps = episodes(segs, a.merge_gap)
        win = r["windows"][w]
        t, p = np.array(win["t"]), np.array(win["p"])
        keep = t >= a.t0
        t, p = t[keep], p[keep]
        st = stats[r["dataset"]]
        st["recordings"] += 1
        falls = [e for e in eps if e["onset"] is not None and e["onset"] >= a.t0]
        st["skipped"] += sum(e["onset"] is not None and e["onset"] < a.t0 for e in eps)
        st["falls"] += len(falls)

        def zone(x):  # "fall" (inside an episode window), "ignore" (unlabelled), or "neg"
            if any(e["start"] - a.pre_tol <= x <= e["end"] + a.post_tol for e in eps):
                return "fall"
            if not any(s <= x <= e for s, e, _ in segs):
                return "ignore"
            return "neg"

        st["neg_s"] += sum(zone(x) == "neg" for x in t) * a.stride
        hit = set()
        for x in alarms(t, p, th, k, a.cooldown):
            match = [i for i, e in enumerate(falls) if e["onset"] - a.pre_tol <= x <= e["end"] + a.post_tol]
            if match:
                i = match[0]
                if i not in hit:
                    hit.add(i)
                    st["delays"].append(x - falls[i]["onset"])
            elif zone(x) == "neg":
                st["fa"] += 1
        st["hits"] += len(hit)
    return stats


def summarize(stats):
    tot = {"falls": 0, "hits": 0, "fa": 0, "neg_s": 0.0, "delays": [], "skipped": 0, "recordings": 0}
    rows = {}
    for d in sorted(stats):
        rows[d] = stats[d]
        for key in tot:
            tot[key] += stats[d][key]
    rows["all"] = tot
    out = {}
    for d, s in rows.items():
        dl = sorted(s["delays"])
        out[d] = {"recordings": s["recordings"], "falls": s["falls"], "detected": s["hits"],
                  "recall": s["hits"] / s["falls"] if s["falls"] else float("nan"),
                  "false_alarms": s["fa"], "neg_hours": s["neg_s"] / 3600,
                  "fa_per_hour": s["fa"] / (s["neg_s"] / 3600) if s["neg_s"] else float("nan"),
                  "delay_median": float(np.median(dl)) if dl else float("nan"),
                  "delay_p90": float(np.percentile(dl, 90)) if dl else float("nan"),
                  "skipped_early_falls": s["skipped"]}
    return out


def better(x, y):
    """Is summary x (pooled) a better operating point than y?"""
    if y is None:
        return True
    kx = (x["recall"], -x["fa_per_hour"], -np.nan_to_num(x["delay_median"], nan=1e9))
    ky = (y["recall"], -y["fa_per_hour"], -np.nan_to_num(y["delay_median"], nan=1e9))
    return kx > ky


def table(summ, title):
    lines = [f"\n{title}",
             f"{'dataset':8} {'recs':>5} {'falls':>5} {'caught':>6} {'recall':>7} {'false al.':>9} {'non-fall h':>10} "
             f"{'FA/hour':>8} {'delay med':>9} {'delay p90':>9}"]
    for d, s in summ.items():
        lines.append(f"{d:8} {s['recordings']:5d} {s['falls']:5d} {s['detected']:6d} {s['recall']:7.1%} "
                     f"{s['false_alarms']:9d} {s['neg_hours']:10.2f} {s['fa_per_hour']:8.2f} "
                     f"{s['delay_median']:8.1f}s {s['delay_p90']:8.1f}s")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--select", type=Path, required=True, help="scores file used to choose W, k, threshold (val)")
    ap.add_argument("--report", type=Path, help="scores file the chosen settings are applied to once (test)")
    ap.add_argument("--max-fa-per-hour", type=float, default=1.0)
    ap.add_argument("--k", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--pre-tol", type=float, default=0.5)
    ap.add_argument("--post-tol", type=float, default=2.0)
    ap.add_argument("--merge-gap", type=float, default=1.0)
    ap.add_argument("--t0", type=float, default=0.0, help="ignore window ends and fall onsets before this time (s)")
    ap.add_argument("--sweep", action="store_true", help="print the full threshold sweep for every (W, k)")
    ap.add_argument("--json", type=Path, help="write the chosen settings and all summaries here")
    a = ap.parse_args()

    sel = load_scores(a.select)
    cfg = json.loads((a.select.parent / "config.json").read_text())
    a.stride = float(cfg["stride"])
    ws = sorted(sel[0]["windows"], key=float)
    truth = load_truth(sel)
    print(f"select on {a.select}: {len(sel)} recordings | windows {ws} s | stride {a.stride} s | "
          f"evaluated from t >= {a.t0} s | target <= {a.max_fa_per_hour} false alarms/hour")

    best, per_wk, curves = None, {}, {}
    for w in ws:
        for k in a.k:
            choice, curve = None, []
            for th in THRESHOLDS:
                s = summarize(evaluate(sel, truth, w, th, k, a))
                curve.append((th, s["all"]))
                if s["all"]["fa_per_hour"] <= a.max_fa_per_hour and better(s["all"], choice and choice[1]["all"]):
                    choice = (th, s)
            curves[(w, k)] = curve
            per_wk[(w, k)] = choice
            if choice and better(choice[1]["all"], best and best[3]["all"]):
                best = (w, k, choice[0], choice[1])

    print(f"\nBest threshold per (window, k) on {a.select.stem}, pooled over datasets:")
    print(f"{'W':>4} {'k':>2} {'th':>6} {'recall':>7} {'FA/hour':>8} {'delay med':>9}")
    for (w, k), c in per_wk.items():
        if c:
            s = c[1]["all"]
            print(f"{w:>4} {k:2d} {c[0]:6.3f} {s['recall']:7.1%} {s['fa_per_hour']:8.2f} {s['delay_median']:8.1f}s")
        else:
            print(f"{w:>4} {k:2d}   none meets the false-alarm target")
    if a.sweep:
        for (w, k), curve in curves.items():
            print(f"\nsweep W={w} k={k}: " + "  ".join(f"th {th}: rec {s['recall']:.0%} fa/h {s['fa_per_hour']:.1f}"
                                                     for th, s in curve))
    if best is None:
        print("\nNo setting meets the target on the selection split; raise --max-fa-per-hour.")
        return 1
    w, k, th, s_sel = best
    print(f"\nCHOSEN: window {w} s, k = {k}, threshold {th}, cooldown {a.cooldown} s")
    print(table(s_sel, f"{a.select.stem} (used to choose the settings)"))
    result = {"window": w, "k": k, "threshold": th, "cooldown": a.cooldown, "select": s_sel,
              "select_curve": [(th_, s_) for th_, s_ in curves[(w, k)]]}
    if a.report:
        rep = load_scores(a.report)
        s_rep = summarize(evaluate(rep, load_truth(rep), w, th, k, a))
        print(table(s_rep, f"{a.report.stem} (settings fixed above, run once)"))
        result["report"] = s_rep
    if a.json:
        a.json.write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
