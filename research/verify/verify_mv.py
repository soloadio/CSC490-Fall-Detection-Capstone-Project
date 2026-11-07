#!/usr/bin/env python3
"""Multi-camera verification: VideoMAE triggers, every camera's Gemma looks, then the cameras compare notes.

Builds on verify_alarms.py (same clip format, frame cutting and Gemma scoring). Stages:
  events   (CPU, light) group the deployed-setting VideoMAE alarms of one take into trigger events (alarms of
           different cameras within --merge s of each other, on the reference camera's clock) and list a clip for
           EVERY camera of the take (the first --cams cameras) at the event time, offsets from the labels as in
           fusion_metrics.py. Writes clips_mv.jsonl (cut and score it with verify_alarms.py --clips clips_mv.jsonl)
           and events_mv.jsonl.
  report   (CPU, light) per event, the cameras' Gemma verdicts combined by simple rules (any / majority / all) and
           compared with the single-camera and VideoMAE-only baselines.

Usage:
    python3 verify_mv.py events [--variant wait1]
    python3 verify_alarms.py frames --clips clips_mv.jsonl ; python3 verify_alarms.py score --clips clips_mv.jsonl
    python3 verify_mv.py report
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)] + [str(p) for p in HERE.parent.iterdir() if p.is_dir()]  # sibling folders of research/
import eval_stream as es  # noqa: E402
import fusion_metrics as F  # noqa: E402
from stream_metrics import load_scores, load_truth  # noqa: E402
from verify_alarms import SCORES, SETTINGS, VARIANTS, classify  # noqa: E402


def stage_events(a):
    clips, events, n = [], [], 0
    w, k, th = SETTINGS["deployed"]
    for split in ("val", "test"):
        recs = load_scores(SCORES / f"scores_{split}.jsonl")
        truth = load_truth(recs)
        files = {r["video"]: r["file"] for r in es.recordings(split, ["edf", "mcfd", "up_fall"])}
        alarms = classify(recs, truth, "deployed", w, k, th)
        takes = defaultdict(list)
        for r in recs:
            takes[F.take_key(r)].append(r)
        for key, cams in sorted(takes.items()):
            cams = sorted(cams, key=lambda r: int(r["cam"]))[:a.cams]
            if len(cams) < 2:
                continue
            ref = truth.get(cams[0]["video"], [])
            off = {c["video"]: 0.0 if c is cams[0] else F.clock_offset(ref, truth.get(c["video"], []))[0] for c in cams}
            trig = sorted(((al["t"] - off[al["video"]], al) for al in alarms if al["video"] in off), key=lambda x: x[0])
            groups = []
            for t, al in trig:
                if groups and t - groups[-1][-1][0] <= a.merge:
                    groups[-1].append((t, al))
                else:
                    groups.append([(t, al)])
            for g in groups:
                kinds = {al["kind"] for _, al in g}
                kind = "hit" if "hit" in kinds else "fa" if kinds == {"fa"} else "neutral"
                if kind == "neutral":
                    continue
                t_ref = g[0][0]
                ev = {"event_id": f"{split}_ev{n:04d}", "split": split, "take": key, "t_ref": round(t_ref, 3),
                      "kind": kind, "triggered_by": sorted({al["cam"] for _, al in g}), "clips": {}}
                for c in cams:
                    cid = f"{split}_mv{n:04d}_cam{c['cam']}"
                    clips.append({"clip_id": cid, "split": split, "variant": a.variant, "file": files[c["video"]],
                                  "t_end": t_ref + off[c["video"]] + VARIANTS[a.variant], "video": c["video"],
                                  "dataset": c["dataset"], "scenario": c["scenario"], "cam": c["cam"],
                                  "t": t_ref + off[c["video"]], "alarm": {}})
                    ev["clips"][c["cam"]] = cid
                events.append(ev)
                n += 1
    (a.out / "clips_mv.jsonl").write_text("".join(json.dumps(c) + "\n" for c in clips))
    (a.out / "events_mv.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    by = defaultdict(int)
    for e in events:
        by[e["split"], e["kind"], e["take"].split("/")[0]] += 1
    print(f"{len(events)} trigger events, {len(clips)} clips (cameras per take: up to {a.cams}) -> {a.out}")
    for key in sorted(by):
        print("  ", *key, by[key])


def load_results(out, tag, suffix):
    p = out / f"gemma_{tag}{suffix}.jsonl"
    return {r["clip_id"]: r for r in map(json.loads, open(p)) if "error" not in r} if p.exists() else {}


def stage_report(a):
    events = [json.loads(l) for l in open(a.out / "events_mv.jsonl")]
    lines = []
    for tag in ("zeroshot", "lora"):
        res = load_results(a.out, tag, "_clips_mv")
        if not res:
            continue
        for tau in a.taus:
            lines.append(f"\n## Gemma {tag}, confirm threshold tau = {tau}")
            lines.append(f"{'split':5} {'rule':26} {'falls kept':>12} {'false alarms removed':>22}")
            for split in ("val", "test"):
                ev = [e for e in events if e["split"] == split and all(c in res for c in e["clips"].values())]
                hit, fa = [e for e in ev if e["kind"] == "hit"], [e for e in ev if e["kind"] == "fa"]

                def votes(e):
                    return [res[c]["p_fall"] >= tau for c in e["clips"].values()]

                rules = {"VideoMAE only (no check)": lambda e: True,
                         "first camera alone": lambda e: votes(e)[0],
                         "any camera confirms": lambda e: any(votes(e)),
                         "majority confirms": lambda e: sum(votes(e)) * 2 > len(votes(e)),
                         "all cameras confirm": lambda e: all(votes(e))}
                for name, rule in rules.items():
                    kept = sum(rule(e) for e in hit)
                    removed = sum(not rule(e) for e in fa)
                    lines.append(f"{split:5} {name:26} {kept:5d}/{len(hit):<6d} {removed:10d}/{len(fa):<10d}")
    text = "\n".join(lines)
    (a.out / "report_mv.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["events", "report"])
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/verify_gemma"))
    ap.add_argument("--cams", type=int, default=3, help="cameras per take (the first N by camera number)")
    ap.add_argument("--merge", type=float, default=3.0, help="alarms of different cameras this close are one event (s)")
    ap.add_argument("--variant", choices=list(VARIANTS), default="wait1", help="clip ends at the alarm or 1 s after")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.9])
    a = ap.parse_args()
    {"events": stage_events, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
