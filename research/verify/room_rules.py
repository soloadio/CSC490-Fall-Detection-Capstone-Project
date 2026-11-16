#!/usr/bin/env python3
"""How should the room combine the checkers' answers? Compares decision rules offline on saved checker results.

Each rule gets, per alarm, one report per (camera, checker): P(fall) and the evidence sentence. Three settings:
  no occlusion      the multi-camera alarm events (events_mv.jsonl), every camera as filmed
  one cam blocked   the same events with one camera blinded by a fake occluder (occlusion_exp.py), each camera in
                    turn; only scenarios where another camera's VideoMAE alarmed (one camera triggers the room)
  single camera     the single-camera alarm clips (clips.jsonl, wait1)

A camera whose checkers say "no visible movement / remains in the same position" is treated as unable to see the
person (app/verifier.py NO_MOVEMENT): never said about a visible real fall, said about about half of blocked ones.

    python3 room_rules.py                                   # Qwen results of the deployed adapter
    python3 room_rules.py --qwen /scratch/wuhungma/verify_qwen_px346k --qwen-occ <occluded run>   # another Qwen
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "app")]
from verifier import NO_MOVEMENT  # noqa: E402  the app's own pattern

G_DIR = Path("/scratch/wuhungma/verify_gemma_reasons")
OCC = Path("/scratch/wuhungma/occlusion")
TAU = 0.5


def load(p):
    return {r["clip_id"]: r for r in map(json.loads, open(p)) if "p_fall" in r}


def evidence(r):
    t = r["text"]
    try:
        return json.loads(t[t.index("{"):t.rindex("}") + 1]).get("evidence", "")
    except Exception:  # noqa: BLE001
        return ""


def vote(cam, m, r, blind=False):
    return dict(cam=cam, m=m, p=r["p_fall"], still=bool(NO_MOVEMENT.search(evidence(r))), blind=blind)


def seeing(v):
    """Votes of the cameras that can see (no checker on that camera said 'no movement'); all votes if none can."""
    still = {x["cam"] for x in v if x["still"]}
    return [x for x in v if x["cam"] not in still] or v


def mean(v):
    return sum(x["p"] for x in v) / len(v)


RULES = {
    "old: >= half of all votes, tie keeps": lambda v: sum(x["p"] >= TAU for x in v) * 2 >= len(v),
    "mean P over all votes >= 0.5": lambda v: mean(v) >= TAU,
    "mean P over cameras that can see": lambda v: mean(seeing(v)) >= TAU,
    "APP: any checker on a camera that can see": lambda v: any(x["p"] >= TAU for x in seeing(v)),
    "oracle: mean P without the blocked camera": lambda v: mean([x for x in v if not x["blind"]] or v) >= TAU,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qwen", type=Path, default=Path("/scratch/wuhungma/verify_qwen_reasons"))
    ap.add_argument("--qwen-occ", type=Path, default=OCC / "qwen")
    a = ap.parse_args()
    r0 = {"G": load(G_DIR / "gemma_lora_clips_mv.jsonl"), "Q": load(a.qwen / "gemma_lora_clips_mv.jsonl")}
    r1 = {"G": load(OCC / "gemma_lora_clips_mv.jsonl"), "Q": load(a.qwen_occ / "gemma_lora_clips_mv.jsonl")}
    events = [json.loads(l) for l in open(G_DIR / "events_mv.jsonl")]
    rows, n = collections.Counter(), collections.Counter()

    def add(cond, split, kind, v):
        n[(cond, split, kind)] += 1
        for name, f in RULES.items():
            rows[(cond, split, kind, name)] += f(v)

    for e in events:
        kind = "fall" if e["kind"] == "hit" else "fa"
        cams = sorted(e["clips"])
        add("no occlusion", e["split"], kind, [vote(c, m, r0[m][e["clips"][c]]) for c in cams for m in "GQ"])
        for X in cams:
            if not set(e["triggered_by"]) - {X} or any(e["clips"][X] not in r1[m] for m in "GQ"):
                continue
            add("one cam blocked", e["split"], kind,
                [vote(c, m, (r1 if c == X else r0)[m][e["clips"][c]], c == X) for c in cams for m in "GQ"])
    g, q = load(G_DIR / "gemma_lora.jsonl"), load(a.qwen / "gemma_lora.jsonl")
    for c in map(json.loads, open(G_DIR / "clips.jsonl")):
        k = c.get("alarm", {}).get("deployed", {}).get("kind")
        if c.get("variant") == "wait1" and k in ("hit", "fa") and c["clip_id"] in g and c["clip_id"] in q:
            add("single camera", c["split"], "fall" if k == "hit" else "fa",
                [vote("1", "G", g[c["clip_id"]]), vote("1", "Q", q[c["clip_id"]])])

    for cond in ("no occlusion", "one cam blocked", "single camera"):
        print(f"\n[{cond}]   falls kept: val  test | false alarms kept: val  test")
        for name in RULES:
            f = lambda s, k: f"{rows[(cond, s, k, name)]}/{n[(cond, s, k)]}"  # noqa: E731
            print(f"  {name:44} {f('val', 'fall'):>7} {f('test', 'fall'):>8} | {f('val', 'fa'):>5} {f('test', 'fa'):>5}")


if __name__ == "__main__":
    main()
