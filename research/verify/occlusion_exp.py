#!/usr/bin/env python3
"""Fake-occlusion test of the multi-camera design: one camera cannot see the person (e.g. a fall behind a sofa).

Events = the multi-camera VideoMAE alarm events of verify_mv.py (events_mv.jsonl): an event starts when ANY camera's
VideoMAE alarms ("escalate on one camera") and every camera of the take gets a clip. In each scenario one camera X of
an event is blinded: a solid "furniture" box is drawn over the place where the person is at the end of the clip
(union of RT-DETR person boxes over the last 4 frames with a person, padded), in all 32 frames, as a static occluder would be.

Stages:
  mask     (GPU, small)  person boxes + occluded copies of every multi-camera clip in <out>/clips/<clip_id>/fNN.jpg,
                         plus how often a person is still detected after masking; previews in <out>/previews/
  (then score the occluded clips with the existing scripts, see occlusion_exp.sbatch:
     verify_alarms.py score --out <out> --clips clips_mv.jsonl ...   -> <out>/gemma_lora_clips_mv.jsonl
     qwen/verify_qwen.py --clips-dir <out> --out <out>/qwen ...       -> <out>/qwen/gemma_lora_clips_mv.jsonl)
  round2   (GPU)         the blinded camera's checkers again, with the other (unoccluded) cameras' reports in the prompt
  report   (CPU)         room decisions under each rule, for every (event, blinded camera) scenario

The VideoMAE stage is not re-run: a blinded camera is assumed not to alarm (worst case), so with the current
"two cameras agree" rule an event can only fire if two OTHER cameras alarmed; with "escalate on one camera" it fires if
any other camera alarmed (from triggered_by, merged within 3 s, so the two-agree count is approximate).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
import peer_round2 as P  # noqa: E402

SRC = Path("/scratch/wuhungma/verify_gemma_reasons")  # clips_mv.jsonl, events_mv.jsonl, clips/ (unoccluded)
G_ORIG, Q_ORIG = SRC, Path("/scratch/wuhungma/verify_qwen_reasons")  # unoccluded results (deployed adapters)
DET = "/scratch/wuhungma/models/rtdetr_r50vd_coco_o365"
FILL = (96, 78, 62)  # brown-grey "furniture"
TAU = 0.5


def parse(text):
    """The checker's JSON answer -> dict ({} if unreadable)."""
    try:
        return json.loads(text[text.index("{"):text.rindex("}") + 1])
    except Exception:  # noqa: BLE001
        return {}


def stage_mask(a):
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, RTDetrForObjectDetection
    proc = AutoProcessor.from_pretrained(DET)
    det = RTDetrForObjectDetection.from_pretrained(DET).to(a.device).eval()
    person = int([k for k, v in det.config.id2label.items() if v == "person"][0])

    def boxes(imgs):
        with torch.inference_mode():
            out = det(**proc(images=imgs, return_tensors="pt").to(a.device))
        res = proc.post_process_object_detection(out, target_sizes=torch.tensor([im.size[::-1] for im in imgs]),
                                                 threshold=a.det_threshold)
        best = []
        for r in res:
            m = r["labels"] == person
            best.append(r["boxes"][m][r["scores"][m].argmax()].tolist() if m.any() else None)
        return best

    clips = [json.loads(l) for l in open(SRC / "clips_mv.jsonl")][:a.limit or None]
    (a.out / "previews").mkdir(parents=True, exist_ok=True)
    for name in ("clips_mv.jsonl", "events_mv.jsonl"):
        (a.out / name).write_text((SRC / name).read_text())
    log = []
    for n, c in enumerate(clips, 1):
        cid = c["clip_id"]
        imgs = [Image.open(SRC / "clips" / cid / f"f{i:02d}.jpg").convert("RGB") for i in range(32)]
        bx = boxes(imgs)
        late = [b for b in bx if b][-a.last:]  # where the person ends up
        rec = {"clip_id": cid, "person_frames": sum(b is not None for b in bx)}
        d = a.out / "clips" / cid
        d.mkdir(parents=True, exist_ok=True)
        if late:
            W, H = imgs[0].size
            x0, y0 = min(b[0] for b in late), min(b[1] for b in late)
            x1, y1 = max(b[2] for b in late), max(b[3] for b in late)
            px, py = a.pad * (x1 - x0), a.pad * (y1 - y0)
            occ = [max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)]
            rec["occluder"] = [round(v, 1) for v in occ]
            rec["occluded_area"] = round((occ[2] - occ[0]) * (occ[3] - occ[1]) / (W * H), 3)
            for im in imgs:
                ImageDraw.Draw(im).rectangle(occ, fill=FILL)
            rec["person_frames_after"] = sum(b is not None for b in boxes(imgs))
        for i, im in enumerate(imgs):
            im.save(d / f"f{i:02d}.jpg", quality=95)
        if n <= 12:  # first, middle and last frame side by side
            strip = Image.new("RGB", (imgs[0].width * 3, imgs[0].height))
            for j, i in enumerate((0, 16, 31)):
                strip.paste(imgs[i], (j * imgs[0].width, 0))
            strip.save(a.out / "previews" / f"{cid}.jpg", quality=85)
        log.append(rec)
        if n % 25 == 0 or n == len(clips):
            print(f"  masked {n}/{len(clips)}", flush=True)
    (a.out / "masks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))


def load_occluded(a):
    def res(p):
        return {r["clip_id"]: r for r in map(json.loads, open(p)) if "p_fall" in r}
    return {"G": res(a.out / "gemma_lora_clips_mv.jsonl"), "Q": res(a.out / "qwen" / "gemma_lora_clips_mv.jsonl")}


def stage_round2(a):
    import torch
    from PIL import Image
    sys.path[:0] = [P.SPACE]
    import gemma_binary_scoring as gs
    P.Q_DIR = Q_ORIG
    events, r1 = P.load_round1()  # unoccluded round-1 results of the other cameras
    out_file = a.out / "round2.jsonl"
    done = {(r["event_id"], r["cam"], r["checker"]) for r in map(json.loads, open(out_file))} if out_file.exists() else set()
    for m in ("G", "Q"):
        todo = [(e, cam) for e in events for cam in sorted(e["clips"]) if (e["event_id"], cam, m) not in done]
        print(f"[{P.CHECKERS[m]}] round 2 on blinded cameras: {len(todo)} to do", flush=True)
        if not todo:
            continue
        p_fall, model = P.make_scorer(m, a)
        with open(out_file, "a") as f:
            for n, (e, cam) in enumerate(todo, 1):
                cid = e["clips"][cam]
                frames = [Image.open(a.out / "clips" / cid / f"f{i:02d}.jpg").convert("RGB") for i in range(32)]
                p, mass = p_fall(P.build_content(gs, frames, P.peer_notes(e, r1, cam)))
                f.write(json.dumps({"event_id": e["event_id"], "cam": cam, "checker": m, "clip_id": cid,
                                    "p2": round(float(p), 5), "mass": round(float(mass), 4)}) + "\n")
                f.flush()
                if n % 25 == 0 or n == len(todo):
                    print(f"  {n}/{len(todo)}", flush=True)
        del model
        torch.cuda.empty_cache()


def stage_report(a):
    P.Q_DIR = Q_ORIG
    events, r1 = P.load_round1()
    occ = load_occluded(a)
    r2 = {(r["event_id"], r["cam"], r["checker"]): r["p2"] for r in map(json.loads, open(a.out / "round2.jsonl"))} \
        if (a.out / "round2.jsonl").exists() else {}
    masks = {r["clip_id"]: r for r in map(json.loads, open(a.out / "masks.jsonl"))}
    L = []

    # how blind is the blinded camera, and what do its checkers say?
    ms = list(masks.values())
    vis = [r["person_frames_after"] / max(1, r["person_frames"]) for r in ms if "occluder" in r]
    L.append(f"Masking: {sum('occluder' in r for r in ms)}/{len(ms)} clips had a person to hide; after masking a person "
             f"is still detected in {100 * sum(vis) / len(vis):.0f}% of the frames where one was before (partial "
             f"occlusion), mean occluded area {100 * sum(r['occluded_area'] for r in ms if 'occluder' in r) / len(vis):.0f}% of the frame.")
    kinds = {cid: e["kind"] for e in events for cid in e["clips"].values()}
    L.append("\nBlinded camera's own checker (no exchange), per clip:")
    L.append(f"{'checker':7} {'event':12} {'n':>4} {'says fall':>10} {'mean P(fall)':>13} {'occlusion_flag':>15} | unoccluded: says fall")
    for m in ("G", "Q"):
        for kind in ("hit", "fa"):
            ids = [cid for cid in occ[m] if kinds.get(cid) == kind]
            o = [occ[m][c] for c in ids]
            fl = sum(parse(r["text"]).get("occlusion_flag") is True for r in o)
            L.append(f"{P.m_name(m):7} {'real fall' if kind == 'hit' else 'false alarm':12} {len(o):4d} "
                     f"{sum(r['p_fall'] >= TAU for r in o):10d} {sum(r['p_fall'] for r in o) / len(o):13.2f} {fl:15d} | "
                     f"{sum(r1[m][c]['p_fall'] >= TAU for c in ids):d}")

    # room decisions per (event, blinded camera) scenario
    rules = ["no occlusion (reference)", "today: two agree + majority", "escalate + majority",
             "escalate + abstain if flagged", "escalate + exchange (no->yes)", "escalate + abstain + exchange",
             "escalate + blind camera ignored (oracle)"]
    rows, n_sc = Counter(), Counter()
    for e in events:
        kind = "fall" if e["kind"] == "hit" else "false alarm"
        cams = sorted(e["clips"])
        for X in cams:
            pairs = [(c, m) for c in cams for m in ("G", "Q")]
            if any(e["clips"][X] not in occ[m] for m in ("G", "Q")):
                continue
            trig = set(e["triggered_by"]) - {X}
            n_sc[(e["split"], kind)] += 1

            def vote(c, m, round2=False):
                if c != X:
                    return r1[m][e["clips"][c]]["p_fall"] >= TAU
                v = occ[m][e["clips"][c]]["p_fall"] >= TAU
                return v or (round2 and r2.get((e["event_id"], c, m), 0) >= TAU and any(
                    r1[mm][e["clips"][cc]]["p_fall"] >= TAU for cc in cams if cc != X for mm in ("G", "Q")))

            def flagged(c, m):
                return c == X and parse(occ[m][e["clips"][c]]["text"]).get("occlusion_flag") is True

            def keeps(votes):  # app decide(): >= half say fall; nobody left to vote -> keep (fail safe)
                return True if not votes else P.room_keeps(votes)

            ref = [r1[m][e["clips"][c]]["p_fall"] >= TAU for c, m in pairs]
            dec = {
                rules[0]: keeps(ref),
                rules[1]: len(trig) >= 2 and keeps([vote(c, m) for c, m in pairs]),
                rules[2]: bool(trig) and keeps([vote(c, m) for c, m in pairs]),
                rules[3]: bool(trig) and keeps([vote(c, m) for c, m in pairs if not flagged(c, m)]),
                rules[4]: bool(trig) and keeps([vote(c, m, True) for c, m in pairs]),
                rules[5]: bool(trig) and keeps([vote(c, m, True) for c, m in pairs if not flagged(c, m)]),
                rules[6]: bool(trig) and keeps([vote(c, m) for c, m in pairs if c != X]),
            }
            for r, v in dec.items():
                rows[(e["split"], kind, r)] += v
    L.append("\nRoom decisions; one scenario = one event with one of its cameras blinded (each camera in turn).")
    L.append("'escalate' = the event fires if any OTHER camera's VideoMAE alarmed; checkers vote, alarm kept if >= half say fall.")
    L.append(f"{'split':5} {'rule':42} {'falls kept':>12} {'false alarms kept':>18}")
    for split in ("val", "test"):
        for r in rules:
            nf, na = n_sc[(split, "fall")], n_sc[(split, "false alarm")]
            L.append(f"{split:5} {r:42} {rows[(split, 'fall', r)]:>6}/{nf:<5} {rows[(split, 'false alarm', r)]:>11}/{na}")
    text = "\n".join(L)
    (a.out / "report.txt").write_text(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["mask", "round2", "report"])
    ap.add_argument("--out", type=Path, default=Path("/scratch/wuhungma/occlusion"))
    ap.add_argument("--pad", type=float, default=0.15, help="occluder padding, fraction of the box size per side")
    ap.add_argument("--det-threshold", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--last", type=int, default=4, help="hide where the person is in the last N frames with a person")
    ap.add_argument("--limit", type=int, default=0, help="mask only the first N clips (testing)")
    ap.add_argument("--gemma-base", default="/scratch/wuhungma/models/gemma-3-12b-it")
    ap.add_argument("--gemma-adapter", default="/scratch/wuhungma/lora/r16_reasons_ep1/adapter")
    ap.add_argument("--qwen-base", default="/scratch/wuhungma/models/qwen2-vl-7b-instruct")
    ap.add_argument("--qwen-adapter", default="/scratch/wuhungma/lora/qwen_r16_reasons/adapter")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    {"mask": stage_mask, "round2": stage_round2, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main()
