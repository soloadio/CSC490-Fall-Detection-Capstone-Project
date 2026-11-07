"""Label probabilities from Gemma 3 instead of a verbalized "confidence".

The assistant turn is forced to start with  {"verdict": "  and we compute, for each of the 10 labels,
log P(label tokens | prompt + forced prefix) with one prefill pass plus short teacher-forced
continuations on a copy of the KV cache. Probabilities are normalized over the 10 labels;
`mass` is the unnormalized total, i.e. how much probability the model puts on the valid labels at all.
"""
import copy, math, torch
from PIL import Image

CLASSES = ["walk", "fall", "fallen", "sit_down", "sitting", "lie_down", "lying", "stand_up", "standing", "other"]
PREFIX = '{"verdict": "'
DEFS = """- walk: Move around, including jogging and running, but not as part of a special exercise; includes carrying something small.
- fall: The act of falling (from any previous state). Includes falling on a bed, if the process is not a controlled lying down with arms as support.
- fallen: Being on the ground or a mattress after a fall.
- sit_down: Sitting down on a bed, chair or the ground.
- sitting: Sitting on a bed, chair or the ground.
- lie_down: Lying down intentionally (in contrast to a fall) on the ground or a bed.
- lying: Being in a lying position (in bed or on the ground) after intentionally getting into that position.
- stand_up: Standing up from a fallen state, from lying or sitting.
- standing: Standing around without walking.
- other: Any other activity, including walking while pushing an object like a chair."""

def instruction(k):
    return f"""You are watching {k} consecutive frames, in time order, of one person filmed by a single camera.
Decide which ONE activity the person is doing during this moment. Choose exactly one label:
{DEFS}

Answer with a single JSON object and nothing else:
{{"verdict": "<one label>", "evidence": "<one short sentence about what you see>", "occlusion_flag": <true if the person is hidden or partly hidden, else false>}}"""

def messages_for(frame_paths):
    content = [{"type": "text", "text": instruction(len(frame_paths))}]
    n = len(frame_paths)
    for i, p in enumerate(frame_paths):
        tag = "earliest" if i == 0 else "latest" if i == n - 1 else "later"
        content += [{"type": "text", "text": f"Frame {i+1} ({tag}):"}, {"type": "image", "image": Image.open(p).convert("RGB")}]
    return [{"role": "user", "content": content}]

def label_plan(tok):
    """Token ids of the forced prefix and of each label's continuation (label tokens only)."""
    pre = tok.encode(PREFIX, add_special_tokens=False)
    plan, bad = {}, []
    for lab in CLASSES:
        full = tok.encode(PREFIX + lab, add_special_tokens=False)   # no closing quote: the model's own JSON
        if full[:len(pre)] != pre:                                  # continues with `",` (one merged token)
            bad.append(lab); cont = tok.encode(lab, add_special_tokens=False)
        else:
            cont = full[len(pre):]
        plan[lab] = cont
    return pre, plan, bad

@torch.inference_mode()
def score_labels(model, inputs, pre, plan):
    dev = inputs["input_ids"].device
    ids = torch.cat([inputs["input_ids"], torch.tensor([pre], device=dev)], dim=1)
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=inputs["pixel_values"],
              use_cache=True, logits_to_keep=1)
    if "token_type_ids" in inputs:
        kw["token_type_ids"] = torch.cat([inputs["token_type_ids"], torch.zeros(1, len(pre), dtype=inputs["token_type_ids"].dtype, device=dev)], dim=1)
    out = model(**kw)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    cache, L = out.past_key_values, ids.shape[1]
    logp = {}
    for lab, cont in plan.items():
        lp = first[cont[0]].item()
        if len(cont) > 1:
            c = copy.deepcopy(cache)
            o = model(input_ids=torch.tensor([cont[:-1]], device=dev),
                      attention_mask=torch.ones(1, L + len(cont) - 1, dtype=torch.long, device=dev),
                      past_key_values=c, use_cache=True)
            lps = torch.log_softmax(o.logits[0].float(), dim=-1)
            lp += sum(lps[j - 1, cont[j]].item() for j in range(1, len(cont)))
            del c, o
        logp[lab] = lp
    return logp

def to_probs(logp):
    m = max(logp.values()); z = sum(math.exp(v - m) for v in logp.values())
    probs = {k: math.exp(v - m) / z for k, v in logp.items()}
    mass = sum(math.exp(v) for v in logp.values())
    ent = -sum(p * math.log(p) for p in probs.values() if p > 0)
    return probs, mass, ent
