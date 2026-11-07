"""Two-class version of gemma_scoring: fall (falling or fallen) vs no_fall.

Same interface as gemma_scoring so run_probs.py / run_multiview.py can use it via --scoring gemma_binary_scoring.
Gold mapping for evaluation: fall = OmniFall labels {fall, fallen}; no_fall = the other 8 labels.
"""
from PIL import Image
from gemma_scoring import score_labels, to_probs  # noqa: F401  (re-exported)

CLASSES = ["fall", "no_fall"]
PREFIX = '{"verdict": "'
POSITIVE_GOLD = {"fall", "fallen"}

DEFS = """- fall: The person is falling, or is on the ground or a mattress as a result of a fall (they did not get there through a controlled, intentional movement). Includes falling onto a bed, if the movement is not a controlled lying down with arms as support.
- no_fall: Anything else: walking, standing, sitting down or sitting, intentionally lying down or lying in a bed or on the ground after getting there on purpose, getting up (standing up), or any other activity."""

ANSWER = """Answer with a single JSON object and nothing else:
{"verdict": "<fall or no_fall>", "evidence": "<one short sentence about what you see>", "occlusion_flag": <true if the person is hidden or partly hidden, else false>}"""

def instruction(k):
    return f"""You are watching {k} consecutive frames, in time order, of one person filmed by a single camera.
Decide whether this person is falling or has fallen. Choose exactly one label:
{DEFS}

{ANSWER}"""

def instruction_mv(n_cams, k):
    return f"""You are watching one person filmed at about the same time by {n_cams} different cameras. For each camera you get {k} frames in time order.
The cameras show the same moment from different viewpoints; the person may be hidden in some views.
Decide whether this person is falling or has fallen. Choose exactly one label:
{DEFS}

{ANSWER}"""

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
        full = tok.encode(PREFIX + lab, add_special_tokens=False)
        if full[:len(pre)] != pre:
            bad.append(lab); cont = tok.encode(lab, add_special_tokens=False)
        else:
            cont = full[len(pre):]
        plan[lab] = cont
    return pre, plan, bad
