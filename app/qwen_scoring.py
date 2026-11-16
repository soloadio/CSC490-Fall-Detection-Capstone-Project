"""Qwen2-VL-7B-Instruct + LoRA fall check, the same way as the Gemma checker (gemma_binary_scoring.py).

Same prompt, frame tags and forced answer prefix {"verdict": "  as Gemma; P(fall) is read off the next-token
probabilities, then the answer text is generated for the "evidence" sentence. The Qwen-specific parts (the KV-cache
scoring with explicit M-RoPE positions, mm_token_type_ids padding, --max-pixels 78000 per frame) are taken from
Solomon's research/qwen/run_probs_qwen.py, so the scores match his runs.

    qs = QwenScorer(base, adapter); qs.load(); qs.check(list_of_PIL_frames) -> {p_fall, mass, text}

evidence_first=True (adapters trained with research/qwen/train_lora_qwen.py --evidence-first): the answer is
{"evidence": "...", "verdict": "...", "occlusion_flag": ...}. The evidence sentence is generated first (forced prefix
{"evidence": "), then P(fall) is read off the next-token probabilities after  ", "verdict": ", so the verdict depends on
what the model has just described instead of the description justifying a verdict already given.
"""
import copy

import gemma_binary_scoring as gs
from gemma_scoring import to_probs

EF_PREFIX = '{"evidence": "'
VERDICT_KEY = '", "verdict": "'
ANSWER_EF = """Answer with a single JSON object and nothing else:
{"evidence": "<one or two short sentences: the pose in the first frame, between which frames the pose changes, the pose in the last frame and what touches the floor>", "verdict": "<fall or no_fall>", "occlusion_flag": <true if the person is hidden or partly hidden, else false>}"""


def instruction_ef(k):
    return f"""You are watching {k} consecutive frames, in time order, of one person filmed by a single camera.
Decide whether this person is falling or has fallen. Choose exactly one label:
{gs.DEFS}

{ANSWER_EF}"""


def messages(frames, evidence_first=False):
    """Chat messages for one clip: instruction, then 'Frame i (tag):' + image for each frame (PIL images or paths)."""
    from PIL import Image
    content = [{"type": "text", "text": (instruction_ef if evidence_first else gs.instruction)(len(frames))}]
    for i, im in enumerate(frames):
        tag = "earliest" if i == 0 else "latest" if i == len(frames) - 1 else "later"
        im = im if hasattr(im, "size") else Image.open(im).convert("RGB")
        content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
    return [{"role": "user", "content": content}]


def label_plan_ef(tok):
    """Token ids of each label after VERDICT_KEY (checked not to merge with the quote before it)."""
    key = tok.encode(VERDICT_KEY, add_special_tokens=False)
    plan = {}
    for lab in gs.CLASSES:
        full = tok.encode(VERDICT_KEY + lab, add_special_tokens=False)
        assert full[:len(key)] == key, lab
        plan[lab] = full[len(key):]
    return plan


MAX_PIXELS = 78000  # per frame, as in Solomon's runs (edf/mcfd 320x240 unchanged, 640x480 scaled to ~308x224)


def _vision_kw(inputs, n_extra):
    import torch
    kw = dict(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
    for k, v in inputs.items():
        if "token_type_ids" in k:  # needed for M-RoPE; text (0) for tokens appended after the prompt
            kw[k] = torch.cat([v, torch.zeros(v.shape[0], n_extra, dtype=v.dtype, device=v.device)], dim=1)
    return kw


def score_labels(model, inputs, pre, plan):
    """log P(label) after the forced prefix: one prefill, then each label teacher-forced on a copy of the cache."""
    import torch
    with torch.inference_mode():
        dev = inputs["input_ids"].device
        ids = torch.cat([inputs["input_ids"], torch.tensor([pre], device=dev)], dim=1)
        L = ids.shape[1]
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), **_vision_kw(inputs, len(pre)),
                    use_cache=True, logits_to_keep=1)
        first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        cache, logp = out.past_key_values, {}
        delta = model.model.rope_deltas  # M-RoPE offset of the text after the images, set by the prefill
        for lab, cont in plan.items():
            lp = first[cont[0]].item()
            if len(cont) > 1:
                c, n = copy.deepcopy(cache), len(cont) - 1
                pos = (torch.arange(L, L + n, device=dev) + delta.to(dev)).view(1, 1, n).expand(3, 1, n)
                o = model(input_ids=torch.tensor([cont[:-1]], device=dev),
                          attention_mask=torch.ones(1, L + n, dtype=torch.long, device=dev),
                          position_ids=pos, cache_position=torch.arange(L, L + n, device=dev),
                          past_key_values=c, use_cache=True)
                lps = torch.log_softmax(o.logits[0].float(), dim=-1)
                lp += sum(lps[j - 1, cont[j]].item() for j in range(1, len(cont)))
                del c, o
            logp[lab] = lp
        return logp


class QwenScorer:
    def __init__(self, base="Qwen/Qwen2-VL-7B-Instruct", adapter=None, max_pixels=MAX_PIXELS, max_new_tokens=100,
                 evidence_first=False):
        self.base, self.adapter, self.max_pixels, self.max_new_tokens = base, adapter, max_pixels, max_new_tokens
        self.evidence_first = evidence_first

    def load(self, device_map="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.base)
        self.processor.image_processor.size["longest_edge"] = self.max_pixels
        self.pre, self.plan, bad = gs.label_plan(self.processor.tokenizer)
        assert not bad, bad
        self.plan_ef = label_plan_ef(self.processor.tokenizer)
        model = Qwen2VLForConditionalGeneration.from_pretrained(self.base, dtype=torch.bfloat16,
                                                                device_map=device_map).eval()
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload().eval()
        self.model = model
        return self

    def check(self, frames):
        """frames: PIL images in time order (32) -> {p_fall, mass, text}."""
        if self.evidence_first:
            return self.check_ef(frames)
        import torch
        content = [{"type": "text", "text": gs.instruction(len(frames))}]
        for i, im in enumerate(frames):
            tag = "earliest" if i == 0 else "latest" if i == len(frames) - 1 else "later"
            content += [{"type": "text", "text": f"Frame {i + 1} ({tag}):"}, {"type": "image", "image": im}]
        inp = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                 tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        probs, mass, _ = to_probs(score_labels(self.model, inp, self.pre, self.plan))
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self.processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        return {"p_fall": float(probs["fall"]), "mass": float(mass), "text": text}

    def check_ef(self, frames):
        """Evidence first: generate the evidence, then score the verdict after it."""
        import torch
        tok = self.processor.tokenizer
        prompt = self.processor.apply_chat_template(messages(frames, True), add_generation_prompt=True, tokenize=False)

        def enc(text):
            inp = self.processor(text=[text], images=list(frames), return_tensors="pt").to(self.model.device)
            inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
            return inp
        inp = enc(prompt + EF_PREFIX)
        with torch.inference_mode():
            gen = self.model.generate(**inp, max_new_tokens=self.max_new_tokens, do_sample=False)
        cont = self.processor.decode(gen[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
        k = cont.find(VERDICT_KEY)
        evidence = cont[:k] if k >= 0 else cont.split('"')[0]
        pre = tok.encode(EF_PREFIX + evidence + VERDICT_KEY, add_special_tokens=False)
        probs, mass, _ = to_probs(score_labels(self.model, enc(prompt), pre, self.plan_ef))
        return {"p_fall": float(probs["fall"]), "mass": float(mass), "text": EF_PREFIX + cont}
