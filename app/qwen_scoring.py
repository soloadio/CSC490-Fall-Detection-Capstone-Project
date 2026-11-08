"""Qwen2-VL-7B-Instruct + LoRA fall check, the same way as the Gemma checker (gemma_binary_scoring.py).

Same prompt, frame tags and forced answer prefix {"verdict": "  as Gemma; P(fall) is read off the next-token
probabilities, then the answer text is generated for the "evidence" sentence. The Qwen-specific parts (the KV-cache
scoring with explicit M-RoPE positions, mm_token_type_ids padding, --max-pixels 78000 per frame) are taken from
Solomon's research/qwen/run_probs_qwen.py, so the scores match his runs.

    qs = QwenScorer(base, adapter); qs.load(); qs.check(list_of_PIL_frames) -> {p_fall, mass, text}
"""
import copy

import gemma_binary_scoring as gs
from gemma_scoring import to_probs

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
    def __init__(self, base="Qwen/Qwen2-VL-7B-Instruct", adapter=None, max_pixels=MAX_PIXELS, max_new_tokens=100):
        self.base, self.adapter, self.max_pixels, self.max_new_tokens = base, adapter, max_pixels, max_new_tokens

    def load(self, device_map="cuda"):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(self.base)
        self.processor.image_processor.size["longest_edge"] = self.max_pixels
        self.pre, self.plan, bad = gs.label_plan(self.processor.tokenizer)
        assert not bad, bad
        model = Qwen2VLForConditionalGeneration.from_pretrained(self.base, dtype=torch.bfloat16,
                                                                device_map=device_map).eval()
        if self.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload().eval()
        self.model = model
        return self

    def check(self, frames):
        """frames: PIL images in time order (32) -> {p_fall, mass, text}."""
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
