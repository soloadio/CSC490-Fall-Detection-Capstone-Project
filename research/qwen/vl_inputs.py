"""Build model inputs for one 32-frame clip, as separate images or as one video (Qwen2-VL / Qwen3-VL processors).

    inp = build(processor, frames, prompt, mode="video")   # frames: PIL images in time order, 8 per second
"""
FPS = 8
MAX_PIXELS = 345600  # per frame: up_fall 640x480 and mcfd 720x480 at full size


def build(proc, frames, prompt, mode="video", max_pixels=MAX_PIXELS):
    if mode == "images":
        proc.image_processor.size["longest_edge"] = max_pixels
        content = [{"type": "text", "text": prompt}]
        for i, _ in enumerate(frames):
            content += [{"type": "text", "text": f"Frame {i + 1}:"}, {"type": "image"}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                        tokenize=False)
        return proc(text=[text], images=list(frames), return_tensors="pt")
    from transformers.video_utils import VideoMetadata
    proc.video_processor.size["longest_edge"] = max_pixels * len(frames)  # the video processor caps the whole clip
    content = [{"type": "video"}, {"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)
    meta = VideoMetadata(total_num_frames=len(frames), fps=FPS, width=frames[0].width, height=frames[0].height,
                         duration=len(frames) / FPS, frames_indices=list(range(len(frames))))
    return proc(text=[text], videos=[list(frames)], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
