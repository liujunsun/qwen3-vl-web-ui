# Copyright (c) Alibaba Cloud.
#
# Message construction + streaming text generation for the HF backend. The core of
# this logic is ported from the reference Gradio demo
# (Qwen3-VL/web_demo_mm.py, functions call_local_model / _transform_messages /
# _remove_image_special / _is_video_file), minus the Gradio-specific HTML munging.
#
# One deliberate divergence from the reference demo: video frames are sampled by
# app.video.sample_frames and handed to the processor with `do_sample_frames=False`,
# instead of letting transformers decode the whole file. That means building the model
# inputs in two steps (chat template -> text, then processor(text, images, videos))
# rather than one `apply_chat_template(tokenize=True)` call. The two paths were verified
# to produce byte-identical input_ids / pixel values for text-only, image, video,
# image+video, multi-video and multi-turn conversations - see README.
import os
import queue
import re
from dataclasses import replace
from threading import Thread
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import TextIteratorStreamer
from transformers.image_utils import load_image

from .model_runtime import ModelRuntime
from .runtime_settings import GenerationDefaults
from .schemas import ChatMessage, GenerationOptions
from .video import sample_frames

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg")

# How long to wait for the next streamed token before giving up on the generation.
STREAM_TIMEOUT_S = float(os.environ.get("QWEN_STREAM_TIMEOUT", "600"))


def is_video_file(filename: str) -> bool:
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def _remove_image_special(text: str) -> str:
    text = text.replace("<ref>", "").replace("</ref>", "")
    return re.sub(r"<box>.*?(</box>|$)", "", text)


def resolve_options(
    defaults: GenerationDefaults, options: Optional[GenerationOptions], legacy_max_new_tokens: Optional[int] = None
) -> GenerationDefaults:
    """Layer per-request overrides on top of the server defaults.

    Precedence: request options > legacy top-level max_new_tokens > server defaults.
    """
    patch = options.model_dump(exclude_none=True) if options else {}
    if "max_new_tokens" not in patch and legacy_max_new_tokens:
        patch["max_new_tokens"] = legacy_max_new_tokens
    if not patch:
        return defaults
    return replace(defaults, **patch)


def build_messages(
    messages: List[ChatMessage],
    resolve_path: Callable[[str], str],
) -> List[Dict]:
    """Turn the API's message list into the format transformers' chat template expects.

    `resolve_path` maps an upload id to an absolute file path on disk.
    """
    out: List[Dict] = []
    for msg in messages:
        content = []
        for part in msg.content:
            if part.type == "text":
                content.append({"type": "text", "text": part.text or ""})
            elif part.type == "image":
                content.append({"type": "image", "image": resolve_path(part.id)})
            elif part.type == "video":
                content.append({"type": "video", "video": resolve_path(part.id)})
        out.append({"role": msg.role, "content": content})
    return out


def prepare_inputs(
    runtime: ModelRuntime, messages: List[Dict], video_fps: float, video_max_frames: int
) -> Tuple[Dict, Dict]:
    """Build model inputs, sampling any video ourselves.

    Returns (inputs, stats) where stats describes what the model will actually see.
    """
    processor = runtime.processor

    images, videos, metadata = [], [], []
    # Count frames here, not from the metadata afterwards: Qwen3VLProcessor's
    # _calculate_timestamps pads `frames_indices` in place (it extends the very list we
    # pass in) when the count is odd, so reading it back would over-report by one.
    frame_counts = []
    for msg in messages:
        for part in msg["content"]:
            if part.get("type") == "image":
                images.append(load_image(part["image"]))
            elif part.get("type") == "video":
                frames, meta = sample_frames(part["video"], video_fps, video_max_frames)
                videos.append(frames)
                metadata.append(meta)
                frame_counts.append(int(frames.shape[0]))

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    kwargs: Dict = {}
    if images:
        kwargs["images"] = images
    if videos:
        # `video_metadata` carries the ORIGINAL frame indices, which is what Qwen3-VL
        # turns into the "<12.5 seconds>" markers in the prompt. Do not pass
        # `return_metadata` here: the processor only pops metadata out of its output
        # when the caller did not ask for it, and a non-tensor value in the batch would
        # break the .to(device) below.
        kwargs.update(videos=videos, video_metadata=metadata, do_sample_frames=False)

    inputs = processor(text=[text], return_tensors="pt", **kwargs)
    stats = {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "images": len(images),
        "videos": len(videos),
        "frames": frame_counts,
    }
    return inputs, stats


def stream_generate(
    runtime: ModelRuntime,
    messages: List[Dict],
    opts: GenerationDefaults,
    on_stats: Optional[Callable[[Dict], None]] = None,
) -> Iterator[str]:
    """Yield the growing generated string as tokens arrive."""
    model = runtime.model
    processor = runtime.processor

    # Decoding/sampling happens before a single token is generated, so a broken upload
    # surfaces here. Report it in-band - the SSE stream has already started.
    try:
        inputs, stats = prepare_inputs(runtime, messages, opts.video_fps, opts.video_max_frames)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:  # noqa: BLE001
        runtime.gc()
        yield f"**[Error preparing input: {e}]**"
        return
    if on_stats:
        on_stats(stats)

    # The timeout is the gap allowed *before the first token* as well as between tokens,
    # and prefill on a long video is genuinely slow: at the Qwen3-VL default cap of 768
    # frames a 3-minute clip is ~13k visual tokens, which can take minutes on a busy GPU.
    # 60s was low enough to abort legitimate work, so allow well past the worst case.
    streamer = TextIteratorStreamer(
        processor.tokenizer, timeout=STREAM_TIMEOUT_S, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=opts.max_new_tokens,
        streamer=streamer,
        use_cache=True,
    )
    # temperature 0 means greedy. Passing sampling params alongside do_sample=False makes
    # transformers warn, so only send them when sampling is actually on.
    if opts.temperature and opts.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=opts.temperature, top_p=opts.top_p)
        if opts.top_k:
            gen_kwargs["top_k"] = opts.top_k
    else:
        gen_kwargs["do_sample"] = False

    generation_error: Dict[str, BaseException] = {}

    def _generate():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except BaseException as e:  # noqa: BLE001 - surfaced to the client below
            generation_error["error"] = e
            streamer.end()

    thread = Thread(target=_generate, daemon=True)
    thread.start()

    generated = ""
    # Iterating the streamer raises queue.Empty on timeout. Left unguarded that escapes
    # mid-SSE and the client just sees a truncated stream with no explanation.
    try:
        for new_text in streamer:
            generated += new_text
            yield _remove_image_special(generated)
    except queue.Empty:
        runtime.gc()
        generated += (
            f"\n\n**[Error: no output for {STREAM_TIMEOUT_S:.0f}s - giving up. The GPU may be "
            f"contended or the input too large. Try fewer frames (Settings -> Video max "
            f"frames), or raise QWEN_STREAM_TIMEOUT.]**"
        )
        yield _remove_image_special(generated)
        return

    thread.join(timeout=STREAM_TIMEOUT_S)

    if "error" in generation_error:
        err = generation_error["error"]
        runtime.gc()
        if isinstance(err, torch.cuda.OutOfMemoryError):
            generated += (
                "\n\n**[Error: ran out of GPU memory processing this input. "
                "Try a shorter video, fewer frames (Settings -> Video max frames), "
                "or a shorter prompt. GPU memory has been freed for the next attempt.]**"
            )
        else:
            generated += f"\n\n**[Error during generation: {err}]**"
        yield _remove_image_special(generated)
