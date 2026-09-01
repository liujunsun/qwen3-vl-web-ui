# Copyright (c) Alibaba Cloud.
#
# Message construction + streaming text generation for the HF backend. The core of
# this logic is ported from the reference Gradio demo
# (Qwen3-VL/web_demo_mm.py, functions call_local_model / _transform_messages /
# _remove_image_special / _is_video_file), minus the Gradio-specific HTML munging.
import re
from threading import Thread
from typing import Callable, Dict, Iterator, List

import torch
from transformers import TextIteratorStreamer

from .model_runtime import ModelRuntime
from .schemas import ChatMessage

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".mpeg")


def is_video_file(filename: str) -> bool:
    return filename.lower().endswith(VIDEO_EXTENSIONS)


def _remove_image_special(text: str) -> str:
    text = text.replace("<ref>", "").replace("</ref>", "")
    return re.sub(r"<box>.*?(</box>|$)", "", text)


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


def stream_generate(
    runtime: ModelRuntime,
    messages: List[Dict],
    max_new_tokens: int,
) -> Iterator[str]:
    """Yield the growing generated string as tokens arrive."""
    model = runtime.model
    processor = runtime.processor

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        processor.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        use_cache=True,
    )

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
    for new_text in streamer:
        generated += new_text
        yield _remove_image_special(generated)

    thread.join()

    if "error" in generation_error:
        err = generation_error["error"]
        runtime.gc()
        if isinstance(err, torch.cuda.OutOfMemoryError):
            generated += (
                "\n\n**[Error: ran out of GPU memory processing this input. "
                "Try a shorter video, lower resolution, or a shorter prompt. "
                "GPU memory has been freed for the next attempt.]**"
            )
        else:
            generated += f"\n\n**[Error during generation: {err}]**"
        yield _remove_image_special(generated)
