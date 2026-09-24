# Copyright (c) Alibaba Cloud.
#
# Message construction + streaming text generation for the HF backend. The core of
# this logic is ported from the reference Gradio demo
# (Qwen3-VL/web_demo_mm.py, functions call_local_model / _transform_messages /
# _remove_image_special / _is_video_file), minus the Gradio-specific HTML munging.
#
# One deliberate divergence from the reference demo: video frames are sampled by
# app.video.sample_frames_window and handed to the processor with `do_sample_frames=False`,
# instead of letting transformers decode the whole file. That means building the model
# inputs in two steps (chat template -> text, then processor(text, images, videos))
# rather than one `apply_chat_template(tokenize=True)` call. The two paths were verified
# to produce byte-identical input_ids / pixel values for text-only, image, video,
# image+video, multi-video and multi-turn conversations - see README.
import copy
import os
import queue
import re
import time
from dataclasses import replace
from threading import Thread
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from transformers import TextIteratorStreamer
from transformers.image_utils import load_image

from .model_runtime import ModelRuntime
from .runtime_settings import GenerationDefaults
from .schemas import ChatMessage, GenerationOptions
from .video import motion_score, sample_frames_window

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

    `resolve_path` maps an upload id to an absolute file path on disk. Video is not a
    valid ContentPart type here - every video question goes through the chunked
    /api/chat/segmented path instead, which builds its own messages directly.
    """
    out: List[Dict] = []
    for msg in messages:
        content = []
        for part in msg.content:
            if part.type == "text":
                content.append({"type": "text", "text": part.text or ""})
            elif part.type == "image":
                content.append({"type": "image", "image": resolve_path(part.id)})
        out.append({"role": msg.role, "content": content})
    return out


def prepare_inputs(
    runtime: ModelRuntime,
    messages: List[Dict],
    window: Optional[Dict] = None,
    sample_fps: Optional[float] = None,
    max_video_tokens: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Build model inputs, sampling any video ourselves.

    Returns (inputs, stats) where stats describes what the model will actually see.

    Every video is analysed chunk-by-chunk (see /api/chat/segmented), so any video part
    in `messages` requires `window` (one time span: {"start": s, "end": s, "lead": s}) -
    a chunked request only ever carries one video part. `messages` built from /api/chat
    never contains a video part (ContentPart has no "video" type), so `window` stays
    unused there. Each chunk sizes its own frame cap / resolution from its own duration
    (video.optimize_video_params); `sample_fps` pins the fps used in that sizing to the
    same constant for every chunk instead of scaling it off the chunk's length.
    """
    media = sample_media(messages, window=window, sample_fps=sample_fps)
    return build_inputs(runtime, messages, media, max_video_tokens=max_video_tokens)


def sample_media(
    messages: List[Dict],
    window: Optional[Dict] = None,
    sample_fps: Optional[float] = None,
) -> Dict:
    """Decode the images / video frames `messages` refers to (CPU only, no processor).

    Split from `build_inputs` so a chunk can be checked for motion before any tokens
    are built, and rebuilt at a smaller token budget after an OOM without decoding the
    video a second time.
    """
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
                frames, meta = sample_frames_window(
                    part["video"],
                    2.0,
                    768,
                    window["start"],
                    window["end"],
                    window.get("lead", 0.0),
                    optimize=True,
                    fixed_fps=sample_fps,
                )
                videos.append(frames)
                metadata.append(meta)
                frame_counts.append(int(frames.shape[0]))
    return {"images": images, "videos": videos, "metadata": metadata, "frame_counts": frame_counts}


def build_inputs(
    runtime: ModelRuntime,
    messages: List[Dict],
    media: Dict,
    max_video_tokens: Optional[int] = None,
) -> Tuple[Dict, Dict]:
    """Turn already-sampled media into processor inputs. Returns (inputs, stats)."""
    processor = runtime.processor
    images, videos = media["images"], media["videos"]

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
        # All video options go in one `videos_kwargs` dict: when it is present the
        # processor ignores top-level video kwargs, which would silently re-enable its
        # own frame sampling. The processor pads `frames_indices` in place, so hand it
        # copies - a rebuild at a smaller budget must see the original metadata.
        videos_kwargs: Dict = {
            "video_metadata": [copy.deepcopy(m) for m in media["metadata"]],
            "do_sample_frames": False,
        }
        if max_video_tokens:
            # One visual token = a 32x32 pixel block across 2 frames (16px patch, 2x2
            # merge, temporal patch 2), so the processor's pixel budget is tokens * 2048.
            videos_kwargs["size"] = {
                "longest_edge": int(max_video_tokens) * 2048,
                "shortest_edge": processor.video_processor.size["shortest_edge"],
            }
        kwargs.update(videos=videos, videos_kwargs=videos_kwargs)

    inputs = processor(text=[text], return_tensors="pt", **kwargs)
    frame_size = None
    if videos:
        # What the model actually saw, after the processor fit the frames to the token
        # budget: the patch grid times the patch size.
        _, gh, gw = (int(x) for x in inputs["video_grid_thw"][0])
        patch = int(getattr(processor.video_processor, "patch_size", 16))
        frame_size = [gw * patch, gh * patch]
    stats = {
        "prompt_tokens": int(inputs["input_ids"].shape[-1]),
        "images": len(images),
        "videos": len(videos),
        "frames": media["frame_counts"],
        "frame_size": frame_size,
    }
    if videos:
        stats["motion"] = round(motion_score(videos[0]), 2)
    return inputs, stats


# --- GPU failure handling ------------------------------------------------------------
#
# Two very different failures look alike from the outside:
#   * out of memory - the allocation failed, but the CUDA context is fine. Dropping
#     every reference to the failed attempt's tensors and emptying the cache fully
#     recovers; nothing needs restarting.
#   * a CUDA fault (illegal memory access, device-side assert, driver reset...) - the
#     context is poisoned and every later CUDA call in this process fails too. The only
#     fix is a fresh process, which run.py's supervisor provides.


class GpuOutOfMemory(Exception):
    """Generation ran out of VRAM; the attempt's memory has already been released."""


class GpuFault(Exception):
    """Unrecoverable CUDA error; the process must be restarted."""


_OOM_MARKERS = ("out of memory", "cublas_status_alloc_failed")
_FAULT_MARKERS = (
    "cuda error", "illegal memory access", "device-side assert", "unspecified launch failure",
    "cublas_status", "cudnn_status", "acceleratorerror", "cuda driver",
)


def is_oom(err: BaseException) -> bool:
    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    return any(m in str(err).lower() for m in _OOM_MARKERS)


def is_gpu_fault(err: BaseException) -> bool:
    if is_oom(err):
        return False
    msg = f"{type(err).__name__}: {err}".lower()
    return any(m in msg for m in _FAULT_MARKERS)


def stream_generate(
    runtime: ModelRuntime,
    messages: List[Dict],
    opts: GenerationDefaults,
    on_stats: Optional[Callable[[Dict], None]] = None,
    window: Optional[Dict] = None,
    prepared: Optional[Tuple[Dict, Dict]] = None,
    seed: Optional[int] = None,
    raise_gpu_errors: bool = False,
) -> Iterator[str]:
    """Yield the growing generated string as tokens arrive.

    `window` restricts video sampling to one time span (segmented analysis); see
    `prepare_inputs`. `prepared` skips that step with (inputs, stats) built by the
    caller. `seed` fixes the sampling RNG so the same inputs always give the same text
    - which is what lets an OOM retry reproduce the answer exactly. With
    `raise_gpu_errors`, an OOM raises GpuOutOfMemory (after freeing the attempt's
    memory) and a CUDA fault raises GpuFault, instead of being reported in-band.
    """
    model = runtime.model
    processor = runtime.processor

    # Decoding/sampling happens before a single token is generated, so a broken upload
    # surfaces here. Report it in-band - the SSE stream has already started.
    try:
        if prepared is None:
            prepared = prepare_inputs(
                runtime, messages, window=window, sample_fps=opts.sample_fps,
                max_video_tokens=opts.max_video_tokens,
            )
        inputs, stats = prepared
        prepared = None
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    except Exception as e:  # noqa: BLE001
        oom, fault, msg = is_oom(e), is_gpu_fault(e), str(e)
        e.__traceback__ = None  # the traceback pins the frame's tensors
        inputs = None
        if fault:
            runtime.faulted = True
        runtime.gc()
        if raise_gpu_errors and oom:
            raise GpuOutOfMemory(msg)
        if raise_gpu_errors and fault:
            raise GpuFault(msg)
        yield f"**[Error preparing input: {msg}]**"
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
    # No sampling params here: leaving do_sample/temperature/top_p/top_k unset makes
    # generate() fall back to the checkpoint's own generation_config, exactly as loaded
    # from the model - there is no settings-page override for it.
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=opts.max_new_tokens,
        streamer=streamer,
        use_cache=True,
    )
    inputs = None

    generation_error: Dict[str, BaseException] = {}

    def _generate():
        try:
            if seed is not None:
                # Only one generation runs at a time (the server's lock), so seeding
                # the global generators here is race-free.
                torch.manual_seed(seed)
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
            f"contended or the input too large. Try a shorter chunk length (Settings -> "
            f"Chunk length), or raise QWEN_STREAM_TIMEOUT.]**"
        )
        yield _remove_image_special(generated)
        return

    thread.join(timeout=STREAM_TIMEOUT_S)

    if "error" in generation_error:
        err = generation_error.pop("error")
        oom, fault, msg = is_oom(err), is_gpu_fault(err), str(err)
        # The exception's traceback holds generate()'s frames and with them the KV cache
        # and activations - release it, and the inputs, before emptying the cache, or
        # the "freed" memory is still pinned.
        err.__traceback__ = None
        err = None
        gen_kwargs.clear()
        if fault:
            runtime.faulted = True
        runtime.gc()
        if raise_gpu_errors and oom:
            raise GpuOutOfMemory(msg)
        if raise_gpu_errors and fault:
            raise GpuFault(msg)
        if oom:
            generated += (
                "\n\n**[Error: ran out of GPU memory processing this input. "
                "Try a shorter chunk length (Settings -> Chunk length) "
                "or a shorter prompt. GPU memory has been freed for the next attempt.]**"
            )
        elif fault:
            generated += f"\n\n**[GPU fault: {msg}. The server will restart itself to recover.]**"
        else:
            generated += f"\n\n**[Error during generation: {msg}]**"
        yield _remove_image_special(generated)


# --- one chunk of a segmented run, with motion skip + OOM recovery --------------------

# Pause before retrying after an OOM, so a transient spike from another program (a game
# loading, a browser tab decoding video) has a moment to pass.
OOM_RETRY_WAIT_S = 2.0
# Never shrink the video below this many tokens while recovering - past this point the
# frames are too small to be worth answering from.
MIN_RECOVERY_VIDEO_TOKENS = 512


def recovery_budgets(max_video_tokens: int) -> List[int]:
    """Token budgets to try in order: the requested one twice, then 1/2, then 1/4.

    The second try at the same budget, with the same seed, reproduces exactly the
    answer the first try would have given - most OOMs are a transient squeeze, so the
    result is usually unaffected. Only if that fails too is detail traded for memory.
    """
    budgets = [max_video_tokens, max_video_tokens]
    for div in (2, 4):
        b = max(MIN_RECOVERY_VIDEO_TOKENS, max_video_tokens // div)
        if b < budgets[-1]:
            budgets.append(b)
    return budgets


def generate_chunk(
    runtime: ModelRuntime,
    messages: List[Dict],
    opts: GenerationDefaults,
    window: Dict,
    seed: int,
    motion_threshold: float = 0.0,
) -> Iterator[Tuple[str, object]]:
    """Run one chunk of a segmented analysis, yielding (kind, payload) events:

    ("stats", dict)    what the model will see - re-sent if a retry changes it
    ("text", str)      the growing answer; a retry restarts it from empty
    ("skipped", dict)  the chunk had no motion and the model was not run (final)
    ("retry", dict)    an OOM was recovered from; {"attempt", "max_video_tokens", "reason"}
    ("fault", str)     unrecoverable CUDA error; the process needs a restart (final)
    """
    try:
        media = sample_media(messages, window=window, sample_fps=opts.sample_fps)
        budget = int(opts.max_video_tokens)
        inputs, stats = build_inputs(runtime, messages, media, max_video_tokens=budget)
    except Exception as e:  # noqa: BLE001 - a broken upload / window, reported in-band
        yield ("text", f"**[Error preparing input: {e}]**")
        return

    if motion_threshold > 0 and stats.get("motion", 100.0) < motion_threshold:
        yield ("skipped", {**stats, "skipped": True, "motion_threshold": motion_threshold})
        return

    budgets = recovery_budgets(budget)
    last_error = ""
    for attempt, b in enumerate(budgets):
        if attempt:
            time.sleep(OOM_RETRY_WAIT_S)
            yield ("retry", {"attempt": attempt + 1, "max_video_tokens": b, "reason": last_error})
            if b != budget:
                inputs, stats = build_inputs(runtime, messages, media, max_video_tokens=b)
                budget = b
            stats = {**stats, "recovery": {
                "attempt": attempt + 1, "max_video_tokens": b, "requested": budgets[0],
            }}
        yield ("stats", stats)
        try:
            for text in stream_generate(
                runtime, messages, opts, prepared=(inputs, stats), seed=seed, raise_gpu_errors=True
            ):
                yield ("text", text)
            return
        except GpuOutOfMemory as e:
            last_error = str(e).splitlines()[0][:200]
            print(f"[recovery] OOM at {b} video tokens (attempt {attempt + 1}/{len(budgets)}); "
                  f"memory freed", flush=True)
        except GpuFault as e:
            yield ("text", f"**[GPU fault: {e}. The server will restart itself to recover.]**")
            yield ("fault", str(e))
            return

    yield ("text", (
        f"**[Error: ran out of GPU memory {len(budgets)} times, down to {budgets[-1]} video "
        f"tokens. Something else is probably using the GPU - close it, or lower Max video "
        f"tokens / Max new tokens in Settings.]**"
    ))
