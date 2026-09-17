# Copyright (c) Alibaba Cloud.
#
# Frame sampling for video inputs.
#
# transformers' default path (`do_sample_frames=True`) hands the file to
# torchvision/torchcodec, which decodes EVERY frame into memory and only then keeps
# the handful we asked for. That is the hard ceiling on how long a clip the server can
# accept: a 3-minute 640x360 clip takes ~218s to decode and materialises 4298 frames
# (~3 GB); cost and memory both grow linearly with duration.
#
# We instead walk the container once and convert only the frames we keep - measured at
# ~4.6s for the same clip (~48x faster), with memory bounded by `max_frames` rather
# than by video length. The sampled frames are then handed to the processor with
# `do_sample_frames=False`.
#
# Qwen3-VL builds "<12.5 seconds>" markers into the prompt from
# `VideoMetadata.frames_indices` + `.fps` (see Qwen3VLProcessor.__call__), so the
# indices we report must be positions in the ORIGINAL video - otherwise the model's
# sense of time is silently wrong.
#
# Every video is analysed chunk-by-chunk (see app.main /api/chat/segmented) - there is
# no whole-clip sampling path. `sample_frames_window` decodes just one time span, with
# `optimize=True` always on so each chunk's frame cap / frame size are derived from
# that chunk's own duration (`optimize_video_params`); fps is the constant, editable
# "Sampling rate" setting instead of scaling per chunk.
from typing import Optional, Tuple

import av
import numpy as np
from transformers.video_utils import VideoMetadata

# ffmpeg's fixed internal time base for container-level timestamps (AV_TIME_BASE).
AV_TIME_BASE = 1_000_000


def _resize_dims(width: int, height: int, longest_edge: int):
    """Target (w, h) to shrink a frame so its longer side is <= `longest_edge`.

    Returns None when nothing should change - the cap is off (`longest_edge` <= 0) or
    the frame is already within it. Never upscales. Both dimensions are rounded to even
    numbers, which libswscale prefers.
    """
    if longest_edge <= 0 or not width or not height:
        return None
    longer = max(width, height)
    if longer <= longest_edge:
        return None
    scale = longest_edge / longer
    return max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2)


def _frame_rgb(frame, resize_to):
    """One decoded frame -> HWC uint8 RGB array, optionally downscaled first."""
    if resize_to is not None:
        frame = frame.reformat(width=resize_to[0], height=resize_to[1], format="rgb24")
    return frame.to_ndarray(format="rgb24")


def optimize_video_params(seconds: float, fixed_fps: Optional[float] = None) -> Tuple[float, int, int]:
    """Pick (fps, max_frames, longest_edge) for a clip (or one chunk) of `seconds`.

    With `fixed_fps` unset, sample densely while the clip is short and it's cheap,
    thin out as it grows so the frame count and the raw-frame buffer both stay
    bounded. Passing `fixed_fps` (the user-editable "Sampling rate" setting) skips
    that duration-based table and uses the same fps for every chunk in a run,
    including a shorter final chunk. Either way, frames shrink once there are enough
    of them that the pixel budget would downscale them anyway. Kept deliberately
    simple and mirrored in frontend/app.js.
    """
    s = max(1.0, float(seconds))
    if fixed_fps:
        fps = float(fixed_fps)
    elif s <= 20:
        fps = 6.0
    elif s <= 60:
        fps = 4.0
    elif s <= 180:
        fps = 2.0
    elif s <= 600:
        fps = 1.0
    elif s <= 1800:
        fps = 0.5
    else:
        fps = 0.25
    max_frames = int(min(1024, max(48, round(s * fps))))
    if max_frames <= 96:
        side = 0           # few frames -> keep full detail
    elif max_frames <= 256:
        side = 1280
    elif max_frames <= 512:
        side = 896
    else:
        side = 640
    return fps, max_frames, side


def probe_video(path: str) -> dict:
    """Cheap header read: duration / fps / geometry, without decoding any frames.

    Used to plan segmented ("live view") analysis before touching the GPU.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        native_fps = float(stream.average_rate) if stream.average_rate else 24.0
        tb = stream.time_base
        duration = float(stream.duration * tb) if (stream.duration and tb) else None
        if not duration and container.duration:
            duration = float(container.duration) / AV_TIME_BASE
        total = stream.frames or (int(duration * native_fps) if duration else 0)
        width, height = stream.width, stream.height
    return {
        "duration": float(duration or 0.0),
        "native_fps": native_fps,
        "total_frames": int(total or 0),
        "width": int(width or 0),
        "height": int(height or 0),
    }


def sample_frames_window(
    path: str,
    target_fps: float,
    max_frames: int,
    start_time: float,
    end_time: float,
    lead_time: float = 0.0,
    min_frames: int = 4,
    longest_edge: int = 0,
    optimize: bool = False,
    fixed_fps: Optional[float] = None,
) -> Tuple[np.ndarray, VideoMetadata]:
    """Decode only the frames in ``[start_time - lead_time, end_time]`` of ``path``.

    ``lead_time`` pulls a few extra frames in from *before* the window so a chunk
    boundary that lands on a frozen / black frame still carries live context. Frame
    indices in the returned metadata are positions in the ORIGINAL video, so Qwen3-VL's
    "<12.5 seconds>" time markers stay absolute across chunks. ``longest_edge`` (px,
    0 = off) downscales each frame to that longer-side cap before the processor.
    ``optimize`` derives frame cap / size cap from this window's own length instead of
    using the three passed values; ``fixed_fps``, when given, pins the fps used in that
    derivation instead of scaling it off the window length (see optimize_video_params).

    Uses ``container.seek`` to jump to the window instead of decoding the whole file,
    so cost is bounded by the window length, not by where the window sits in the clip.
    """
    win_start = max(0.0, start_time - max(0.0, lead_time))

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        native_fps = float(stream.average_rate) if stream.average_rate else 24.0
        tb = stream.time_base
        stream_frames = stream.frames or 0
        full_duration = float(stream.duration * tb) if (stream.duration and tb) else None
        if full_duration:
            end_time = min(end_time, full_duration)

        if optimize:
            # Size from the nominal chunk length, not the lead-padded window, so a
            # small carry-over can't tip the chunk into a coarser fps bucket.
            target_fps, max_frames, longest_edge = optimize_video_params(
                max(end_time - start_time, 1.0), fixed_fps
            )

        resize_to = _resize_dims(stream.width, stream.height, longest_edge)
        win_end = max(end_time, win_start + 1.0 / max(target_fps, 0.1))

        span = win_end - win_start
        n = int(span * target_fps)
        n = max(n, min_frames if span * native_fps >= min_frames else 1)
        n = min(n, max_frames)
        n = max(n, 1)
        wanted = list(np.linspace(win_start, win_end, n))

        # Jump to the keyframe at/just before the window rather than decoding from 0.
        try:
            if tb:
                container.seek(int(win_start / tb), stream=stream, backward=True, any_frame=False)
            else:
                container.seek(int(win_start * AV_TIME_BASE), backward=True, any_frame=False)
        except Exception:  # noqa: BLE001 - fall back to a plain forward decode
            pass

        kept, kept_t, wi = [], [], 0
        for frame in container.decode(video=0):
            t = frame.time
            if t is None:
                continue
            if t + 1e-3 < win_start:
                continue
            if wi < n and t + 1e-6 >= wanted[wi]:
                kept.append(_frame_rgb(frame, resize_to))
                kept_t.append(t)
                wi += 1
                while wi < n and wanted[wi] <= t:  # skip wanted stamps already covered
                    wi += 1
            if wi >= n or t > win_end + 1e-3:
                break

    if not kept:
        raise ValueError(
            f"no frames could be decoded from {path} in [{win_start:.2f}s, {win_end:.2f}s]"
        )

    frames = np.stack(kept)
    # Absolute source-frame indices, forced strictly increasing (the processor's
    # timestamp math and its odd-count padding both assume a monotonic list).
    indices, last = [], -1
    for t in kept_t:
        v = int(round(t * native_fps))
        if v <= last:
            v = last + 1
        indices.append(v)
        last = v

    total = stream_frames or int((full_duration or win_end) * native_fps) or (indices[-1] + 1)
    meta = VideoMetadata(
        total_num_frames=total,
        fps=native_fps,
        width=frames.shape[2],
        height=frames.shape[1],
        duration=full_duration,
        video_backend="pyav",
        frames_indices=list(indices),
    )
    return frames, meta
