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
from typing import List, Tuple

import av
import numpy as np
from transformers.video_utils import VideoMetadata


def sample_frames(path: str, target_fps: float, max_frames: int, min_frames: int = 4):
    """Decode `path` at ~`target_fps`, capped at `max_frames`.

    Returns (frames HWC uint8 array, VideoMetadata) ready for a `do_sample_frames=False`
    processor call.
    """
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"  # let ffmpeg use all cores for decode

        native_fps = float(stream.average_rate) if stream.average_rate else 24.0
        total = stream.frames
        duration = float(stream.duration * stream.time_base) if stream.duration else None
        if not total:  # some containers don't store a frame count
            total = int((duration or 0) * native_fps) or max_frames

        indices = _plan_indices(total, native_fps, target_fps, max_frames, min_frames)
        wanted = set(indices)
        kept, i = [], 0
        for frame in container.decode(video=0):
            if i in wanted:
                kept.append(frame.to_ndarray(format="rgb24"))
                if len(kept) == len(indices):
                    break
            i += 1

    if not kept:
        raise ValueError(f"no frames could be decoded from {path}")

    # A short/truncated file can yield fewer frames than planned; keep metadata honest.
    indices = indices[: len(kept)]
    frames = np.stack(kept)
    meta = VideoMetadata(
        total_num_frames=total,
        fps=native_fps,
        width=frames.shape[2],
        height=frames.shape[1],
        duration=duration,
        video_backend="pyav",
        frames_indices=list(indices),
    )
    return frames, meta


def _plan_indices(
    total: int, native_fps: float, target_fps: float, max_frames: int, min_frames: int
) -> List[int]:
    """Uniformly spaced source-frame indices, matching Qwen3VLVideoProcessor.sample_frames."""
    n = int(total / native_fps * target_fps) if native_fps else max_frames
    n = min(min(max(n, min_frames), max_frames), total)
    n = max(n, 1)
    return np.linspace(0, total - 1, n).round().astype(int).tolist()
