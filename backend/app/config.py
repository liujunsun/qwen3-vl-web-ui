# Copyright (c) Alibaba Cloud.
#
# Runtime configuration for the Qwen3-VL backend. Values are read from environment
# variables that `run.py` sets from its CLI arguments, so the FastAPI app can be
# started either through `run.py` or directly with `uvicorn app.main:app`.
import os
from dataclasses import dataclass
from pathlib import Path

# backend/  (this file is backend/app/config.py)
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BACKEND_DIR.parent


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    checkpoint_path: str = "Qwen/Qwen3-VL-8B-Instruct"
    # 'cuda:0' keeps the whole model on one GPU (lowest latency when it fits).
    # 'auto' splits across visible GPUs. 'cpu' forces CPU (slow).
    device: str = "cuda:0"
    flash_attn2: bool = False
    max_new_tokens: int = 1024
    # Every video question is answered chunk-by-chunk ("segmented analysis"); a video
    # is never sampled and fed to the model as a single whole-clip pass. Frame cap /
    # frame size are derived per chunk from the chunk's own duration (see
    # video.optimize_video_params); fps is the constant `sample_fps` below, applied
    # the same way to every chunk in a run.
    segment_seconds: int = 60
    segment_overlap_frames: int = 4
    # Frames sampled per second of video, per chunk. Constant across every chunk in a
    # run (including a shorter final chunk) rather than scaling with chunk length.
    sample_fps: float = 4.0
    # Cap on visual tokens per chunk. The processor spreads a fixed pixel budget over
    # however many frames it gets, so fewer frames just means bigger frames - this is
    # the knob that actually bounds VRAM. 12288 matches the processor's own default.
    max_video_tokens: int = 4096
    # Skip a chunk (no model call) when less than this % of the picture changes.
    # 0 turns it off. See video.motion_score for how it is measured.
    motion_threshold: float = 1.0
    # Analytics history: every segmented-analysis run is logged permanently (prompt,
    # settings, per-chunk output, stats). The archived video copy is deleted after this
    # many days to bound disk use; the record itself is kept forever. 0 = never archive
    # the video at all (metadata/output only).
    history_retention_days: int = 30
    host: str = "127.0.0.1"
    port: int = 8000
    upload_dir: Path = BACKEND_DIR / ".uploads"
    history_dir: Path = BACKEND_DIR / ".history"
    frontend_dir: Path = PROJECT_DIR / "frontend"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            checkpoint_path=_env_str("QWEN_CHECKPOINT", cls.checkpoint_path),
            device=_env_str("QWEN_DEVICE", cls.device),
            flash_attn2=_env_bool("QWEN_FLASH_ATTN2", cls.flash_attn2),
            max_new_tokens=_env_int("QWEN_MAX_NEW_TOKENS", cls.max_new_tokens),
            segment_seconds=_env_int("QWEN_SEGMENT_SECONDS", cls.segment_seconds),
            segment_overlap_frames=_env_int("QWEN_SEGMENT_OVERLAP_FRAMES", cls.segment_overlap_frames),
            sample_fps=_env_float("QWEN_SAMPLE_FPS", cls.sample_fps),
            max_video_tokens=_env_int("QWEN_MAX_VIDEO_TOKENS", cls.max_video_tokens),
            motion_threshold=_env_float("QWEN_MOTION_THRESHOLD", cls.motion_threshold),
            history_retention_days=_env_int("QWEN_HISTORY_RETENTION_DAYS", cls.history_retention_days),
            host=_env_str("QWEN_HOST", cls.host),
            port=_env_int("QWEN_PORT", cls.port),
        )
