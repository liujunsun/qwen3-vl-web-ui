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
    video_fps: float = 1.0
    video_max_frames: int = 128
    max_new_tokens: int = 1024
    host: str = "127.0.0.1"
    port: int = 8000
    upload_dir: Path = BACKEND_DIR / ".uploads"
    frontend_dir: Path = PROJECT_DIR / "frontend"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            checkpoint_path=_env_str("QWEN_CHECKPOINT", cls.checkpoint_path),
            device=_env_str("QWEN_DEVICE", cls.device),
            flash_attn2=_env_bool("QWEN_FLASH_ATTN2", cls.flash_attn2),
            video_fps=_env_float("QWEN_VIDEO_FPS", cls.video_fps),
            video_max_frames=_env_int("QWEN_VIDEO_MAX_FRAMES", cls.video_max_frames),
            max_new_tokens=_env_int("QWEN_MAX_NEW_TOKENS", cls.max_new_tokens),
            host=_env_str("QWEN_HOST", cls.host),
            port=_env_int("QWEN_PORT", cls.port),
        )
