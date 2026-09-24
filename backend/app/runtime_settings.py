# Copyright (c) Alibaba Cloud.
#
# Live-editable generation defaults ("tier 2" settings).
#
# `Settings` (config.py) is a frozen dataclass built once from the CLI/env at boot -
# that stays the immutable record of how the server was launched. This module holds
# the subset of knobs that can change *between requests* without reloading the model,
# and persists them so they survive a restart.
#
# Precedence for any one field:
#     per-request options  >  saved override (settings.json)  >  launch default (CLI/env)
#
# Only knobs that need no reload live here. checkpoint_path / device / flash_attn2 are
# baked into `from_pretrained` and are deliberately NOT editable at runtime. Every video
# is analysed chunk-by-chunk; each chunk's frame cap / frame size are derived
# automatically from its own duration (see video.optimize_video_params), while its
# sampling rate (fps) is the constant `sample_fps` knob below - the same for every
# chunk in a run. Generation always samples with the checkpoint's own generation_config
# (no temperature / top-p / top-k override).
import json
import math
import threading
from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, Optional, Set

from .config import BACKEND_DIR, Settings

SETTINGS_FILE = BACKEND_DIR / "settings.json"

# field -> (kind, range, label, help). The frontend renders its controls straight from
# this, so ranges are defined in exactly one place.
FIELD_SPEC: Dict[str, Dict[str, Any]] = {
    "max_new_tokens": {
        "kind": "int", "min": 16, "max": 8192, "step": 16,
        "label": "Max new tokens",
        "help": "Ceiling on response length. A cap, not a target - generation still stops at "
                "EOS, so raising it costs nothing unless the model actually uses it.",
    },
    "segment_seconds": {
        "kind": "int", "min": 10, "max": 7200, "step": 10,
        "label": "Chunk length (seconds)",
        "help": "Every video question is answered chunk-by-chunk. Seconds of source video per "
                "chunk - shorter means more chunks, each cheaper and quicker to first answer; "
                "longer means fewer chunks with more context each.",
    },
    "segment_overlap_frames": {
        "kind": "int", "min": 0, "max": 240, "step": 1,
        "label": "Carry-over frames",
        "help": "Sampled frames from the end of the previous chunk prepended to the next one, "
                "so a cut that lands on a frozen or black frame still has live context.",
    },
    "sample_fps": {
        "kind": "float", "min": 0.1, "max": 10.0, "step": 0.1,
        "label": "Sampling rate (fps)",
        "help": "Frames sampled per second of video, per chunk - the same rate for every chunk "
                "in a run, including a shorter final chunk. Frame cap and frame size are still "
                "derived automatically from this rate and each chunk's length.",
    },
    "max_video_tokens": {
        "kind": "int", "min": 256, "max": 16384, "step": 256,
        "label": "Max video tokens per chunk",
        "help": "The main VRAM knob. Each chunk's frames are downscaled to fit this many "
                "visual tokens - lowering fps alone does not save memory, it only makes "
                "each frame larger. ~4096 suits a 12 GB GPU with the 2B model; raise it for "
                "more detail if you have headroom.",
    },
    "motion_threshold": {
        "kind": "float", "min": 0.0, "max": 20.0, "step": 0.1,
        "label": "Motion skip threshold (% changed)",
        "help": "Chunks where less than this % of the picture changes are skipped without "
                "running the model - saves GPU time on idle footage. On a fixed camera, idle "
                "chunks measure ~0.3-0.9% and real activity 1.5%+. The first chunk is never "
                "skipped. 0 = off.",
    },
}


@dataclass(frozen=True)
class GenerationDefaults:
    """The live-editable knobs. Frozen so it is swapped atomically, never mutated."""

    max_new_tokens: int = 1024
    segment_seconds: int = 60
    segment_overlap_frames: int = 4
    sample_fps: float = 4.0
    max_video_tokens: int = 4096
    motion_threshold: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce(field: str, value: Any) -> Any:
    """Clamp `value` into the field's declared range. Raises ValueError if unusable."""
    spec = FIELD_SPEC[field]
    if spec["kind"] == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    try:
        num = int(round(float(value))) if spec["kind"] == "int" else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field}: expected a number, got {value!r}")
    if isinstance(num, float) and not math.isfinite(num):
        raise ValueError(f"{field}: must be a finite number")
    return max(spec["min"], min(spec["max"], num))


class _Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._launch = GenerationDefaults()
        self._current = GenerationDefaults()
        self._saved_keys: Set[str] = set()

    def init(self, settings: Settings) -> None:
        """Seed launch defaults from the CLI/env, then layer saved overrides on top."""
        launch = GenerationDefaults(
            max_new_tokens=settings.max_new_tokens,
            segment_seconds=settings.segment_seconds,
            segment_overlap_frames=settings.segment_overlap_frames,
            sample_fps=settings.sample_fps,
            max_video_tokens=settings.max_video_tokens,
            motion_threshold=settings.motion_threshold,
        )

        with self._lock:
            self._launch = launch
            saved = self._read_file()
            self._saved_keys = set(saved)
            self._current = replace(launch, **saved) if saved else launch
        if saved:
            print(f"[settings] loaded saved overrides: {', '.join(sorted(saved))}", flush=True)

    # --- persistence -------------------------------------------------------------
    def _read_file(self) -> Dict[str, Any]:
        if not SETTINGS_FILE.exists():
            return {}
        try:
            raw = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[settings] ignoring unreadable {SETTINGS_FILE.name}: {e}", flush=True)
            return {}
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        for key, value in raw.items():
            if key in FIELD_SPEC:
                try:
                    out[key] = coerce(key, value)
                except ValueError:
                    pass  # drop junk rather than refusing to boot
        return out

    def _write_file(self, values: Dict[str, Any]) -> None:
        try:
            if values:
                SETTINGS_FILE.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
            elif SETTINGS_FILE.exists():
                SETTINGS_FILE.unlink()
        except OSError as e:
            print(f"[settings] could not persist {SETTINGS_FILE.name}: {e}", flush=True)

    # --- accessors ---------------------------------------------------------------
    def current(self) -> GenerationDefaults:
        with self._lock:
            return self._current

    def launch(self) -> GenerationDefaults:
        with self._lock:
            return self._launch

    def saved_keys(self) -> Set[str]:
        with self._lock:
            return set(self._saved_keys)

    def update(self, patch: Dict[str, Any]) -> GenerationDefaults:
        """Apply a partial update, persist it, and return the new defaults."""
        clean = {k: coerce(k, v) for k, v in patch.items() if k in FIELD_SPEC and v is not None}
        if not clean:
            return self.current()
        with self._lock:
            self._current = replace(self._current, **clean)
            # Persist only what differs from launch, so the file stays a readable record
            # of "what I changed" rather than a full snapshot that shadows future flags.
            launch = self._launch.as_dict()
            diff = {k: v for k, v in self._current.as_dict().items() if v != launch[k]}
            self._saved_keys = set(diff)
            self._write_file(diff)
            return self._current

    def reset(self) -> GenerationDefaults:
        """Drop every saved override and go back to how the server was launched."""
        with self._lock:
            self._current = self._launch
            self._saved_keys = set()
            self._write_file({})
            return self._current


_store = _Store()

init_defaults = _store.init
get_defaults = _store.current
get_launch_defaults = _store.launch
get_saved_keys = _store.saved_keys
update_defaults = _store.update
reset_defaults = _store.reset
