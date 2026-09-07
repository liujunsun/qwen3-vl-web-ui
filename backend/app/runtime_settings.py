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
# baked into `from_pretrained` and are deliberately NOT editable at runtime.
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
    "temperature": {
        "kind": "float", "min": 0.0, "max": 2.0, "step": 0.05,
        "label": "Temperature",
        "help": "0 = greedy and deterministic (sampling off). Higher = more varied wording.",
    },
    "top_p": {
        "kind": "float", "min": 0.05, "max": 1.0, "step": 0.05,
        "label": "Top-p",
        "help": "Nucleus sampling cutoff. Ignored when temperature is 0.",
    },
    "top_k": {
        "kind": "int", "min": 0, "max": 200, "step": 1,
        "label": "Top-k",
        "help": "0 disables top-k. Ignored when temperature is 0.",
    },
    "video_fps": {
        "kind": "float", "min": 0.1, "max": 8.0, "step": 0.1,
        "label": "Video FPS",
        "help": "Frames sampled per second of video. Binds on short clips; on long clips the "
                "frame cap takes over first.",
    },
    "video_max_frames": {
        "kind": "int", "min": 4, "max": 768, "step": 4,
        "label": "Video max frames",
        "help": "Hard cap on sampled frames. The single biggest lever on prompt length, VRAM "
                "and time-to-first-token for video.",
    },
}

# Named starting points, shown as one-click presets in the UI.
PRESETS: Dict[str, Dict[str, Any]] = {
    "fast": {"video_fps": 0.5, "video_max_frames": 24, "max_new_tokens": 512},
    "balanced": {"video_fps": 1.0, "video_max_frames": 64, "max_new_tokens": 1024},
    # More frames than this stops buying detail on a long clip: past the model's pixel
    # budget the frames get downscaled, so a lower cap keeps more spatial resolution.
    "detailed": {"video_fps": 2.0, "video_max_frames": 256, "max_new_tokens": 2048},
    # What Qwen3-VL itself uses when nothing overrides it.
    "qwen default": {"video_fps": 2.0, "video_max_frames": 768, "max_new_tokens": 1024},
}


@dataclass(frozen=True)
class GenerationDefaults:
    """The live-editable knobs. Frozen so it is swapped atomically, never mutated."""

    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    video_fps: float = 2.0
    video_max_frames: int = 768

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def coerce(field: str, value: Any) -> Any:
    """Clamp `value` into the field's declared range. Raises ValueError if unusable."""
    spec = FIELD_SPEC[field]
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

    def init(self, settings: Settings, generation_config: Optional[Any] = None) -> None:
        """Seed launch defaults from the CLI/env, then layer saved overrides on top."""
        launch = GenerationDefaults(
            max_new_tokens=settings.max_new_tokens,
            video_fps=settings.video_fps,
            video_max_frames=settings.video_max_frames,
        )
        # Sampling params have no CLI flag, so take the checkpoint's own generation_config;
        # the UI then opens showing what the model would actually have done.
        if generation_config is not None:
            picked: Dict[str, Any] = {}
            for field in ("temperature", "top_p", "top_k"):
                value = getattr(generation_config, field, None)
                if value is not None:
                    try:
                        picked[field] = coerce(field, value)
                    except ValueError:
                        pass
            # do_sample=False in the checkpoint means greedy; surface that as temperature 0.
            if getattr(generation_config, "do_sample", True) is False:
                picked["temperature"] = 0.0
            launch = replace(launch, **picked)

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
