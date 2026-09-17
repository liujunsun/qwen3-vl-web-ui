# Copyright (c) Alibaba Cloud.
#
# Loads the Qwen3-VL model + processor exactly once and keeps them resident for the
# lifetime of the server process. This is the single biggest latency win over the
# original Gradio demo, which was fine because it also loaded once - but here we make
# the "load once, warm up, then serve" contract explicit.
import gc
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import Settings


class ModelRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.processor = None
        self.backend = "hf"

    # --- loading -----------------------------------------------------------------
    def load(self) -> None:
        s = self.settings
        device_map = s.device  # 'cuda:0' | 'auto' | 'cpu'

        load_kwargs = {"torch_dtype": "auto", "device_map": device_map}
        if s.flash_attn2:
            load_kwargs["attn_implementation"] = "flash_attention_2"

        print(f"[model] loading {s.checkpoint_path} (device_map={device_map}, "
              f"flash_attn2={s.flash_attn2}) ...", flush=True)
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(s.checkpoint_path, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(s.checkpoint_path)
        # video_processor.fps / .max_frames are never consulted: every video is pre-
        # sampled by app.video.sample_frames_window and handed in with
        # do_sample_frames=False, so the processor never samples frames itself.

        self.model.eval()
        print(f"[model] loaded in {time.time() - t0:.1f}s", flush=True)

    # --- warmup ----------------------------------------------------------------
    def warmup(self) -> None:
        """Run one tiny generation so CUDA kernels / caches are initialised before the
        first real request hits."""
        if self.model is None:
            return
        try:
            t0 = time.time()
            messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.inference_mode():
                self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
            print(f"[model] warmup done in {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # warmup is best-effort
            print(f"[model] warmup skipped: {e}", flush=True)

    # --- housekeeping --------------------------------------------------------------
    @staticmethod
    def gc() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def device_str(self) -> str:
        try:
            return str(self.model.device)
        except Exception:
            return self.settings.device


_runtime: "ModelRuntime | None" = None


def init_runtime(settings: Settings) -> ModelRuntime:
    global _runtime
    _runtime = ModelRuntime(settings)
    _runtime.load()
    _runtime.warmup()
    return _runtime


def get_runtime() -> ModelRuntime:
    if _runtime is None:
        raise RuntimeError("Model runtime not initialised yet.")
    return _runtime
