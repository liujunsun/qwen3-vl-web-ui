# Copyright (c) Alibaba Cloud.
#
# CLI entrypoint. Mirrors the useful flags of the reference demo
# (Qwen3-VL/web_demo_mm.py) and launches a single-worker uvicorn server that keeps
# the model resident for the lowest possible per-request latency.
#
# Examples:
#   python run.py -c Qwen/Qwen3-VL-8B-Instruct
#   python run.py -c Qwen/Qwen3-VL-8B-Instruct --device auto
#   python run.py --flash-attn2 --port 8080
import argparse
import os
from argparse import ArgumentParser


def main() -> None:
    parser = ArgumentParser(description="Run the Qwen3-VL UI backend (HF backend).")
    parser.add_argument("-c", "--checkpoint-path", default="Qwen/Qwen3-VL-8B-Instruct",
                        help="Checkpoint name or path (default: %(default)s)")
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "auto", "cpu"],
                        help="cuda:0 = whole model on GPU 0 (lowest latency when it fits); "
                             "auto = split across GPUs; cpu = CPU only (slow). Default: %(default)s")
    parser.add_argument("--flash-attn2", action="store_true",
                        help="Load with attn_implementation=flash_attention_2 (needs flash-attn).")

    # --- startup preflight (free VRAM before loading the model) -------------------
    parser.add_argument("--free-ollama", action=argparse.BooleanOptionalAction, default=True,
                        help="Unload any resident Ollama models to free GPU memory (default: on). "
                             "Use --no-free-ollama to leave them alone.")
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help="Preflight VRAM check: warn if the target GPU has less than this many "
                             "GB free. 0 = report only (default).")
    parser.add_argument("--strict-vram", action="store_true",
                        help="With --min-free-gb, abort startup instead of warning when VRAM is low.")
    parser.add_argument("--preflight-wait", type=int, default=15,
                        help="Seconds to wait for other processes to release VRAM before the "
                             "check (default: %(default)s).")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the GPU cache-clear / VRAM preflight entirely.")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Default generation length (default: %(default)s)")
    parser.add_argument("--segment-seconds", type=int, default=60,
                        help="Seconds of video per chunk - every video question is answered "
                             "chunk-by-chunk (default: %(default)s)")
    parser.add_argument("--segment-overlap-frames", type=int, default=4,
                        help="Frames of the previous chunk carried into the next (default: %(default)s)")
    parser.add_argument("--sample-fps", type=float, default=4.0,
                        help="Frames sampled per second of video, per chunk - the same rate for "
                             "every chunk in a run (default: %(default)s)")
    parser.add_argument("--history-retention-days", type=int, default=30,
                        help="Every segmented-analysis run is logged permanently for the analytics "
                             "dashboard; the archived video copy is deleted after this many days to "
                             "bound disk use (0 = never archive the video at all). Default: %(default)s")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload (reloads the model too).")
    args = parser.parse_args()

    os.environ["QWEN_CHECKPOINT"] = args.checkpoint_path
    os.environ["QWEN_DEVICE"] = args.device
    os.environ["QWEN_FLASH_ATTN2"] = "1" if args.flash_attn2 else "0"
    os.environ["QWEN_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["QWEN_SEGMENT_SECONDS"] = str(args.segment_seconds)
    os.environ["QWEN_SEGMENT_OVERLAP_FRAMES"] = str(args.segment_overlap_frames)
    os.environ["QWEN_SAMPLE_FPS"] = str(args.sample_fps)
    os.environ["QWEN_HISTORY_RETENTION_DAYS"] = str(args.history_retention_days)
    os.environ["QWEN_HOST"] = args.host
    os.environ["QWEN_PORT"] = str(args.port)
    os.environ["QWEN_FREE_OLLAMA"] = "1" if args.free_ollama else "0"
    os.environ["QWEN_MIN_FREE_GB"] = str(args.min_free_gb)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.skip_preflight:
        # Also stop the app-process lifespan from running its own fallback preflight.
        os.environ["QWEN_SKIP_PREFLIGHT"] = "1"
    else:
        from app.preflight import preflight

        preflight(
            args.device,
            free_ollama=args.free_ollama,
            min_free_gb=args.min_free_gb,
            strict=args.strict_vram,
            wait_seconds=args.preflight_wait,
        )
        # Tell the app process the preflight already ran (matters only under --reload).
        os.environ["QWEN_PREFLIGHT_DONE"] = "1"

    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        app_dir=os.path.dirname(os.path.abspath(__file__)),
    )


if __name__ == "__main__":
    main()
