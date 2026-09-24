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
import subprocess
import sys
import time
from argparse import ArgumentParser

# Supervisor policy: restart a crashed server, but give up if it keeps dying (a broken
# install or a GPU that is truly gone would otherwise restart forever).
SUPERVISOR_ENV = "QWEN_SUPERVISED_CHILD"
RESTART_DELAY_S = 3
MAX_RESTARTS = 5
RESTART_WINDOW_S = 600
GPU_FAULT_EXIT_CODE = 75  # app.main.GPU_FAULT_EXIT_CODE
UVICORN_STARTUP_FAILURE = 3  # config problem (port in use...) - restarting won't help


def supervise() -> int:
    """Run the server as a child process and restart it whenever it dies.

    Out-of-memory errors never reach this - they are recovered inside the server
    without losing the model. This covers what can't be fixed in-process: a CUDA fault
    that poisons the context (the server exits with GPU_FAULT_EXIT_CODE on purpose),
    a driver reset, or a hard crash. A restart reloads the model (~15s) and re-runs the
    preflight, which frees VRAM held by anything stale.
    """
    cmd = [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    env = {**os.environ, SUPERVISOR_ENV: "1"}
    restarts: list = []
    while True:
        child = subprocess.Popen(cmd, env=env)
        try:
            code = child.wait()
        except KeyboardInterrupt:
            # Ctrl+C reaches the child too (same console); let it shut down cleanly.
            try:
                child.wait(timeout=30)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                child.kill()
            return 0
        if code == 0:
            return 0
        if code == UVICORN_STARTUP_FAILURE:
            print("[supervisor] server failed to start (is another run.py already using the "
                  "port?) - not restarting.", flush=True)
            return code
        now = time.time()
        restarts = [t for t in restarts if now - t < RESTART_WINDOW_S] + [now]
        reason = "GPU fault" if code == GPU_FAULT_EXIT_CODE else f"exit code {code}"
        if len(restarts) > MAX_RESTARTS:
            print(f"[supervisor] server died ({reason}) {len(restarts)} times in "
                  f"{RESTART_WINDOW_S // 60} min - giving up.", flush=True)
            return code
        print(f"[supervisor] server died ({reason}); restarting in {RESTART_DELAY_S}s "
              f"({len(restarts)}/{MAX_RESTARTS})...", flush=True)
        time.sleep(RESTART_DELAY_S)


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
    parser.add_argument("--max-video-tokens", type=int, default=4096,
                        help="Visual tokens per chunk - frames are downscaled to fit. The main "
                             "VRAM knob; lower it if you run out of GPU memory (default: %(default)s)")
    parser.add_argument("--motion-threshold", type=float, default=1.0,
                        help="Skip chunks where less than this %% of the picture changes "
                             "(0 = off, default: %(default)s)")
    parser.add_argument("--history-retention-days", type=int, default=30,
                        help="Every segmented-analysis run is logged permanently for the analytics "
                             "dashboard; the archived video copy is deleted after this many days to "
                             "bound disk use (0 = never archive the video at all). Default: %(default)s")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Dev auto-reload (reloads the model too).")
    parser.add_argument("--supervise", action=argparse.BooleanOptionalAction, default=True,
                        help="Restart the server automatically if it crashes or hits an "
                             "unrecoverable GPU error (default: on). --no-supervise to disable.")
    args = parser.parse_args()

    # --reload already runs the app in a reloader subprocess; don't stack a supervisor on it.
    if args.supervise and not args.reload and os.environ.get(SUPERVISOR_ENV) != "1":
        sys.exit(supervise())

    os.environ["QWEN_CHECKPOINT"] = args.checkpoint_path
    os.environ["QWEN_DEVICE"] = args.device
    os.environ["QWEN_FLASH_ATTN2"] = "1" if args.flash_attn2 else "0"
    os.environ["QWEN_MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    os.environ["QWEN_SEGMENT_SECONDS"] = str(args.segment_seconds)
    os.environ["QWEN_SEGMENT_OVERLAP_FRAMES"] = str(args.segment_overlap_frames)
    os.environ["QWEN_SAMPLE_FPS"] = str(args.sample_fps)
    os.environ["QWEN_MAX_VIDEO_TOKENS"] = str(args.max_video_tokens)
    os.environ["QWEN_MOTION_THRESHOLD"] = str(args.motion_threshold)
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
