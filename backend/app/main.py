# Copyright (c) Alibaba Cloud.
#
# FastAPI application: one persistent model, streamed responses, static frontend.
import asyncio
import json
import os
import time
import uuid
from dataclasses import replace
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from . import history
from .config import Settings
from .timeline import CONCISE_MAX_NEW_TOKENS, Timeline, normalize_answer
from .inference import (
    build_messages,
    generate_chunk,
    is_video_file,
    resolve_options,
    stream_generate,
)
from .model_runtime import get_runtime, init_runtime
from .routes_history import router as history_router
from .runtime_settings import (
    FIELD_SPEC,
    get_defaults,
    get_launch_defaults,
    get_saved_keys,
    init_defaults,
    reset_defaults,
    update_defaults,
)
from .schemas import (
    ChatRequest,
    HealthResponse,
    SegmentedChatRequest,
    SettingsResponse,
    SettingsUpdate,
    UploadResponse,
)
from .video import optimize_video_params, probe_video

settings = Settings.from_env()

# Only one generation can run at a time (single model instance).
_gen_lock = asyncio.Lock()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

# How often to sweep for archived videos past their retention window (seconds).
HISTORY_CLEANUP_INTERVAL_S = 3600


async def _history_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(HISTORY_CLEANUP_INTERVAL_S)
        try:
            removed = await asyncio.to_thread(history.prune_expired)
            if removed:
                print(f"[history] pruned {removed} expired video(s)", flush=True)
        except Exception as e:  # noqa: BLE001 - a missed sweep is not fatal
            print(f"[history] cleanup sweep failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    history.init(settings.history_dir, settings.history_retention_days)
    await asyncio.to_thread(history.prune_expired)
    cleanup_task = asyncio.create_task(_history_cleanup_loop())

    # Free VRAM before loading the model. run.py normally does this first (and sets
    # QWEN_PREFLIGHT_DONE); this covers `uvicorn app.main:app` being run directly.
    if os.environ.get("QWEN_PREFLIGHT_DONE") != "1" and os.environ.get("QWEN_SKIP_PREFLIGHT") != "1":
        from .preflight import preflight

        await asyncio.to_thread(
            preflight,
            settings.device,
            free_ollama=os.environ.get("QWEN_FREE_OLLAMA", "1") != "0",
            min_free_gb=float(os.environ.get("QWEN_MIN_FREE_GB", "0") or 0),
        )

    # Blocking load - run it off the event loop so startup logs stream cleanly.
    rt = await asyncio.to_thread(init_runtime, settings)

    # Seed the editable defaults from the CLI/env, then re-apply anything previously
    # saved from the settings page. Sampling stays the checkpoint's own
    # generation_config - it is never overridden here.
    init_defaults(settings)
    yield
    cleanup_task.cancel()


app = FastAPI(title="Qwen3-VL UI backend", lifespan=lifespan)
app.include_router(history_router)


# Exit code run.py's supervisor treats as "GPU fault - restart me". Not 3: uvicorn
# already uses that for a startup failure (e.g. port in use), which must not restart.
GPU_FAULT_EXIT_CODE = 75


def _restart_if_faulted() -> None:
    """After an unrecoverable CUDA error, exit so the supervisor starts a fresh process.

    Called once a response has been fully written. The short delay lets the final SSE
    bytes reach the browser before the process goes away. An OOM never gets here - it
    is recovered in-process (see inference.generate_chunk).
    """
    if get_runtime().faulted:
        print("[recovery] CUDA context is unusable - exiting so the supervisor restarts the "
              "server", flush=True)
        asyncio.get_running_loop().call_later(1.0, os._exit, GPU_FAULT_EXIT_CODE)


def _upload_path(file_id: str) -> str:
    matches = list(settings.upload_dir.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=400, detail=f"unknown upload id: {file_id}")
    return str(matches[0].resolve())


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    rt = get_runtime()
    return HealthResponse(
        status="faulted" if rt.faulted else "ok",
        model=settings.checkpoint_path,
        device=rt.device_str,
        backend=rt.backend,
        flash_attn2=settings.flash_attn2,
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    ext = Path(file.filename or "").suffix.lower()
    if is_video_file(file.filename or ""):
        kind = "video"
    elif ext in IMAGE_EXTENSIONS:
        kind = "image"
    else:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {ext or '?'}")

    file_id = uuid.uuid4().hex
    dest = settings.upload_dir / f"{file_id}{ext}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)

    duration = None
    if kind == "video":
        # Cheap header read - lets the UI show what sampling each chunk will get.
        try:
            info = await asyncio.to_thread(probe_video, str(dest.resolve()))
            duration = info["duration"] or None
        except Exception:  # noqa: BLE001 - a duration the UI can't show is not fatal
            pass

    return UploadResponse(
        id=file_id, kind=kind, name=file.filename or dest.name, duration=duration
    )


def _settings_payload() -> SettingsResponse:
    return SettingsResponse(
        defaults=get_defaults().as_dict(),
        launch_defaults=get_launch_defaults().as_dict(),
        overridden=sorted(get_saved_keys()),
        fields=FIELD_SPEC,
        # Load-time flags: changing these needs a model reload, so they are read-only here.
        locked={
            "checkpoint_path": settings.checkpoint_path,
            "device": get_runtime().device_str,
            "flash_attn2": settings.flash_attn2,
            "history_retention_days": settings.history_retention_days,
            "host": settings.host,
            "port": settings.port,
        },
    )


@app.get("/api/settings", response_model=SettingsResponse)
async def read_settings() -> SettingsResponse:
    return _settings_payload()


@app.put("/api/settings", response_model=SettingsResponse)
async def write_settings(update: SettingsUpdate) -> SettingsResponse:
    patch = update.model_dump(exclude_none=True)
    try:
        new = update_defaults(patch)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    print(f"[settings] updated: {new.as_dict()}", flush=True)
    return _settings_payload()


@app.post("/api/settings/reset", response_model=SettingsResponse)
async def reset_settings() -> SettingsResponse:
    reset_defaults()
    print("[settings] reset to launch defaults", flush=True)
    return _settings_payload()


@app.post("/api/reset")
async def reset() -> dict:
    removed = 0
    for p in settings.upload_dir.glob("*"):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return {"removed": removed}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    rt = get_runtime()
    hf_messages = build_messages(req.messages, resolve_path=_upload_path)
    # Per-request options win, then the saved settings-page defaults, then launch flags.
    opts = resolve_options(get_defaults(), req.options, req.max_new_tokens)

    # Log the media the model will actually see. The whole history is re-sent on every
    # request, so this is the quickest way to tell "the model still knows about the
    # image" apart from "the model is guessing". Video never appears here - it only
    # ever goes through /api/chat/segmented.
    media = [
        f"{part.type}:{part.id[:8]}"
        for msg in req.messages
        for part in msg.content
        if part.type == "image" and part.id
    ]
    print(
        f"[chat] {len(req.messages)} msg(s), media={', '.join(media) or 'none'}, "
        f"max_new_tokens={opts.max_new_tokens}",
        flush=True,
    )

    async def event_stream():
        async with _gen_lock:
            prev = ""
            stats: dict = {}

            def _capture(s: dict) -> None:
                stats.update(s)
                print(
                    f"[chat] prompt={s['prompt_tokens']} tokens "
                    f"(images={s['images']}, video frames={s['frames'] or 'none'})",
                    flush=True,
                )

            sync_gen = stream_generate(rt, hf_messages, opts, on_stats=_capture)
            first = True
            async for full_text in iterate_in_threadpool(sync_gen):
                if first and stats:
                    # Let the UI show what the current settings actually cost.
                    first = False
                    yield f"data: {json.dumps({'stats': stats})}\n\n"
                if full_text.startswith(prev):
                    payload = {"delta": full_text[len(prev):]}
                else:
                    payload = {"replace": full_text}
                prev = full_text
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        _restart_if_faulted()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _fmt_ts(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# Layered onto the user's prompt for every chunk, per `answer_style` - see
# SegmentedChatRequest.answer_style. "auto" does not pick concise/detailed itself
# (there's no classifier); it just tells the model to size its own answer, instead of
# leaving the prompt completely unmodified.
ANSWER_STYLE_INSTRUCTIONS = {
    "auto": (
        "If this is a yes/no or counting question, answer with just that (a word or a "
        "number) - no explanation. If it asks you to describe or explain something, "
        "answer with a brief explanation instead."
    ),
    "concise": (
        'Answer with the bare minimum: reply with exactly "Yes" or "No" for yes/no '
        "questions, or just the number for counting questions. Output only that word or "
        "number - no explanation, no punctuation, no extra sentence."
    ),
    "detailed": (
        "Answer with a clear, well-reasoned explanation in a few sentences, describing "
        "what you observe in this part of the video and why."
    ),
}

# Fixed per-chunk sampling seed: the same chunk with the same inputs always gets the
# same answer, so an OOM retry at the same budget reproduces it exactly (and re-asking
# a question is repeatable).
CHUNK_SEED_BASE = 20250101


@app.post("/api/chat/segmented")
async def chat_segmented(req: SegmentedChatRequest):
    """Answer one question chunk-by-chunk over a long video.

    Each chunk is sampled and generated on its own (like a short standalone clip), so a
    10-minute video that a single `/api/chat` call can never fit becomes N small
    prefills. Chunks 2..N prepend a few frames of the previous chunk to cover dead
    frames at the cut. Each chunk's answer is independent - nothing is fed back into the
    model, and there is no final reconciliation pass; the model does not need to hold
    any previous chunk's text output in its context. Events:
        {plan:{duration,segment_seconds,count,segments:[{index,start,end,lead}]}}
        {segment_start:i}
        {segment_stats:{index,stats}}                      - once, before the first token
        {segment_delta:{index,delta}} | {segment_replace:{index,text}}
        {segment_done:{index,text,stats}}
        {done:true}
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    rt = get_runtime()
    path = _upload_path(req.video_id)
    video_name = req.video_name or Path(path).name
    try:
        info = await asyncio.to_thread(probe_video, path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not read video: {e}")

    duration = info["duration"]
    if duration <= 0:
        raise HTTPException(status_code=400, detail="could not determine the video's duration")

    defaults = get_defaults()
    opts = resolve_options(defaults, req.options)
    seg_len_raw = req.segment_seconds if req.segment_seconds is not None else defaults.segment_seconds
    overlap_frames = (
        req.overlap_frames if req.overlap_frames is not None else defaults.segment_overlap_frames
    )
    seg_len = min(max(float(seg_len_raw), 5.0), max(duration, 5.0))
    # "Carry-over frames" is a frame count; turn it into seconds using the fps each
    # chunk will actually be sampled at (the constant sample_fps setting).
    chunk_fps = optimize_video_params(seg_len, opts.sample_fps)[0]
    lead = (overlap_frames / chunk_fps) if chunk_fps else 0.0

    segments = []
    i = 0
    while i * seg_len < duration - 1e-3:
        start = round(i * seg_len, 3)
        end = round(min((i + 1) * seg_len, duration), 3)
        segments.append(
            {"index": len(segments), "start": start, "end": end,
             "lead": round(lead, 3) if i > 0 else 0.0}
        )
        i += 1
    if not segments:  # video shorter than one chunk
        segments = [{"index": 0, "start": 0.0, "end": round(duration, 3), "lead": 0.0}]
    n_segments = len(segments)

    style_instruction = ANSWER_STYLE_INSTRUCTIONS.get(req.answer_style)
    # Concise answers are one word / number per chunk: cap generation to match, and
    # join them into a whole-video timeline (see app/timeline.py).
    concise = req.answer_style == "concise"
    if concise:
        opts = replace(opts, max_new_tokens=min(opts.max_new_tokens, CONCISE_MAX_NEW_TOKENS))

    print(
        f"[segmented] {req.video_id[:8]} {_fmt_ts(duration)} -> {n_segments} x "
        f"{_fmt_ts(seg_len)} chunk(s), lead={lead:.2f}s, answer_style={req.answer_style}, "
        f"optimise({optimize_video_params(seg_len, opts.sample_fps)}) "
        f"max_new_tokens={opts.max_new_tokens} max_video_tokens={opts.max_video_tokens} "
        f"motion_threshold={opts.motion_threshold}",
        flush=True,
    )

    record_id = history.new_id()
    created_at = time.time()

    async def event_stream():
        # Copy the video into permanent storage in parallel with generation, so a big
        # file doesn't add to time-to-first-token; only awaited once the run finishes.
        # archive_video dedups on req.video_id, so asking several questions about the
        # same staged video reuses one archived copy instead of one per question.
        archive_task = asyncio.create_task(
            asyncio.to_thread(history.archive_video, record_id, path, req.video_id)
        )
        chunk_records: list = []
        run_status = "ok"
        faulted = False
        timeline = Timeline()
        t0 = time.monotonic()
        try:
            async with _gen_lock:
                yield _sse({"plan": {
                    "duration": duration,
                    "segment_seconds": seg_len,
                    "overlap_frames": overlap_frames,
                    "count": n_segments,
                    "segments": segments,
                }})

                for seg in segments:
                    idx = seg["index"]
                    yield _sse({"segment_start": idx})

                    label = (
                        f"[Segment {idx + 1} of {n_segments}: {_fmt_ts(seg['start'])}"
                        f"–{_fmt_ts(seg['end'])} of a {_fmt_ts(duration)} video]"
                    )
                    prompt_text = f"{label}\n{req.prompt}"
                    if style_instruction:
                        prompt_text += f"\n\n{style_instruction}"
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "video", "video": path},
                            {"type": "text", "text": prompt_text},
                        ],
                    }]
                    window = {"start": seg["start"], "end": seg["end"], "lead": seg["lead"]}
                    stats: dict = {}
                    prev = ""
                    skipped = False
                    retries = 0
                    # The first chunk is never skipped: there is nothing earlier whose
                    # answer a quiet stretch would be continuing.
                    threshold = opts.motion_threshold if idx > 0 else 0.0
                    sync_gen = generate_chunk(
                        rt, messages, opts, window,
                        seed=CHUNK_SEED_BASE + idx, motion_threshold=threshold,
                    )
                    async for kind, payload in iterate_in_threadpool(sync_gen):
                        if kind == "stats":
                            stats = payload
                            yield _sse({"segment_stats": {"index": idx, "stats": stats}})
                        elif kind == "text":
                            if payload.startswith(prev):
                                out = {"segment_delta": {"index": idx, "delta": payload[len(prev):]}}
                            else:
                                out = {"segment_replace": {"index": idx, "text": payload}}
                            prev = payload
                            yield _sse(out)
                        elif kind == "retry":
                            # The failed attempt's partial text is discarded; the retry
                            # streams its answer from scratch.
                            retries += 1
                            prev = ""
                            yield _sse({"segment_retry": {"index": idx, **payload}})
                            yield _sse({"segment_replace": {"index": idx, "text": ""}})
                        elif kind == "skipped":
                            skipped = True
                            stats = payload
                            prev = (
                                f"_No significant motion ({stats['motion']:.1f}% of the picture "
                                f"changed, below the {stats['motion_threshold']:g}% threshold) - "
                                f"skipped, the model was not run on this chunk._"
                            )
                            yield _sse({"segment_stats": {"index": idx, "stats": stats}})
                            yield _sse({"segment_replace": {"index": idx, "text": prev}})
                        elif kind == "fault":
                            faulted = True

                    if "**[" in prev:
                        run_status = "error"
                    record = {"index": idx, "start": seg["start"], "end": seg["end"],
                              "text": prev, "stats": stats}
                    if skipped:
                        record["skipped"] = True
                    if retries:
                        record["retries"] = retries
                    if concise:
                        answer = timeline.add(
                            idx, seg["start"], seg["end"],
                            None if skipped else normalize_answer(prev), skipped=skipped,
                        )
                        if skipped and answer is not None:
                            prev = (
                                f"**{answer}** _(carried over - no motion: {stats['motion']:.1f}% "
                                f"of the picture changed, below the {stats['motion_threshold']:g}% "
                                f"threshold, so the model was not run on this chunk)_"
                            )
                            yield _sse({"segment_replace": {"index": idx, "text": prev}})
                        record["answer"] = answer
                        yield _sse({"timeline": {"spans": timeline.as_list()}})
                    chunk_records.append(record)
                    yield _sse({"segment_done": {"index": idx, "text": prev, "stats": stats}})
                    if not skipped:
                        # Hand this chunk's cached blocks back between chunks: every chunk
                        # has a slightly different shape, and over a long run the cache
                        # fragments until it no longer fits beside the model.
                        await asyncio.to_thread(rt.gc)
                    note = " skipped (no motion)" if skipped else (f" after {retries} retr{'y' if retries == 1 else 'ies'}" if retries else "")
                    print(f"[segmented] chunk {idx + 1}/{n_segments} done ({len(prev)} chars){note}", flush=True)
                    if faulted:
                        # Every later CUDA call would fail too - stop here; the process
                        # restarts once this response is flushed.
                        yield _sse({"fault": True})
                        break

                yield _sse({"done": True})
        finally:
            # Runs on normal completion, on an unhandled error, and on early client
            # disconnect (GeneratorExit) alike, so every run - partial or not - is
            # logged with whatever chunks it actually produced.
            try:
                video_path = await archive_task
            except Exception:  # noqa: BLE001 - a missing archive copy is not fatal
                video_path = None
            fps, max_frames, max_side = optimize_video_params(seg_len, opts.sample_fps)
            settings_snapshot = {
                "segment_seconds": seg_len,
                "overlap_frames": overlap_frames,
                "max_new_tokens": opts.max_new_tokens,
                "sample_fps": opts.sample_fps,
                "answer_style": req.answer_style,
                "max_video_tokens": opts.max_video_tokens,
                "motion_threshold": opts.motion_threshold,
                "sampling": {"fps": fps, "max_frames": max_frames, "max_side": max_side},
            }
            if concise:
                settings_snapshot["timeline"] = timeline.as_list()
            await asyncio.to_thread(
                history.finalize_record,
                record_id, created_at, video_name, duration, video_path,
                req.prompt, settings_snapshot, chunk_records,
                time.monotonic() - t0, run_status if chunk_records else "error",
                req.video_id,
            )
            _restart_if_faulted()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class NoCacheStaticFiles(StaticFiles):
    """Serve the frontend with revalidate-always so edits show up on a plain reload
    (no more stale app.js / styles.css during development)."""

    def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: D401
        return False

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


# Static frontend last, so /api/* wins. html=True serves index.html at "/".
if settings.frontend_dir.is_dir():
    app.mount("/", NoCacheStaticFiles(directory=str(settings.frontend_dir), html=True), name="frontend")
