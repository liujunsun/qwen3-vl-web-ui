# Copyright (c) Alibaba Cloud.
#
# FastAPI application: one persistent model, streamed responses, static frontend.
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import iterate_in_threadpool

from .config import Settings
from .inference import build_messages, is_video_file, stream_generate
from .model_runtime import get_runtime, init_runtime
from .schemas import ChatRequest, HealthResponse, UploadResponse

settings = Settings.from_env()

# Only one generation can run at a time (single model instance).
_gen_lock = asyncio.Lock()

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

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
    await asyncio.to_thread(init_runtime, settings)
    yield


app = FastAPI(title="Qwen3-VL UI backend", lifespan=lifespan)


def _upload_path(file_id: str) -> str:
    matches = list(settings.upload_dir.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=400, detail=f"unknown upload id: {file_id}")
    return str(matches[0].resolve())


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    rt = get_runtime()
    return HealthResponse(
        status="ok",
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
    return UploadResponse(id=file_id, kind=kind, name=file.filename or dest.name)


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
    max_new_tokens = req.max_new_tokens or settings.max_new_tokens

    async def event_stream():
        async with _gen_lock:
            prev = ""
            sync_gen = stream_generate(rt, hf_messages, max_new_tokens)
            async for full_text in iterate_in_threadpool(sync_gen):
                if full_text.startswith(prev):
                    payload = {"delta": full_text[len(prev):]}
                else:
                    payload = {"replace": full_text}
                prev = full_text
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

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
