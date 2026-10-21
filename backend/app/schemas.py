# Copyright (c) Alibaba Cloud.
#
# Request / response models for the JSON API.
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    """A part of a /api/chat message. Video is deliberately not an option here: every
    video question goes through POST /api/chat/segmented (chunked analysis) instead -
    there is no whole-clip chat path."""

    type: Literal["text", "image"]
    text: Optional[str] = None
    # id returned by POST /api/upload (for image parts)
    id: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: List[ContentPart]


class GenerationOptions(BaseModel):
    """Per-request overrides. Any field left None falls back to the server defaults.

    Bounds mirror runtime_settings.FIELD_SPEC; that module clamps as well, so these are
    here to reject nonsense early with a 422 rather than to be the only guard.
    """

    max_new_tokens: Optional[int] = Field(None, ge=16, le=8192)


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    options: Optional[GenerationOptions] = None
    # Kept for backwards compatibility with the original API; `options.max_new_tokens`
    # wins when both are sent.
    max_new_tokens: Optional[int] = None


class SegmentedChatRequest(BaseModel):
    """POST /api/chat/segmented - split one long video into equal chunks and stream a
    separate answer for each as it finishes ("live view" analysis)."""

    # upload id from POST /api/upload
    video_id: str
    prompt: str
    # Original filename, as returned by POST /api/upload - purely cosmetic, used to
    # label this run in the analytics history (the server only knows the on-disk
    # uuid-based path otherwise). Falls back to the disk filename if omitted.
    video_name: Optional[str] = None
    options: Optional[GenerationOptions] = None
    # Chunk length / carry-over frames. Left None, they fall back to the saved
    # Segmented-analysis settings (runtime_settings.FIELD_SPEC).
    segment_seconds: Optional[float] = Field(None, ge=5.0, le=7200.0)
    overlap_frames: Optional[int] = Field(None, ge=0, le=240)
    # Layers an extra instruction onto the prompt so the answer's shape matches the
    # question: "concise" for yes/no/counting questions, "detailed" for descriptive or
    # reasoning ones. "auto" (default) adds no instruction - the model's own judgement.
    # "concise" answers are also joined into a whole-video timeline (app/timeline.py).
    answer_style: Literal["auto", "concise", "detailed"] = "auto"
    # Tags this run in the analytics history so every video in one Batch Processing
    # run (frontend/batch.html) can be filtered back out as a group afterward. Not used
    # outside Batch Processing - left unset for ordinary chat-page runs.
    batch_label: Optional[str] = Field(None, max_length=200)
    # Live View (frontend/index.html's webcam feed): one call per rolling recorded
    # clip, each already exactly one chunk long. `live_session` marks every clip (so
    # the server skips archiving a throwaway recording and tags the record);
    # `continue_record_id` is set on every clip after the first, carrying the record id
    # the first clip's `{done:true, record_id}` event returned, so this clip's chunk is
    # appended to that same history row (history.append_record) instead of starting a
    # new one - a whole session is one growing log, not one row per clip.
    live_session: bool = False
    continue_record_id: Optional[str] = None


class UploadResponse(BaseModel):
    id: str
    kind: Literal["image", "video"]
    name: str
    # Seconds, for videos whose header could be read; None otherwise. The UI uses it to
    # preview what "optimised" would sample.
    duration: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    backend: str
    flash_attn2: bool
    quantize: str


class SettingsResponse(BaseModel):
    """Everything the settings page needs to render itself."""

    # Currently effective defaults (launch defaults + saved overrides).
    defaults: Dict[str, Any]
    # How the server was actually launched, so the UI can show what a reset would restore.
    launch_defaults: Dict[str, Any]
    # Field names that currently differ from launch and are persisted to settings.json.
    overridden: List[str]
    # Control ranges + help text, so the frontend does not hardcode any of it.
    fields: Dict[str, Dict[str, Any]]
    # Read-only, reload-required flags, shown for context.
    locked: Dict[str, Any]


class SettingsUpdate(BaseModel):
    """PUT /api/settings body. Partial - only the fields present are changed."""

    max_new_tokens: Optional[int] = Field(None, ge=16, le=8192)
    segment_seconds: Optional[int] = Field(None, ge=10, le=7200)
    segment_overlap_frames: Optional[int] = Field(None, ge=0, le=240)
    sample_fps: Optional[float] = Field(None, ge=0.1, le=10.0)
    max_video_tokens: Optional[int] = Field(None, ge=256, le=16384)
    motion_threshold: Optional[float] = Field(None, ge=0.0, le=20.0)
    default_prompt: Optional[str] = Field(None, max_length=4000)
