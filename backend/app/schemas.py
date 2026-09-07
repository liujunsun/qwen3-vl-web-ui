# Copyright (c) Alibaba Cloud.
#
# Request / response models for the JSON API.
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ContentPart(BaseModel):
    type: Literal["text", "image", "video"]
    text: Optional[str] = None
    # id returned by POST /api/upload (for image / video parts)
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
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0, le=200)
    video_fps: Optional[float] = Field(None, gt=0.0, le=8.0)
    video_max_frames: Optional[int] = Field(None, ge=4, le=768)


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    options: Optional[GenerationOptions] = None
    # Kept for backwards compatibility with the original API; `options.max_new_tokens`
    # wins when both are sent.
    max_new_tokens: Optional[int] = None


class UploadResponse(BaseModel):
    id: str
    kind: Literal["image", "video"]
    name: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    backend: str
    flash_attn2: bool


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
    presets: Dict[str, Dict[str, Any]]
    # Read-only, reload-required flags, shown for context.
    locked: Dict[str, Any]
    # Video patch geometry, so the settings page can predict visual-token cost exactly
    # instead of hardcoding numbers that would drift if the checkpoint changed.
    geometry: Dict[str, Any]


class SettingsUpdate(BaseModel):
    """PUT /api/settings body. Partial - only the fields present are changed."""

    max_new_tokens: Optional[int] = Field(None, ge=16, le=8192)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, gt=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=0, le=200)
    video_fps: Optional[float] = Field(None, gt=0.0, le=8.0)
    video_max_frames: Optional[int] = Field(None, ge=4, le=768)
