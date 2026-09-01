# Copyright (c) Alibaba Cloud.
#
# Request / response models for the JSON API.
from typing import List, Literal, Optional

from pydantic import BaseModel


class ContentPart(BaseModel):
    type: Literal["text", "image", "video"]
    text: Optional[str] = None
    # id returned by POST /api/upload (for image / video parts)
    id: Optional[str] = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: List[ContentPart]


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
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
