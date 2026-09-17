# Copyright (c) Alibaba Cloud.
#
# Analytics dashboard API: list/inspect/delete/export past segmented-analysis runs and
# the aggregate stats the dashboard charts. See app.history for the storage layer.
import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from . import history

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("")
async def list_history(
    q: str = "", date_from: Optional[float] = None, date_to: Optional[float] = None,
    limit: int = 50, offset: int = 0,
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    items, total = await asyncio.to_thread(
        history.list_records, q or None, date_from, date_to, limit, offset
    )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/stats")
async def get_stats(date_from: Optional[float] = None, date_to: Optional[float] = None):
    return await asyncio.to_thread(history.get_stats, date_from, date_to)


@router.get("/export/{fmt}")
async def export(fmt: str, q: str = "", date_from: Optional[float] = None, date_to: Optional[float] = None):
    if fmt not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'json'")
    data, media_type, filename = await asyncio.to_thread(
        history.export_rows, fmt, q or None, date_from, date_to
    )
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{record_id}")
async def get_one(record_id: str):
    row = await asyncio.to_thread(history.get_record, record_id)
    if not row:
        raise HTTPException(status_code=404, detail="history record not found")
    return row


@router.delete("/{record_id}")
async def delete_one(record_id: str):
    ok = await asyncio.to_thread(history.delete_record, record_id)
    if not ok:
        raise HTTPException(status_code=404, detail="history record not found")
    return {"deleted": True}


@router.get("/{record_id}/video")
async def get_video(record_id: str):
    path = await asyncio.to_thread(history.get_video_path, record_id)
    if not path:
        raise HTTPException(status_code=404, detail="video not available (expired or never archived)")
    return FileResponse(path)
