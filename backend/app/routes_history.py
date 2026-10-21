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
    limit: int = 50, offset: int = 0, batch: Optional[str] = None,
):
    limit = max(1, min(limit, 500))
    offset = max(0, offset)
    # Already drilled into one batch (the Batch Processing page's "View every run from
    # this batch" link, or the batch filter box) -> a flat list of its runs, same as
    # before. Otherwise, group each batch's runs into one collapsible entry so a
    # 30-video batch doesn't swamp the table - see history.list_records_grouped.
    if batch:
        items, total = await asyncio.to_thread(
            history.list_records, q or None, date_from, date_to, limit, offset, batch
        )
    else:
        items, total = await asyncio.to_thread(
            history.list_records_grouped, q or None, date_from, date_to, limit, offset
        )
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/stats")
async def get_stats(
    date_from: Optional[float] = None, date_to: Optional[float] = None,
    batch: Optional[str] = None,
):
    return await asyncio.to_thread(history.get_stats, date_from, date_to, batch or None)


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


@router.get("/{record_id}/export/{fmt}")
async def export_one(record_id: str, fmt: str):
    if fmt not in ("csv", "json"):
        raise HTTPException(status_code=400, detail="format must be 'csv' or 'json'")
    result = await asyncio.to_thread(history.export_record, record_id, fmt)
    if result is None:
        raise HTTPException(status_code=404, detail="history record not found")
    data, media_type, filename = result
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{record_id}/video")
async def get_video(record_id: str):
    path = await asyncio.to_thread(history.get_video_path, record_id)
    if not path:
        raise HTTPException(status_code=404, detail="video not available (expired or never archived)")
    return FileResponse(path)
