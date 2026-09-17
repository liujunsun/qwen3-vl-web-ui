# Copyright (c) Alibaba Cloud.
#
# Analytics history: a permanent log of every segmented ("chunk-by-chunk") video
# analysis run - the prompt, the settings used, the per-chunk output, and timing/token
# stats - backing the analytics dashboard (frontend/analytics.html).
#
# Storage is plain SQLite (stdlib, no extra dependency): one row per run. The archived
# copy of the source video is deleted after `retention_days` to bound disk use, but the
# row itself (prompt/settings/output/stats) is kept forever - "time saved" and every
# other aggregate stat stays correct even after videos age out. A fresh sqlite3
# connection is opened per call rather than shared across threads/requests; at the
# request volumes a single-user local server sees this is simple and fast enough, and
# every write already happens inside the app's single generation lock.
import csv
import io
import json
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

DB_FILE_NAME = "history.db"
VIDEO_SUBDIR = "videos"

_db_path: Optional[Path] = None
_video_dir: Optional[Path] = None
_retention_days: int = 30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    video_name TEXT NOT NULL,
    video_duration REAL NOT NULL,
    video_path TEXT,
    video_expires_at REAL,
    source_video_id TEXT,
    prompt TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    output_json TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    total_prompt_tokens INTEGER NOT NULL,
    total_frames INTEGER NOT NULL,
    processing_seconds REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON history(created_at);
"""
# idx_history_source_video_id is created in init(), after the source_video_id migration
# below runs - creating it here would fail on a pre-dedup DB where the column doesn't
# exist yet (CREATE TABLE IF NOT EXISTS is a no-op against an existing table).


def init(history_dir: Path, retention_days: int) -> None:
    """Open (creating if needed) the history DB and video archive folder."""
    global _db_path, _video_dir, _retention_days
    history_dir.mkdir(parents=True, exist_ok=True)
    _video_dir = history_dir / VIDEO_SUBDIR
    _video_dir.mkdir(parents=True, exist_ok=True)
    _db_path = history_dir / DB_FILE_NAME
    _retention_days = max(0, int(retention_days))
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        # Migrate a DB created before source_video_id existed (dedup needs it) - the
        # CREATE TABLE above is a no-op against an existing table, so an old DB won't
        # have this column yet.
        cols = {row["name"] for row in conn.execute("PRAGMA table_info(history)")}
        if "source_video_id" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN source_video_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_source_video_id ON history(source_video_id)")


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """A connection that is actually closed on exit - `with sqlite3.Connection` on its
    own only commits/rolls back, it never releases the file handle."""
    if _db_path is None:
        raise RuntimeError("history.init() was not called")
    conn = sqlite3.connect(str(_db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# --- writing ---------------------------------------------------------------------


def new_id() -> str:
    return uuid.uuid4().hex


def archive_video(record_id: str, src_path: str, source_video_id: Optional[str] = None) -> Optional[str]:
    """Copy the source video into permanent storage. Returns the archive path, or None
    if archiving is disabled (`retention_days == 0`) or the copy fails.

    Asking several questions about the same staged video reuses one upload id
    (`source_video_id`) across several `/api/chat/segmented` calls; without dedup each
    call would archive its own byte-identical copy. If an earlier run already archived
    this `source_video_id` and that file is still on disk, its path is reused instead of
    copying again - storage cost stays at one copy per distinct video, not per question.
    """
    if _retention_days == 0 or _video_dir is None:
        return None

    if source_video_id:
        with _connect() as conn:
            existing = conn.execute(
                "SELECT video_path FROM history WHERE source_video_id = ? "
                "AND video_path IS NOT NULL ORDER BY created_at DESC LIMIT 1",
                (source_video_id,),
            ).fetchone()
        if existing and Path(existing["video_path"]).is_file():
            return existing["video_path"]

    try:
        ext = Path(src_path).suffix
        dest = _video_dir / f"{record_id}{ext}"
        shutil.copy2(src_path, dest)
        return str(dest)
    except OSError:
        return None  # a missing archive copy is not fatal - the record still saves


def finalize_record(
    record_id: str,
    created_at: float,
    video_name: str,
    video_duration: float,
    video_path: Optional[str],
    prompt: str,
    settings: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    processing_seconds: float,
    status: str,
    source_video_id: Optional[str] = None,
) -> None:
    """Insert the completed (or partially-completed, on error/disconnect) run.

    `source_video_id` (the upload id) is stored so a later question about the same
    staged video can find and reuse this row's `video_path` instead of re-archiving it -
    see `archive_video`.
    """
    total_prompt_tokens = sum(int(c.get("stats", {}).get("prompt_tokens") or 0) for c in chunks)
    total_frames = sum(sum(c.get("stats", {}).get("frames") or []) for c in chunks)
    video_expires_at = (
        created_at + _retention_days * 86400 if (video_path and _retention_days > 0) else None
    )
    with _connect() as conn:
        conn.execute(
            "INSERT INTO history (id, created_at, video_name, video_duration, video_path, "
            "video_expires_at, source_video_id, prompt, settings_json, output_json, "
            "chunk_count, total_prompt_tokens, total_frames, processing_seconds, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id, created_at, video_name, video_duration, video_path,
                video_expires_at, source_video_id, prompt, json.dumps(settings),
                json.dumps(chunks), len(chunks), total_prompt_tokens, total_frames,
                processing_seconds, status,
            ),
        )


# --- reading -----------------------------------------------------------------------


def _row_summary(row: sqlite3.Row) -> Dict[str, Any]:
    # A stat, not just a truthiness check on the column: several rows can share one
    # archived path (see archive_video's dedup), and an earlier-expiring sibling row can
    # have deleted the file already even though this row's own video_path is still set.
    video_available = bool(row["video_path"]) and Path(row["video_path"]).is_file()
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "video_name": row["video_name"],
        "video_duration": row["video_duration"],
        "video_available": video_available,
        "prompt": row["prompt"],
        "chunk_count": row["chunk_count"],
        "total_prompt_tokens": row["total_prompt_tokens"],
        "total_frames": row["total_frames"],
        "processing_seconds": row["processing_seconds"],
        "time_saved_seconds": max(0.0, row["video_duration"] - row["processing_seconds"]),
        "status": row["status"],
    }


def list_records(
    q: Optional[str], date_from: Optional[float], date_to: Optional[float],
    limit: int, offset: int,
) -> Tuple[List[Dict[str, Any]], int]:
    where, params = [], []
    if q:
        where.append("(prompt LIKE ? OR video_name LIKE ?)")
        params += [f"%{q}%", f"%{q}%"]
    if date_from is not None:
        where.append("created_at >= ?")
        params.append(date_from)
    if date_to is not None:
        where.append("created_at <= ?")
        params.append(date_to)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM history {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM history {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return [_row_summary(r) for r in rows], total


def get_record(record_id: str) -> Optional[Dict[str, Any]]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM history WHERE id = ?", (record_id,)).fetchone()
    if not row:
        return None
    out = _row_summary(row)
    out["settings"] = json.loads(row["settings_json"])
    out["chunks"] = json.loads(row["output_json"])
    return out


def get_video_path(record_id: str) -> Optional[str]:
    with _connect() as conn:
        row = conn.execute("SELECT video_path FROM history WHERE id = ?", (record_id,)).fetchone()
    if not row or not row["video_path"]:
        return None
    p = Path(row["video_path"])
    return str(p) if p.is_file() else None


def delete_record(record_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT video_path FROM history WHERE id = ?", (record_id,)).fetchone()
        if not row:
            return False
        if row["video_path"]:
            # Other rows can share this exact path (archive_video's dedup) - only the
            # last one standing actually removes the file.
            sibling = conn.execute(
                "SELECT 1 FROM history WHERE video_path = ? AND id != ? LIMIT 1",
                (row["video_path"], record_id),
            ).fetchone()
            if not sibling:
                try:
                    Path(row["video_path"]).unlink(missing_ok=True)
                except OSError:
                    pass
        conn.execute("DELETE FROM history WHERE id = ?", (record_id,))
    return True


# --- aggregate stats for the dashboard ----------------------------------------------


def get_stats(date_from: Optional[float] = None, date_to: Optional[float] = None) -> Dict[str, Any]:
    where, params = [], []
    if date_from is not None:
        where.append("created_at >= ?")
        params.append(date_from)
    if date_to is not None:
        where.append("created_at <= ?")
        params.append(date_to)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _connect() as conn:
        totals = conn.execute(
            f"""SELECT
                    COUNT(*) AS runs,
                    COALESCE(SUM(video_duration), 0) AS video_seconds,
                    COALESCE(SUM(processing_seconds), 0) AS processing_seconds,
                    COALESCE(SUM(MAX(0, video_duration - processing_seconds)), 0) AS time_saved_seconds,
                    COALESCE(SUM(chunk_count), 0) AS chunks,
                    COALESCE(SUM(total_prompt_tokens), 0) AS prompt_tokens,
                    COALESCE(SUM(total_frames), 0) AS frames,
                    COALESCE(SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END), 0) AS errors
                FROM history {clause}""",
            params,
        ).fetchone()

        daily = conn.execute(
            f"""SELECT
                    date(created_at, 'unixepoch') AS day,
                    COUNT(*) AS runs,
                    COALESCE(SUM(MAX(0, video_duration - processing_seconds)), 0) AS time_saved_seconds,
                    COALESCE(SUM(total_prompt_tokens), 0) AS prompt_tokens
                FROM history {clause}
                GROUP BY day ORDER BY day ASC""",
            params,
        ).fetchall()

        durations = conn.execute(f"SELECT video_duration FROM history {clause}", params).fetchall()

        # answer_style lives inside settings_json (it's a per-request choice, not its
        # own column) - json_extract pulls it out for the GROUP BY; rows saved before
        # this field existed fall back to "auto", same as the server does at request time.
        style_rows = conn.execute(
            f"""SELECT COALESCE(json_extract(settings_json, '$.answer_style'), 'auto') AS style,
                    COUNT(*) AS n
                FROM history {clause}
                GROUP BY style""",
            params,
        ).fetchall()

        # One point per run for the duration-vs-processing-time scatter. Capped and
        # most-recent-first so a long history doesn't balloon the response.
        efficiency_rows = conn.execute(
            f"""SELECT video_duration, processing_seconds, status FROM history {clause}
                ORDER BY created_at DESC LIMIT 500""",
            params,
        ).fetchall()

    # Bucket clip durations into a small fixed histogram for a distribution chart.
    buckets = [("<1m", 60), ("1-5m", 300), ("5-15m", 900), ("15-60m", 3600), ("1h+", None)]
    hist = {label: 0 for label, _ in buckets}
    for (d,) in durations:
        for label, cap in buckets:
            if cap is None or d < cap:
                hist[label] += 1
                break

    answer_style_breakdown = {"auto": 0, "concise": 0, "detailed": 0}
    for row in style_rows:
        if row["style"] in answer_style_breakdown:
            answer_style_breakdown[row["style"]] = row["n"]

    return {
        "totals": {
            "runs": totals["runs"],
            "video_seconds": totals["video_seconds"],
            "processing_seconds": totals["processing_seconds"],
            "time_saved_seconds": totals["time_saved_seconds"],
            "chunks": totals["chunks"],
            "prompt_tokens": totals["prompt_tokens"],
            "frames": totals["frames"],
            "errors": totals["errors"],
        },
        "daily": [dict(row) for row in daily],
        "duration_histogram": hist,
        "answer_style_breakdown": answer_style_breakdown,
        "efficiency_points": [
            {"duration": r["video_duration"], "processing_seconds": r["processing_seconds"],
             "status": r["status"]}
            for r in efficiency_rows
        ],
    }


# --- export --------------------------------------------------------------------------


def export_rows(
    fmt: str, q: Optional[str], date_from: Optional[float], date_to: Optional[float]
) -> Tuple[bytes, str, str]:
    rows, _ = list_records(q, date_from, date_to, limit=1_000_000, offset=0)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if fmt == "json":
        data = json.dumps(rows, indent=2).encode("utf-8")
        return data, "application/json", f"qwen-history-{stamp}.json"

    buf = io.StringIO()
    fields = [
        "id", "created_at", "video_name", "video_duration", "prompt", "chunk_count",
        "total_prompt_tokens", "total_frames", "processing_seconds", "time_saved_seconds",
        "status",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8"), "text/csv", f"qwen-history-{stamp}.csv"


# --- retention -----------------------------------------------------------------------


def prune_expired() -> int:
    """Delete archived video files whose retention window has passed. The history row
    (prompt/settings/output/stats) is untouched, so aggregate stats stay correct."""
    now = time.time()
    removed = 0
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, video_path FROM history WHERE video_path IS NOT NULL "
            "AND video_expires_at IS NOT NULL AND video_expires_at < ?",
            (now,),
        ).fetchall()
        for row in rows:
            # Other rows can share this exact path (archive_video's dedup) - only
            # physically delete once no other row still claims it, expired or not; that
            # sibling's own prune pass will remove the file once it, too, expires.
            sibling = conn.execute(
                "SELECT 1 FROM history WHERE video_path = ? AND id != ? LIMIT 1",
                (row["video_path"], row["id"]),
            ).fetchone()
            if not sibling:
                try:
                    Path(row["video_path"]).unlink(missing_ok=True)
                except OSError:
                    pass
            conn.execute(
                "UPDATE history SET video_path = NULL, video_expires_at = NULL WHERE id = ?",
                (row["id"],),
            )
            removed += 1
    return removed
