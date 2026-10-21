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

from .video import needs_browser_transcode, transcode_to_h264

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
    batch_label TEXT,
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
        # Same migration pattern for batch_label (added for Batch Processing): an old DB
        # won't have this column either, since CREATE TABLE IF NOT EXISTS is a no-op.
        if "batch_label" not in cols:
            conn.execute("ALTER TABLE history ADD COLUMN batch_label TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_batch_label ON history(batch_label)")


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

    A source using a codec browsers can't reliably decode (HEVC, most often) is
    re-encoded to H.264 instead of copied as-is - see `video.needs_browser_transcode` -
    so the archived copy actually plays in the analytics dashboard.

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
        transcode = needs_browser_transcode(src_path)
    except Exception:  # noqa: BLE001 - unreadable codec info, just archive as-is
        transcode = False

    if transcode:
        dest = _video_dir / f"{record_id}.mp4"
        try:
            transcode_to_h264(src_path, str(dest))
            return str(dest)
        except Exception:  # noqa: BLE001 - re-encode failed, fall back to a raw copy
            dest.unlink(missing_ok=True)

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
    batch_label: Optional[str] = None,
) -> None:
    """Insert the completed (or partially-completed, on error/disconnect) run.

    `source_video_id` (the upload id) is stored so a later question about the same
    staged video can find and reuse this row's `video_path` instead of re-archiving it -
    see `archive_video`. `batch_label` tags every video in one Batch Processing run with
    a shared value, so they can be filtered back out as a group (see `list_records` /
    `get_stats`) - a batch is not its own stored entity, just a tag on normal rows.
    """
    total_prompt_tokens = sum(int(c.get("stats", {}).get("prompt_tokens") or 0) for c in chunks)
    total_frames = sum(sum(c.get("stats", {}).get("frames") or []) for c in chunks)
    video_expires_at = (
        created_at + _retention_days * 86400 if (video_path and _retention_days > 0) else None
    )
    with _connect() as conn:
        conn.execute(
            "INSERT INTO history (id, created_at, video_name, video_duration, video_path, "
            "video_expires_at, source_video_id, batch_label, prompt, settings_json, "
            "output_json, chunk_count, total_prompt_tokens, total_frames, "
            "processing_seconds, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id, created_at, video_name, video_duration, video_path,
                video_expires_at, source_video_id, batch_label, prompt, json.dumps(settings),
                json.dumps(chunks), len(chunks), total_prompt_tokens, total_frames,
                processing_seconds, status,
            ),
        )


def append_record(
    record_id: str,
    new_chunks: List[Dict[str, Any]],
    video_duration_delta: float,
    processing_seconds_delta: float,
    status: str,
) -> bool:
    """Append one more clip's chunks to an existing run instead of starting a new row -
    a Live View session (frontend/index.html's webcam feed) is one rolling-clip call
    per chunk, and the whole session should read back as one growing log, not one row
    per clip. `new_chunks` are re-indexed to continue from the record's current
    `chunk_count`. `status` only ever escalates to "error", never back to "ok" once set -
    mirrors the one-way `run_status` in main.py's per-chunk loop, so one bad clip in an
    hour-long session doesn't get silently overwritten by the next good one.

    Returns False if `record_id` no longer exists (e.g. deleted from Analytics mid-
    session) so the caller can fall back to starting a fresh row instead of losing the
    clip outright.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT video_duration, output_json, total_prompt_tokens, total_frames, "
            "processing_seconds, status FROM history WHERE id = ?",
            (record_id,),
        ).fetchone()
        if not row:
            return False

        existing_chunks = json.loads(row["output_json"])
        base_index = len(existing_chunks)
        for i, c in enumerate(new_chunks):
            c["index"] = base_index + i
        merged_chunks = existing_chunks + new_chunks

        new_prompt_tokens = sum(int(c.get("stats", {}).get("prompt_tokens") or 0) for c in new_chunks)
        new_frames = sum(sum(c.get("stats", {}).get("frames") or []) for c in new_chunks)
        final_status = "error" if status == "error" or row["status"] == "error" else status

        conn.execute(
            "UPDATE history SET video_duration = ?, output_json = ?, chunk_count = ?, "
            "total_prompt_tokens = ?, total_frames = ?, processing_seconds = ?, status = ? "
            "WHERE id = ?",
            (
                row["video_duration"] + video_duration_delta,
                json.dumps(merged_chunks),
                len(merged_chunks),
                row["total_prompt_tokens"] + new_prompt_tokens,
                row["total_frames"] + new_frames,
                row["processing_seconds"] + processing_seconds_delta,
                final_status,
                record_id,
            ),
        )
    return True


# --- reading -----------------------------------------------------------------------


def _row_summary(row: sqlite3.Row) -> Dict[str, Any]:
    # A stat, not just a truthiness check on the column: several rows can share one
    # archived path (see archive_video's dedup), and an earlier-expiring sibling row can
    # have deleted the file already even though this row's own video_path is still set.
    video_available = bool(row["video_path"]) and Path(row["video_path"]).is_file()
    # checkpoint_path lives inside settings_json (launch-time, not its own column) - lift
    # it out so the history table/export can show which model produced each run without
    # every caller having to parse settings_json itself. None for runs logged before this
    # field existed.
    try:
        checkpoint_path = json.loads(row["settings_json"]).get("checkpoint_path")
    except (TypeError, ValueError, AttributeError):
        checkpoint_path = None
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "video_name": row["video_name"],
        "video_duration": row["video_duration"],
        "video_available": video_available,
        "prompt": row["prompt"],
        "checkpoint_path": checkpoint_path,
        "batch_label": row["batch_label"],
        "chunk_count": row["chunk_count"],
        "total_prompt_tokens": row["total_prompt_tokens"],
        "total_frames": row["total_frames"],
        "processing_seconds": row["processing_seconds"],
        "time_saved_seconds": max(0.0, row["video_duration"] - row["processing_seconds"]),
        "status": row["status"],
    }


def _build_where(
    q: Optional[str], date_from: Optional[float], date_to: Optional[float],
    batch_label: Optional[str] = None,
) -> Tuple[List[str], List[Any]]:
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
    if batch_label:
        where.append("batch_label = ?")
        params.append(batch_label)
    return where, params


def list_records(
    q: Optional[str], date_from: Optional[float], date_to: Optional[float],
    limit: int, offset: int, batch_label: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    where, params = _build_where(q, date_from, date_to, batch_label)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _connect() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM history {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM history {clause} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
    return [_row_summary(r) for r in rows], total


def _batch_summary(batch_label: str, rows: List[sqlite3.Row]) -> Dict[str, Any]:
    """Collapse every row sharing one `batch_label` (one Batch Processing run, see
    batch.js) into a single History entry - the per-video rows are attached as `runs`
    so the UI can expand the group inline instead of listing N near-identical rows for
    what was, from the user's side, one action."""
    runs = [_row_summary(r) for r in rows]
    runs.sort(key=lambda r: r["created_at"])
    prompts = {r["prompt"] for r in runs}
    checkpoints = {r["checkpoint_path"] for r in runs}
    error_count = sum(1 for r in runs if r["status"] != "ok")
    return {
        "is_batch": True,
        "id": None,
        "batch_label": batch_label,
        "created_at": max(r["created_at"] for r in runs),
        "video_name": f"{len(runs)} video{'s' if len(runs) != 1 else ''}",
        "checkpoint_path": checkpoints.pop() if len(checkpoints) == 1 else None,
        "checkpoint_mixed": len(checkpoints) > 1,
        "video_duration": sum(r["video_duration"] for r in runs),
        "prompt": prompts.pop() if len(prompts) == 1 else f"{len(prompts)} different prompts",
        "chunk_count": sum(r["chunk_count"] for r in runs),
        "processing_seconds": sum(r["processing_seconds"] for r in runs),
        "time_saved_seconds": sum(r["time_saved_seconds"] for r in runs),
        "run_count": len(runs),
        "error_count": error_count,
        "status": "ok" if error_count == 0 else "error",
        "runs": runs,
    }


def list_records_grouped(
    q: Optional[str], date_from: Optional[float], date_to: Optional[float],
    limit: int, offset: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Like `list_records`, but every row sharing a `batch_label` collapses into one
    entry (see `_batch_summary`) before pagination is applied - so a 30-video Batch
    Processing run takes one slot in the page/history list instead of 30, and the
    headline table stays readable. Standalone runs (`batch_label IS NULL`) are
    unaffected, one entry each, keyed by their own id so they still group to size 1.

    Pagination has to happen on the *group* count, not the row count, so the group
    keys for the page are resolved first (cheap - just id/batch_label + created_at),
    then only that page's rows are fetched and grouped in Python.
    """
    where, params = _build_where(q, date_from, date_to)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _connect() as conn:
        key_rows = conn.execute(
            f"SELECT COALESCE(batch_label, id) AS gkey, MAX(created_at) AS created_at "
            f"FROM history {clause} GROUP BY gkey ORDER BY created_at DESC",
            params,
        ).fetchall()
        total = len(key_rows)
        page_keys = [r["gkey"] for r in key_rows[offset:offset + limit]]
        if not page_keys:
            return [], total

        placeholders = ",".join("?" * len(page_keys))
        rows = conn.execute(
            f"SELECT * FROM history {clause} "
            f"{'AND' if where else 'WHERE'} COALESCE(batch_label, id) IN ({placeholders}) "
            f"ORDER BY created_at DESC",
            params + page_keys,
        ).fetchall()

    by_key: Dict[str, List[sqlite3.Row]] = {}
    for row in rows:
        by_key.setdefault(row["batch_label"] or row["id"], []).append(row)

    entries = []
    for key in page_keys:
        group_rows = by_key.get(key, [])
        if not group_rows:
            continue
        if group_rows[0]["batch_label"] is None:
            entry = _row_summary(group_rows[0])
            entry["is_batch"] = False
            entries.append(entry)
        else:
            entries.append(_batch_summary(group_rows[0]["batch_label"], group_rows))
    return entries, total


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


def get_stats(
    date_from: Optional[float] = None, date_to: Optional[float] = None,
    batch_label: Optional[str] = None,
) -> Dict[str, Any]:
    where, params = [], []
    if date_from is not None:
        where.append("created_at >= ?")
        params.append(date_from)
    if date_to is not None:
        where.append("created_at <= ?")
        params.append(date_to)
    if batch_label:
        where.append("batch_label = ?")
        params.append(batch_label)
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

# Cell values starting with one of these are formulas to Excel/Sheets/LibreOffice - a
# model-generated answer or a user prompt that happens to start with one would otherwise
# execute as a formula (including legacy DDE command execution) when the export is
# opened. Prefixing with a plain quote makes every spreadsheet app treat it as inert text.
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(_CSV_FORMULA_PREFIXES):
        return "'" + value
    return value


def _sanitize_csv_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _csv_safe(v) for k, v in row.items()}


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
        "id", "created_at", "video_name", "checkpoint_path", "video_duration", "prompt",
        "chunk_count", "total_prompt_tokens", "total_frames", "processing_seconds",
        "time_saved_seconds", "status",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(_sanitize_csv_row(r) for r in rows)
    return buf.getvalue().encode("utf-8"), "text/csv", f"qwen-history-{stamp}.csv"


def _slug(name: str) -> str:
    # ASCII-only: the result is embedded in a Content-Disposition header, which the
    # HTTP stack encodes as latin-1 - a non-Latin-1 character surviving here (CJK,
    # Cyrillic, accented letters all pass str.isalnum()) would raise UnicodeEncodeError
    # and turn the export into a 500 instead of a download.
    keep = [c if (c.isascii() and c.isalnum()) or c in ("-", "_") else "-" for c in name]
    slug = "".join(keep).strip("-")
    return slug[:60] or "video"


def export_record(record_id: str, fmt: str) -> Optional[Tuple[bytes, str, str]]:
    """One run's full summary (prompt, settings, every chunk's output/stats) as a
    downloadable file. Unlike `export_rows` (the bulk history export), this includes the
    per-chunk output - the actual answers, not just the run's aggregate stats."""
    record = get_record(record_id)
    if record is None:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"qwen-run-{_slug(record['video_name'])}-{stamp}"

    if fmt == "json":
        data = json.dumps(record, indent=2).encode("utf-8")
        return data, "application/json", f"{base}.json"

    buf = io.StringIO()
    fields = [
        "id", "video_name", "checkpoint_path", "created_at", "prompt", "chunk_index",
        "start", "end", "text", "prompt_tokens", "frames",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for chunk in record["chunks"]:
        stats = chunk.get("stats") or {}
        writer.writerow(_sanitize_csv_row({
            "id": record["id"],
            "video_name": record["video_name"],
            "checkpoint_path": record["checkpoint_path"],
            "created_at": record["created_at"],
            "prompt": record["prompt"],
            "chunk_index": chunk.get("index"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "text": chunk.get("text", ""),
            "prompt_tokens": stats.get("prompt_tokens"),
            "frames": sum(stats.get("frames") or []) if stats.get("frames") else None,
        }))
    return buf.getvalue().encode("utf-8"), "text/csv", f"{base}.csv"


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
