"""
Chunked upload router for large video files.

Protocol:
    1. POST /api/v1/upload/init       → get upload_id
    2. POST /api/v1/upload/chunk      → upload one ~10 MB chunk (idempotent)
    3. POST /api/v1/upload/complete   → reassemble + upload to GCS + create pipeline job
    4. GET  /api/v1/upload/{id}/status → check which chunks arrived (for resume)

Design:
    - Chunks are written to /tmp (ephemeral, per-request) with a 4 MB buffer.
    - On complete_upload(), the assembled file is streamed to GCS then deleted
      locally — Cloud Run statelessness is fully respected.
    - Sessions are tracked in-memory and protected by a threading lock.
    - A background task cleans up expired sessions (default 24 h).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Request

from pydantic import BaseModel

from services.observability import track_upload_event

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job-creator callback (injected by main.py to avoid circular / aliased imports)
# ---------------------------------------------------------------------------

# Signature: (job_id, video_path, filename, cv_engine, llm_engine,
#              quality_profile, chunking_interval, gcs_blob_name, user_id) -> None
# gcs_blob_name is the GCS object name for the uploaded video
# (e.g. "uploads/{job_id}.mp4").  Empty string when GCS is disabled.
JobCreatorFn = Callable[
    [str, Path, str, str, str, str, str, str, str],
    Awaitable[None],
]

_job_creator: JobCreatorFn | None = None


def register_job_creator(fn: JobCreatorFn) -> None:
    """Called once by ``main.py`` at startup to inject the job-creation logic.

    This avoids ``from main import _job_store`` which breaks when uvicorn
    loads the app as ``backend.main`` (Python creates two module instances).
    """
    global _job_creator
    _job_creator = fn

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHUNK_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB — client must match this
MAX_TOTAL_SIZE = 12 * 1024 * 1024 * 1024  # 12 GB hard ceiling
SESSION_TTL_SECONDS = 24 * 60 * 60  # 24 hours
COPY_BUFFER = 4 * 1024 * 1024  # 4 MB disk-write buffer

# Use /tmp so chunks never land on the container's overlay filesystem.
# Cloud Run containers have a tmpfs /tmp (up to 512 MB by default on 1st-gen,
# up to 32 GB on 2nd-gen with local SSD).  For videos larger than available
# /tmp, set GCS_ENABLED=false and mount a Cloud Filestore volume instead.
_UPLOAD_ROOT: Path = Path("/tmp") / "gaffer_uploads"


def configure_upload_dir(root: Path) -> None:
    """Allow ``main.py`` to override the uploads temp root at startup.

    In production this should remain /tmp/gaffer_uploads (the default).
    The override exists for local dev environments where a different path
    may be more convenient.
    """
    global _UPLOAD_ROOT
    _UPLOAD_ROOT = root


def _chunks_dir(upload_id: str) -> Path:
    return _UPLOAD_ROOT / "chunks" / upload_id


# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

@dataclass
class UploadSession:
    upload_id: str
    user_id: str
    filename: str
    total_size: int
    total_chunks: int
    created_at: datetime
    received: set[int] = field(default_factory=set)



_sessions: dict[str, UploadSession] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class InitResponse(BaseModel):
    upload_id: str
    chunk_size: int = CHUNK_SIZE_BYTES


class ChunkResponse(BaseModel):
    upload_id: str
    chunk_index: int
    received: int
    total: int
    complete: bool


class CompleteResponse(BaseModel):
    job_id: str
    status: str


class StatusResponse(BaseModel):
    upload_id: str
    filename: str
    total_chunks: int
    received_chunks: list[int]
    complete: bool


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/upload", tags=["chunked-upload"])


@router.post("/init", response_model=InitResponse)
async def init_upload(
    request: Request,
    filename: str = Form(...),
    total_size: int = Form(...),
    total_chunks: int = Form(...),
) -> InitResponse:
    """Start a new chunked upload session."""
    user_id = request.state.user.get("sub", "default-user-id") if hasattr(request.state, "user") else "default-user-id"

    # ── Validate inputs ──────────────────────────────────────────────
    valid_exts = (".mp4", ".mov", ".avi")
    ext = Path(filename).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(valid_exts)} uploads are supported.",
        )

    if total_size <= 0 or total_size > MAX_TOTAL_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size must be between 1 byte and {MAX_TOTAL_SIZE // (1024**3)} GB.",
        )

    if total_chunks <= 0 or total_chunks > 100_000:
        raise HTTPException(
            status_code=400,
            detail="Invalid total_chunks value.",
        )

    # ── Create session ───────────────────────────────────────────────
    upload_id = uuid.uuid4().hex
    session = UploadSession(
        upload_id=upload_id,
        user_id=user_id,
        filename=filename,
        total_size=total_size,
        total_chunks=total_chunks,
        created_at=datetime.now(timezone.utc),
    )

    chunks_path = _chunks_dir(upload_id)
    chunks_path.mkdir(parents=True, exist_ok=True)

    with _lock:
        _sessions[upload_id] = session

    track_upload_event("init", job_id=upload_id, size_bytes=total_size)

    LOGGER.info(
        "Upload session %s created: %s (%d bytes, %d chunks) for User %s",
        upload_id[:8],
        filename,
        total_size,
        total_chunks,
        user_id,
    )

    return InitResponse(upload_id=upload_id)



@router.post("/chunk", response_model=ChunkResponse)
async def upload_chunk(
    upload_id: str = Form(...),
    chunk_index: int = Form(...),
    file: UploadFile = File(...),
) -> ChunkResponse:
    """Upload a single chunk.  Idempotent — resending the same index overwrites."""

    with _lock:
        session = _sessions.get(upload_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found or expired.")

    if chunk_index < 0 or chunk_index >= session.total_chunks:
        raise HTTPException(
            status_code=400,
            detail=f"chunk_index {chunk_index} is out of range [0, {session.total_chunks}).",
        )

    # Write chunk directly to disk with a small buffer.
    chunk_path = _chunks_dir(upload_id) / f"chunk_{chunk_index:06d}.bin"
    try:
        with chunk_path.open("wb") as f_out:
            shutil.copyfileobj(file.file, f_out, length=COPY_BUFFER)
    except Exception as e:
        track_upload_event("failure", job_id=upload_id, chunk_index=chunk_index, failed=True, error_type="chunk_write_error")
        raise
    finally:
        await file.close()

    track_upload_event("chunk", job_id=upload_id, chunk_index=chunk_index)

    with _lock:
        session.received.add(chunk_index)

        received = len(session.received)
        total = session.total_chunks

    return ChunkResponse(
        upload_id=upload_id,
        chunk_index=chunk_index,
        received=received,
        total=total,
        complete=(received == total),
    )


@router.post("/complete", response_model=CompleteResponse)
async def complete_upload(
    request: Request,
    upload_id: str = Form(...),
    cv_engine: str = Form("local"),
    llm_engine: str = Form("local"),
    quality_profile: str = Form("balanced"),
    chunking_interval: str = Form("15-minute intervals"),
) -> CompleteResponse:
    """Reassemble chunks into a single video and kick off the analytics job."""
    user_id = request.state.user.get("sub", "default-user-id") if hasattr(request.state, "user") else "default-user-id"

    with _lock:
        session = _sessions.get(upload_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found or expired.")

    # Scoping security: only the user who initialized the session can complete it
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this upload session.")


    # ── Verify all chunks present ────────────────────────────────────
    missing = [i for i in range(session.total_chunks) if i not in session.received]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Missing {len(missing)} chunk(s).",
                "missing_chunks": missing[:50],  # cap for readability
            },
        )

    # ── Reassemble into /tmp ─────────────────────────────────────────
    ext = Path(session.filename).suffix.lower()
    job_id = uuid.uuid4().hex
    final_path = _UPLOAD_ROOT / f"{job_id}{ext}"
    final_path.parent.mkdir(parents=True, exist_ok=True)

    chunks_path = _chunks_dir(upload_id)
    final_size = 0
    try:
        with final_path.open("wb") as f_out:
            for idx in range(session.total_chunks):
                chunk_file = chunks_path / f"chunk_{idx:06d}.bin"
                with chunk_file.open("rb") as f_in:
                    shutil.copyfileobj(f_in, f_out, length=COPY_BUFFER)
        
        final_size = final_path.stat().st_size
        track_upload_event("complete", job_id=job_id, size_bytes=final_size)

        LOGGER.info(
            "Reassembled %s → %s (%d bytes)",
            upload_id[:8],
            final_path.name,
            final_size,
        )
    except Exception as re_err:
        track_upload_event("failure", job_id=job_id, failed=True, error_type="reassembly_failed")
        # Clean up partial reassembly
        final_path.unlink(missing_ok=True)
        raise
    finally:
        # Chunk directory is always cleaned up immediately after assembly
        shutil.rmtree(chunks_path, ignore_errors=True)

    # ── Upload assembled file to GCS then delete local copy ──────────
    gcs_blob_name: str = ""
    try:
        from services import gcs_service  # local import avoids circular deps
        gcs_blob_name = gcs_service.upload_blob_name(job_id, ext)
        gcs_service.upload_file(final_path, gcs_blob_name, delete_local=True)
        track_upload_event("gcs_sync", job_id=job_id, size_bytes=final_size)
        LOGGER.info("Video uploaded to GCS: %s", gcs_blob_name)
    except Exception as gcs_err:  # noqa: BLE001
        # Non-fatal: if GCS upload fails we keep the local file so the
        # pipeline can still run from /tmp on this same instance.
        track_upload_event("failure", job_id=job_id, failed=True, error_type="gcs_upload_failed")
        LOGGER.warning(
            "GCS upload failed for job %s (%s). Keeping local file for in-process run.",
            job_id, gcs_err,
        )
        gcs_blob_name = ""


    # ── Remove session ───────────────────────────────────────────────
    with _lock:
        _sessions.pop(upload_id, None)

    # ── Create job via injected callback ────────────────────────────────
    if _job_creator is None:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: job creator not registered.",
        )

    await _job_creator(
        job_id,
        final_path,
        session.filename,
        cv_engine,
        llm_engine,
        quality_profile,
        chunking_interval,
        gcs_blob_name,
        session.user_id,
    )



    return CompleteResponse(job_id=job_id, status="pending")


@router.get("/{upload_id}/status", response_model=StatusResponse)
async def upload_status(upload_id: str) -> StatusResponse:
    """Check which chunks have been received (for resume support)."""

    with _lock:
        session = _sessions.get(upload_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found or expired.")

    received_sorted = sorted(session.received)
    return StatusResponse(
        upload_id=upload_id,
        filename=session.filename,
        total_chunks=session.total_chunks,
        received_chunks=received_sorted,
        complete=(len(received_sorted) == session.total_chunks),
    )


# ---------------------------------------------------------------------------
# Background cleanup of expired sessions
# ---------------------------------------------------------------------------

async def cleanup_expired_sessions() -> None:
    """Run periodically to remove stale upload sessions and their chunks."""
    while True:
        await asyncio.sleep(3600)  # every hour
        now = datetime.now(timezone.utc)
        expired_ids: list[str] = []

        with _lock:
            for uid, sess in _sessions.items():
                age = (now - sess.created_at).total_seconds()
                if age > SESSION_TTL_SECONDS:
                    expired_ids.append(uid)

            for uid in expired_ids:
                _sessions.pop(uid, None)

        for uid in expired_ids:
            chunks_path = _chunks_dir(uid)
            shutil.rmtree(chunks_path, ignore_errors=True)
            LOGGER.info("Cleaned up expired upload session %s", uid[:8])
