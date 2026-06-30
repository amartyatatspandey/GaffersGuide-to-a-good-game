"""
GCS Service — single abstraction layer for all Google Cloud Storage operations.

Design principles:
- GCS_ENABLED=false → fall back to local /tmp directories (safe for local dev without GCS creds).
- GCS_ENABLED=true  → all operations go to gs://{GCS_BUCKET_NAME}/ (production).
- Authentication is exclusively via Application Default Credentials (ADC).
  On Cloud Run this resolves to the service account's Workload Identity automatically.
  No key files. No GOOGLE_APPLICATION_CREDENTIALS. No secrets in code.
- All public methods have identical signatures in both modes so callers don't branch.
- Prefixes (uploads/, reports/, exports/) are configurable via env vars.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Generator, Optional

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — read once at import time
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes")


GCS_ENABLED: bool = _env_bool("GCS_ENABLED", True)
GCS_BUCKET_NAME: str = os.getenv("GCS_BUCKET_NAME", "gaffers-guide")
GCS_UPLOAD_PREFIX: str = os.getenv("GCS_UPLOAD_PREFIX", "uploads")
GCS_REPORTS_PREFIX: str = os.getenv("GCS_REPORTS_PREFIX", "reports")
GCS_EXPORTS_PREFIX: str = os.getenv("GCS_EXPORTS_PREFIX", "exports")

# Local fallback root — used only when GCS_ENABLED=false
_LOCAL_ROOT: Path = Path(tempfile.gettempdir()) / "gaffer_gcs_fallback"


# ---------------------------------------------------------------------------
# Lazy GCS client — instantiated on first use to avoid import-time errors
# when google-cloud-storage is not installed (test environments without GCS).
# ---------------------------------------------------------------------------

_gcs_client = None


def _get_client():
    """Return the GCS client, lazily initialised."""
    global _gcs_client
    if _gcs_client is None:
        try:
            from google.cloud import storage as _storage  # type: ignore[import]
            _gcs_client = _storage.Client()
            LOGGER.info("GCS client initialised (bucket=%s)", GCS_BUCKET_NAME)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Failed to initialise GCS client: %s", exc)
            raise
    return _gcs_client


def _get_bucket():
    return _get_client().bucket(GCS_BUCKET_NAME)


# ---------------------------------------------------------------------------
# Local fallback helpers
# ---------------------------------------------------------------------------

def _local_path(blob_name: str) -> Path:
    """Resolve a GCS blob name to a local fallback path."""
    p = (_LOCAL_ROOT / blob_name).resolve()
    # Safety: never escape the fallback root
    if not str(p).startswith(str(_LOCAL_ROOT)):
        raise ValueError(f"Unsafe blob name traversal: {blob_name!r}")
    return p


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upload_file(local_path: Path, blob_name: str, *, delete_local: bool = False) -> str:
    """
    Upload a local file to GCS (or copy to local fallback).

    Returns the GCS URI: gs://{bucket}/{blob_name}
    or the local fallback path string when GCS_ENABLED=false.

    Args:
        local_path:    Absolute path to the source file.
        blob_name:     Destination object name inside the bucket.
        delete_local:  If True, delete local_path after a successful upload.
    """
    if not GCS_ENABLED:
        dest = _local_path(blob_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(local_path), str(dest))
        LOGGER.debug("[GCS-LOCAL] Copied %s → %s", local_path, dest)
        if delete_local:
            local_path.unlink(missing_ok=True)
        return str(dest)

    blob = _get_bucket().blob(blob_name)
    LOGGER.info("Uploading %s → gs://%s/%s", local_path, GCS_BUCKET_NAME, blob_name)
    blob.upload_from_filename(str(local_path))
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    LOGGER.info("Upload complete: %s", uri)
    if delete_local:
        local_path.unlink(missing_ok=True)
        LOGGER.debug("Deleted local temp file %s", local_path)
    return uri


def download_file(blob_name: str, dest_path: Path) -> Path:
    """
    Download a GCS object to a local path.

    Returns dest_path.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if not GCS_ENABLED:
        src = _local_path(blob_name)
        if not src.exists():
            raise FileNotFoundError(f"[GCS-LOCAL] File not found: {src}")
        shutil.copy2(str(src), str(dest_path))
        LOGGER.debug("[GCS-LOCAL] Copied %s → %s", src, dest_path)
        return dest_path

    blob = _get_bucket().blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"GCS object not found: gs://{GCS_BUCKET_NAME}/{blob_name}")
    LOGGER.info("Downloading gs://%s/%s → %s", GCS_BUCKET_NAME, blob_name, dest_path)
    blob.download_to_filename(str(dest_path))
    return dest_path


def write_json(blob_name: str, data: Any) -> str:
    """
    Serialise ``data`` as JSON and write it to GCS (or local fallback).

    Returns the GCS URI or local path string.
    """
    raw = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")

    if not GCS_ENABLED:
        dest = _local_path(blob_name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(raw)
        LOGGER.debug("[GCS-LOCAL] Wrote JSON %s (%d bytes)", dest, len(raw))
        return str(dest)

    blob = _get_bucket().blob(blob_name)
    blob.upload_from_string(raw, content_type="application/json")
    uri = f"gs://{GCS_BUCKET_NAME}/{blob_name}"
    LOGGER.debug("Wrote JSON → %s (%d bytes)", uri, len(raw))
    return uri


def read_json(blob_name: str) -> Optional[Any]:
    """
    Download and parse a JSON object from GCS.

    Returns the parsed object, or None if the object does not exist.
    """
    if not GCS_ENABLED:
        p = _local_path(blob_name)
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    blob = _get_bucket().blob(blob_name)
    if not blob.exists():
        return None
    raw = blob.download_as_bytes()
    return json.loads(raw.decode("utf-8"))


def list_blobs(prefix: str) -> list[str]:
    """
    List all blob names under ``prefix``.

    Returns a list of blob name strings (not full URIs).
    """
    if not GCS_ENABLED:
        base = _local_path(prefix)
        if not base.exists():
            return []
        return [
            str(p.relative_to(_LOCAL_ROOT))
            for p in base.rglob("*")
            if p.is_file()
        ]

    client = _get_client()
    return [b.name for b in client.list_blobs(GCS_BUCKET_NAME, prefix=prefix)]


def delete_blob(blob_name: str) -> bool:
    """
    Delete a GCS object (or local fallback file).

    Returns True if deleted, False if it did not exist.
    """
    if not GCS_ENABLED:
        p = _local_path(blob_name)
        if p.exists():
            p.unlink()
            return True
        return False

    blob = _get_bucket().blob(blob_name)
    if not blob.exists():
        return False
    blob.delete()
    LOGGER.info("Deleted gs://%s/%s", GCS_BUCKET_NAME, blob_name)
    return True


def blob_exists(blob_name: str) -> bool:
    """Return True if the blob exists in GCS (or local fallback)."""
    if not GCS_ENABLED:
        return _local_path(blob_name).exists()
    return _get_bucket().blob(blob_name).exists()


def generate_signed_url(blob_name: str, expiration_seconds: int = 3600) -> str:
    """
    Generate a signed URL for temporary, direct client-side access to a GCS object.

    NOTE: This requires the Cloud Run service account to have the
    ``roles/iam.serviceAccountTokenCreator`` role (needed for self-signing).

    Falls back to returning a placeholder in local mode.
    """
    if not GCS_ENABLED:
        return f"file://{_local_path(blob_name)}"

    import datetime  # noqa: PLC0415
    blob = _get_bucket().blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(seconds=expiration_seconds),
        method="GET",
    )
    return url


# ---------------------------------------------------------------------------
# Convenience prefix helpers — return the full blob name for a given ID
# ---------------------------------------------------------------------------

def upload_blob_name(job_id: str, ext: str = ".mp4") -> str:
    """Return the GCS blob name for an upload: uploads/{job_id}.mp4"""
    return f"{GCS_UPLOAD_PREFIX}/{job_id}{ext}"


def report_blob_name(filename: str) -> str:
    """Return the GCS blob name for a report JSON: reports/{filename}"""
    return f"{GCS_REPORTS_PREFIX}/{filename}"


def export_blob_name(job_id: str, suffix: str) -> str:
    """Return the GCS blob name for a pipeline export artifact: exports/{job_id}/{suffix}"""
    return f"{GCS_EXPORTS_PREFIX}/{job_id}/{suffix}"
