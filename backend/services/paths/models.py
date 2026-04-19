"""Model weights, tactical library, and I/O directories."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .calibration import sn_calibration_resources_dir
from .constants import BACKEND_ROOT
from .homography import (
    GAFFERS_HOMOGRAPHY_JSON_ENV,
    format_homography_missing_error,
    resolve_tracking_homography_json_path,
    validate_homography_json_file,
)


def tracking_model_weights_path() -> Path:
    """
    YOLO weights for ``run_e2e_cloud`` / ``track_teams``.

    Default: ``backend/models/pretrained/best.pt``. Override with absolute path or path
    relative to ``BACKEND_ROOT`` via ``TRACKING_MODEL_PATH``.
    """
    raw = os.getenv("TRACKING_MODEL_PATH", "").strip()
    if not raw:
        return (BACKEND_ROOT / "models" / "pretrained" / "best.pt").resolve()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (BACKEND_ROOT / candidate).resolve()


def tactical_library_path() -> Path:
    return BACKEND_ROOT / "data" / "tactical_library.json"


def uploads_dir() -> Path:
    return BACKEND_ROOT / "data" / "uploads"


def output_dir() -> Path:
    return BACKEND_ROOT / "output"


def format_tracking_model_missing_reason(path: Path) -> str:
    """Human-readable hint when ``path`` is not a readable weights file."""
    if path.is_file():
        return ""
    if path.exists() and not path.is_file():
        return (
            "Path exists but is not a regular file (e.g. directory or broken symlink). "
            "Remove the wrong entry or fix the symlink."
        )
    parent = path.parent
    parts: list[str] = []
    if parent.is_dir():
        try:
            names = sorted(x.name for x in parent.iterdir() if x.is_file())[:12]
            if names:
                parts.append(f"Files present in {parent.name}/: {names!r}")
            else:
                parts.append(f"Directory exists but has no files: {parent}")
        except OSError as exc:
            parts.append(f"Cannot list {parent}: {exc}")
    else:
        parts.append(f"Parent directory missing: {parent}")
    parts.append(
        "Set TRACKING_MODEL_PATH to the absolute path of your .pt file, "
        "or place best.pt exactly at backend/models/pretrained/best.pt (not project root models/)."
    )
    return " ".join(parts)


def ensure_core_pipeline_directories() -> None:
    """Create layout dirs so weights can be dropped in and artifacts can be written."""
    for rel in (
        "models/pretrained",
        "models/calibration/sn-calibration/resources",
        "output",
        "data/uploads",
        "data",
    ):
        (BACKEND_ROOT / rel).mkdir(parents=True, exist_ok=True)


def collect_local_cv_pipeline_gaps(*, video_path: Path | None = None) -> list[str]:
    """
    Return a list of blocking issues before starting an expensive local CV job.

    Each string is one line suitable for logs or ``FileNotFoundError`` messages.
    """
    gaps: list[str] = []
    weights = tracking_model_weights_path()
    if not weights.is_file():
        gaps.append(
            f"[weights] Missing YOLO weights at {weights}. {format_tracking_model_missing_reason(weights)}"
        )

    lib = tactical_library_path()
    if not lib.is_file():
        gaps.append(
            f"[rag] Missing tactical library JSON at {lib}. "
            "Add backend/data/tactical_library.json (RAG synthesizer requires it)."
        )
    else:
        try:
            with lib.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "philosophies" not in payload:
                gaps.append(
                    f"[rag] Invalid tactical_library.json at {lib}: "
                    "expected top-level object with key 'philosophies' (non-empty list)."
                )
            elif not isinstance(payload.get("philosophies"), list):
                gaps.append(f"[rag] tactical_library.json 'philosophies' must be a list at {lib}.")
            elif len(payload["philosophies"]) == 0:
                gaps.append(f"[rag] tactical_library.json has empty philosophies at {lib}.")
        except json.JSONDecodeError as exc:
            gaps.append(f"[rag] tactical_library.json is not valid JSON at {lib}: {exc}")

    if video_path is not None:
        vp = video_path.resolve()
        if not vp.is_file():
            gaps.append(f"[video] Input video is not a file: {vp}")
        else:
            homography_json = resolve_tracking_homography_json_path(vp)
            if not homography_json.is_file():
                env_set = bool(os.getenv(GAFFERS_HOMOGRAPHY_JSON_ENV, "").strip())
                can_auto_calibrate = sn_calibration_resources_dir().is_dir() and not env_set
                if not can_auto_calibrate:
                    gaps.append(
                        f"[homography] {format_homography_missing_error(vp, homography_json)}"
                    )
            else:
                gap = validate_homography_json_file(homography_json)
                if gap:
                    gaps.append(f"[homography] {gap}")

    return gaps


def format_pipeline_prerequisite_errors(gaps: list[str]) -> str:
    if not gaps:
        return ""
    return "Local CV pipeline prerequisites failed:\n" + "\n".join(gaps)
