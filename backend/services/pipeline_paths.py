"""Single source of truth for local CV / coaching pipeline filesystem layout."""

from __future__ import annotations

import json
import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]

GAFFERS_HOMOGRAPHY_JSON_ENV = "GAFFERS_HOMOGRAPHY_JSON"
SN_CALIBRATION_REPO_URL = "https://github.com/SoccerNet/sn-calibration"


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


def sn_calibration_root_dir() -> Path:
    """SoccerNet sn-calibration checkout (Python package lives here; see dynamic_homography)."""
    return BACKEND_ROOT / "references" / "sn-calibration"


def sn_calibration_resources_dir() -> Path:
    """SoccerNet DynamicPitchCalibrator weights (directory must exist to auto-run homography)."""
    return sn_calibration_root_dir() / "resources"


def resolve_tracking_homography_json_path(video_path: Path) -> Path:
    """
    Path to the homography JSON used by TacticalRadar for this video.

    If ``GAFFERS_HOMOGRAPHY_JSON`` is set, that file wins. Otherwise:
    ``backend/output/{video_path.stem}_homographies.json`` (same convention as
    ``run_calibrator_on_video``).
    """
    raw = os.getenv(GAFFERS_HOMOGRAPHY_JSON_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (output_dir() / f"{video_path.stem}_homographies.json").resolve()


def format_homography_blocker_detail(
    video_path: Path,
    expected_json: Path,
    *,
    env_homography_set: bool,
) -> str:
    """Explain why homography JSON is missing when auto-calibration cannot finish the job."""
    root = sn_calibration_root_dir()
    res = sn_calibration_resources_dir()
    if env_homography_set:
        return (
            f"{GAFFERS_HOMOGRAPHY_JSON_ENV} is set; resolved path {expected_json} is not a usable file. "
            "Unset it to use per-upload backend/output/{stem}_homographies.json with auto-calibration, "
            "or set it to an existing homographies JSON path."
        )
    if not root.is_dir():
        return (
            f"Missing SoccerNet checkout at {root}. Clone {SN_CALIBRATION_REPO_URL} into that directory "
            f"(see Development_Protocol.md CV homography section), then populate {res} per upstream README."
        )
    if not res.is_dir():
        return (
            f"SoccerNet checkout exists at {root} but weights directory is missing: {res}. "
            "Follow the sn-calibration README to install resources/ (calibration weights)."
        )
    return (
        f"Auto-calibration did not produce a file. From backend: "
        f"python scripts/run_calibrator_on_video.py --video {video_path} "
        f"--output {expected_json}. Or set {GAFFERS_HOMOGRAPHY_JSON_ENV} to an existing homographies JSON."
    )


def format_homography_missing_error(video_path: Path, expected_json: Path) -> str:
    env_set = bool(os.getenv(GAFFERS_HOMOGRAPHY_JSON_ENV, "").strip())
    detail = format_homography_blocker_detail(
        video_path, expected_json, env_homography_set=env_set
    )
    return f"Homography JSON not found at {expected_json}. {detail}"


def validate_homography_json_file(path: Path) -> str:
    """
    Return a single-line gap message if ``path`` is not a valid non-empty homography artifact, else "".
    """
    if not path.is_file():
        return f"Homography path is not a file: {path}"
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return f"Homography JSON invalid at {path}: {exc}"
    homographies = data.get("homographies")
    if not isinstance(homographies, list) or len(homographies) == 0:
        return (
            f"Homography JSON at {path} has no usable 'homographies' entries. "
            "Re-run scripts/run_calibrator_on_video.py on this match or fix the file."
        )
    first = homographies[0]
    if not isinstance(first, dict) or "homography" not in first:
        return (
            f"Homography JSON at {path} has invalid homographies[0] "
            "(expected dict with 'homography' matrix)."
        )
    return ""


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
    for rel in ("models/pretrained", "output", "data/uploads", "data"):
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
                gaps.append(
                    f"[rag] tactical_library.json 'philosophies' must be a list at {lib}."
                )
            elif len(payload["philosophies"]) == 0:
                gaps.append(
                    f"[rag] tactical_library.json has empty philosophies at {lib}."
                )
        except json.JSONDecodeError as exc:
            gaps.append(
                f"[rag] tactical_library.json is not valid JSON at {lib}: {exc}"
            )

    if video_path is not None:
        vp = video_path.resolve()
        if not vp.is_file():
            gaps.append(f"[video] Input video is not a file: {vp}")
        else:
            homography_json = resolve_tracking_homography_json_path(vp)
            if not homography_json.is_file():
                env_set = bool(os.getenv(GAFFERS_HOMOGRAPHY_JSON_ENV, "").strip())
                can_auto_calibrate = (
                    sn_calibration_resources_dir().is_dir() and not env_set
                )
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
