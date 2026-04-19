"""Single source of truth for local CV / coaching pipeline filesystem layout.

Implementation lives in :mod:`services.paths` (split modules). This module re-exports
the public API for backward compatibility with ``from services.pipeline_paths import ...``.
"""

from __future__ import annotations

from services.paths import (
    BACKEND_ROOT,
    GAFFERS_HOMOGRAPHY_JSON_ENV,
    SN_CALIBRATION_RESOURCES_DIR_ENV,
    SN_CALIBRATION_REPO_URL,
    collect_local_cv_pipeline_gaps,
    ensure_core_pipeline_directories,
    format_homography_blocker_detail,
    format_homography_missing_error,
    format_pipeline_prerequisite_errors,
    format_tracking_model_missing_reason,
    output_dir,
    resolve_tracking_homography_json_path,
    sn_calibration_resources_dir,
    sn_calibration_root_dir,
    sn_calibration_vendor_dir,
    tactical_library_path,
    tracking_model_weights_path,
    uploads_dir,
    validate_homography_json_file,
)

__all__ = [
    "BACKEND_ROOT",
    "GAFFERS_HOMOGRAPHY_JSON_ENV",
    "SN_CALIBRATION_RESOURCES_DIR_ENV",
    "SN_CALIBRATION_REPO_URL",
    "collect_local_cv_pipeline_gaps",
    "ensure_core_pipeline_directories",
    "format_homography_blocker_detail",
    "format_homography_missing_error",
    "format_pipeline_prerequisite_errors",
    "format_tracking_model_missing_reason",
    "output_dir",
    "resolve_tracking_homography_json_path",
    "sn_calibration_resources_dir",
    "sn_calibration_root_dir",
    "sn_calibration_vendor_dir",
    "tactical_library_path",
    "tracking_model_weights_path",
    "uploads_dir",
    "validate_homography_json_file",
]
