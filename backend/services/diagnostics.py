"""
services/diagnostics.py — Structured event logging for Gaffer's Guide
======================================================================

Replaces the previous file-handler implementation (logs/diagnostics.log)
which was ephemeral on Cloud Run.  All events now emit to stdout as
structured JSON (or human-readable text in local dev) via the root logger
configured by services.observability.configure_structured_logging().

Public API is unchanged — all existing call sites in main.py keep working:
  log_event(category, message, data)
  log_error(category, error, context)
  audit_system_imports()
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger("gaffer.diagnostics")


def log_event(category: str, message: str, data: Any = None) -> None:
    """Log a structured system event.

    Args:
        category: Short uppercase label e.g. "JOB_CREATED", "GCS_SYNC".
        message:  Human-readable description.
        data:     Optional dict of safe, non-sensitive context fields.
                  Keys matching sensitive patterns are auto-redacted by
                  the StructuredJSONFormatter.
    """
    extra: dict[str, Any] = {"event_category": category}
    if isinstance(data, dict):
        # Flatten safe fields into extra so they appear as top-level JSON keys
        extra.update(data)
    elif data is not None:
        extra["data"] = str(data)

    LOGGER.info("[%s] %s", category, message, extra=extra)


def log_error(category: str, error: Exception, context: str = "") -> None:
    """Log a structured error with full traceback.

    Args:
        category: Short uppercase label e.g. "DB_ERROR", "PIPELINE_FAIL".
        error:    The caught exception.
        context:  Optional free-text describing where the error occurred.
    """
    LOGGER.error(
        "[%s] FAILURE | context=%s | error=%s",
        category,
        context or "unspecified",
        type(error).__name__,
        exc_info=True,
        extra={"event_category": category, "context": context},
    )


def audit_system_imports() -> list[str]:
    """Check that critical system dependencies are importable.

    Returns:
        List of module names that failed to import (empty = all OK).
    """
    modules = [
        "ultralytics",
        "supervision",
        "cv2",
        "numpy",
        "openai",
        "pydantic",
        "fastapi",
    ]
    missing: list[str] = []
    for m in modules:
        try:
            __import__(m)
            log_event("AUDIT", f"Module {m} verified.")
        except ImportError as exc:
            missing.append(m)
            log_error("AUDIT", exc, context=f"import {m}")
    return missing
