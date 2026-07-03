"""
services/observability.py — Production Observability for Gaffer's Guide
=======================================================================

Provides:
  - PipelineMetricsRegistry   (unchanged — in-memory p50/p95 timers & counters)
  - StructuredJSONFormatter   Cloud Logging-compatible JSON log formatter
  - configure_structured_logging()  Call once at app startup
  - get_correlation_id() / set_correlation_id()  Per-request ID via contextvars
  - track_gemini_call()       Context manager: records Gemini latency & failures
  - track_upload_event()      Fire-and-forget upload phase logger
  - GeminiFailureMonitor      Rolling-window Gemini failure rate (used by /health)
  - UploadFailureMonitor      Rolling-window upload failure rate (used by /health)

Design rules:
  - No business logic lives here.
  - No API keys, secrets, email addresses, or prompt text are ever logged.
  - User identity is logged as a SHA-256-prefixed hash, never raw.
  - On Cloud Run (K_SERVICE set): emits compact JSON to stdout.
  - In local dev: emits human-readable coloured text to stdout.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Generator

# ── Correlation ID context variable ─────────────────────────────────────────
# One ContextVar per asyncio task / thread — no global mutable state needed.
_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


def get_correlation_id() -> str:
    """Return the correlation ID for the current request/task context."""
    return _correlation_id_var.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current request/task context."""
    _correlation_id_var.set(cid)


# ── Sensitive key scrubber ───────────────────────────────────────────────────
_SENSITIVE_KEYS = frozenset({
    "key", "secret", "token", "password", "api_key", "apikey",
    "authorization", "auth", "credential", "private", "jwt",
})


def _scrub(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact values whose key name suggests sensitivity."""
    cleaned: dict[str, Any] = {}
    for k, v in data.items():
        if any(s in k.lower() for s in _SENSITIVE_KEYS):
            cleaned[k] = "[REDACTED]"
        elif isinstance(v, dict):
            cleaned[k] = _scrub(v)
        else:
            cleaned[k] = v
    return cleaned


def _hash_identity(raw: str | None) -> str:
    """One-way hash for user IDs / IPs — 8 hex chars, not reversible."""
    if not raw or raw in ("anonymous", "-", ""):
        return "anon"
    return hashlib.sha256(raw.encode()).hexdigest()[:8]


# ── Cloud Run environment detection ─────────────────────────────────────────
def _is_cloud_run() -> bool:
    return bool(os.getenv("K_SERVICE", "").strip())


def _cloud_revision() -> str:
    return os.getenv("K_REVISION", "local")


def _service_name() -> str:
    return os.getenv("K_SERVICE", "gaffers-guide-api")


# ── Structured JSON Formatter ────────────────────────────────────────────────
class StructuredJSONFormatter(logging.Formatter):
    """
    Emits each log record as a single-line JSON object compatible with
    Google Cloud Logging structured logs.

    Cloud Logging recognises:
      severity, message, timestamp, httpRequest, labels, trace
    Everything else goes into jsonPayload automatically.
    """

    # Map Python log levels → Cloud Logging severity strings
    _SEVERITY = {
        logging.DEBUG:    "DEBUG",
        logging.INFO:     "INFO",
        logging.WARNING:  "WARNING",
        logging.ERROR:    "ERROR",
        logging.CRITICAL: "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        severity = self._SEVERITY.get(record.levelno, "DEFAULT")
        cid = get_correlation_id()

        entry: dict[str, Any] = {
            "severity": severity,
            "message": record.getMessage(),
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "correlation_id": cid,
            "logger": record.name,
            "labels": {
                "service": _service_name(),
                "revision": _cloud_revision(),
                "component": record.name.split(".")[-1],
            },
        }

        # Attach exception info if present
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)

        # Pull any extra structured fields attached via `extra={...}`
        skip = {
            "args", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "message",
            "module", "msecs", "msg", "name", "pathname", "process",
            "processName", "relativeCreated", "stack_info", "thread",
            "threadName", "taskName",
        }
        for k, v in record.__dict__.items():
            if k not in skip and not k.startswith("_"):
                entry[k] = v

        # Safety: scrub any accidentally attached sensitive fields
        entry = _scrub(entry)

        return json.dumps(entry, default=str, ensure_ascii=False)


# ── Pretty formatter for local dev ──────────────────────────────────────────
class _DevFormatter(logging.Formatter):
    _COLOURS = {
        "DEBUG":    "\033[36m",    # cyan
        "INFO":     "\033[32m",    # green
        "WARNING":  "\033[33m",    # yellow
        "ERROR":    "\033[31m",    # red
        "CRITICAL": "\033[35m",    # magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        colour = self._COLOURS.get(record.levelname, "")
        cid = get_correlation_id()
        prefix = f"{colour}{record.levelname:<8}{self._RESET} [{cid[:8]}] {record.name}: "
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return prefix + msg


# ── Logging configuration ────────────────────────────────────────────────────
_logging_configured = False


def configure_structured_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger to emit structured JSON (Cloud Run) or
    pretty text (local dev).  Call once at application startup.

    Idempotent — safe to call multiple times.
    """
    global _logging_configured
    if _logging_configured:
        return
    _logging_configured = True

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers (uvicorn adds its own by default)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if _is_cloud_run():
        handler.setFormatter(StructuredJSONFormatter())
    else:
        handler.setFormatter(_DevFormatter())

    root.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("uvicorn.access", "httpx", "httpcore", "multipart"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ── PipelineMetricsRegistry (unchanged public API) ───────────────────────────
@dataclass(slots=True)
class _TimerSample:
    name: str
    elapsed_ms: float


class PipelineMetricsRegistry:
    """In-memory metrics registry for beta baselining and SLO gates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def incr(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] += value

    def observe_ms(self, key: str, elapsed_ms: float) -> None:
        with self._lock:
            self._timers[key].append(float(elapsed_ms))

    @contextmanager
    def timed(self, key: str) -> Generator[None, None, None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.observe_ms(key, elapsed_ms)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            timers: dict[str, dict[str, float | int]] = {}
            for key, samples in self._timers.items():
                if not samples:
                    continue
                values = sorted(samples)
                count = len(values)
                p50 = values[int((count - 1) * 0.50)]
                p95 = values[int((count - 1) * 0.95)]
                timers[key] = {
                    "count": count,
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "avg_ms": round(sum(values) / count, 2),
                }
            return {
                "counters": dict(self._counters),
                "timers": timers,
            }


# ── Rolling failure monitors ─────────────────────────────────────────────────
class _RollingFailureMonitor:
    """
    Tracks success/failure events in a rolling time window.
    Thread-safe. Used by the /health endpoint.
    """

    def __init__(self, window_seconds: int = 300, max_samples: int = 200) -> None:
        self._window = window_seconds
        self._lock = threading.Lock()
        # Each entry: (timestamp_float, is_failure: bool)
        self._events: deque[tuple[float, bool]] = deque(maxlen=max_samples)

    def record(self, *, failed: bool) -> None:
        with self._lock:
            self._events.append((time.monotonic(), failed))

    def failure_rate_pct(self) -> float:
        """Return failure % over the rolling window. 0.0 if no events."""
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            recent = [e for e in self._events if e[0] >= cutoff]
        if not recent:
            return 0.0
        failures = sum(1 for _, f in recent if f)
        return round(failures / len(recent) * 100.0, 2)

    def total_in_window(self) -> int:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            return sum(1 for ts, _ in self._events if ts >= cutoff)


# Module-level singletons — imported by main.py and llm_service.py
gemini_monitor = _RollingFailureMonitor(window_seconds=300)
upload_monitor = _RollingFailureMonitor(window_seconds=300)

# Aliases used in health check
GeminiFailureMonitor = gemini_monitor
UploadFailureMonitor = upload_monitor


# ── Gemini call tracker ───────────────────────────────────────────────────────
_GEMINI_LOGGER = logging.getLogger("gaffer.gemini")


@contextmanager
def track_gemini_call(
    model_name: str = "unknown",
    correlation_id: str | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that wraps a Gemini API call.

    On success → logs gemini.call with latency.
    On failure → logs gemini.failure with error_type (never prompt/response text).
    Updates gemini_monitor for the /health failure-rate check.

    Usage::

        with track_gemini_call(model_name="gemini-2.5-flash"):
            response = model.generate_content(prompt, ...)
    """
    cid = correlation_id or get_correlation_id()
    t0 = time.perf_counter()
    try:
        yield
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gemini_monitor.record(failed=False)
        _GEMINI_LOGGER.info(
            "Gemini call succeeded",
            extra={
                "event": "gemini.call",
                "model": model_name,
                "latency_ms": latency_ms,
                "correlation_id": cid,
            },
        )
    except RuntimeError:
        # Safety block or empty response — logged as WARNING (expected edge case)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gemini_monitor.record(failed=True)
        _GEMINI_LOGGER.warning(
            "Gemini call blocked or returned no text",
            extra={
                "event": "gemini.failure",
                "error_type": "safety_block_or_empty",
                "model": model_name,
                "latency_ms": latency_ms,
                "correlation_id": cid,
            },
        )
        raise
    except ValueError:
        # Missing API key — logged as ERROR (configuration problem)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gemini_monitor.record(failed=True)
        _GEMINI_LOGGER.error(
            "Gemini not configured — API key missing",
            extra={
                "event": "gemini.failure",
                "error_type": "not_configured",
                "model": model_name,
                "latency_ms": latency_ms,
                "correlation_id": cid,
            },
        )
        raise
    except Exception:
        # Unexpected API error
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        gemini_monitor.record(failed=True)
        _GEMINI_LOGGER.error(
            "Gemini API call failed",
            extra={
                "event": "gemini.failure",
                "error_type": "api_error",
                "model": model_name,
                "latency_ms": latency_ms,
                "correlation_id": cid,
            },
            exc_info=True,
        )
        raise


# ── Upload event tracker ──────────────────────────────────────────────────────
_UPLOAD_LOGGER = logging.getLogger("gaffer.upload")


def track_upload_event(
    phase: str,
    *,
    job_id: str = "-",
    size_bytes: int | None = None,
    chunk_index: int | None = None,
    failed: bool = False,
    error_type: str | None = None,
    correlation_id: str | None = None,
) -> None:
    """
    Fire-and-forget structured upload event log.

    Phases: "init" | "chunk" | "complete" | "gcs_sync" | "failure"

    Never logs filenames, paths, or video content — only job_id and metrics.
    """
    cid = correlation_id or get_correlation_id()
    upload_monitor.record(failed=failed)

    extra: dict[str, Any] = {
        "event": f"upload.{phase}",
        "job_id": job_id,
        "correlation_id": cid,
    }
    if size_bytes is not None:
        extra["size_bytes"] = size_bytes
    if chunk_index is not None:
        extra["chunk_index"] = chunk_index
    if error_type:
        extra["error_type"] = error_type

    if failed:
        _UPLOAD_LOGGER.error(
            "Upload %s failed",
            phase,
            extra=extra,
        )
    else:
        _UPLOAD_LOGGER.info(
            "Upload %s",
            phase,
            extra=extra,
        )


# ── Pipeline Performance Stage Timer ─────────────────────────────────────────
_PERF_LOGGER = logging.getLogger("gaffer.perf")


@contextmanager
def perf_stage(
    logger: logging.Logger,
    job_id: str,
    stage: str,
    **extra: Any,
) -> Generator[None, None, None]:
    """
    Exception-safe performance timing context-manager for pipeline stages.

    Emits a single structured ``PERF_STAGE`` log on exit containing:
        job_id, stage, duration_seconds, status ("ok" | "error")

    Cloud Logging compatible — works with ``StructuredJSONFormatter``.
    Safe alongside ``await`` — the context-manager is synchronous but
    the body may contain ``await`` expressions.

    Usage::

        with perf_stage(LOGGER, job_id, "gcs_download", blob=blob_name):
            gcs_service.download_file(blob_name, dest_path)
    """
    t0 = time.perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        raise
    finally:
        duration = round(time.perf_counter() - t0, 3)
        log_extra: dict[str, Any] = {
            "job_id": job_id,
            "stage": stage,
            "duration_seconds": duration,
            "status": status,
        }
        log_extra.update(extra)
        logger.info("PERF_STAGE", extra=log_extra)
