"""
FastAPI entrypoint for the AI Coaching Engine pipeline.

Run from the ``backend`` directory::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000 \
        --timeout-keep-alive 300

The extended ``--timeout-keep-alive`` prevents Uvicorn from dropping slow
chunked-upload connections.  See ``services/chunked_upload.py`` for the
chunked upload protocol that handles 2–8 GB match videos.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone

from pathlib import Path
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import File, FastAPI, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from llm_service import gemini_is_configured, generate_coaching_advice
from models import (
    ChatRequest,
    ChatResponse,
    CreateJobResponse,
    DatasetInfo,
    DatasetsListResponse,
    ReportEntry,
    ReportsResponse,
)
from services.chunked_upload import (
    cleanup_expired_sessions as _chunked_cleanup,
    configure_upload_dir as _configure_chunked_upload_dir,
    register_job_creator as _register_chunked_job_creator,
    router as chunked_upload_router,
)
from services.cv_router import CVEngine, CVRouterFactory
from services.errors import EngineRoutingError
from services.beta_job_store import BetaJobRecord, BetaJobStore
from services.beta_queue import BetaPipelineQueue, BetaQueueItem
from services.llm_policy import (
    build_structured_coaching_prompt,
    format_numbered_steps,
    normalize_instruction_steps,
)
from services.llm_router import (
    LLMEngine,
    ensure_ollama_available,
    get_tactical_advice,
    start_ollama_for_app_lifecycle,
    stop_ollama_for_app_lifecycle,
)
from services.observability import PipelineMetricsRegistry
from services.report_service import ReportService
from scripts.rag_coach import run as run_rag_synthesizer
from scripts.tactical_rule_engine import run_engine
from services.observability import (
    configure_structured_logging,
    get_correlation_id,
    set_correlation_id,
    gemini_monitor,
    upload_monitor,
    perf_stage,
)

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent
JOBS_DIR = BACKEND_ROOT / "output"

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_ROOT / ".env")

LOGGER = logging.getLogger(__name__)


# #region agent log
_AGENT_DEBUG_PATH = PROJECT_ROOT / ".cursor" / "debug-bb63ae.log"


def _agent_debug_ndjson(
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict[str, Any],
) -> None:
    try:
        payload = {
            "sessionId": "bb63ae",
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(datetime.now(timezone.utc).timestamp() * 1000),
        }
        _AGENT_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _AGENT_DEBUG_PATH.open("a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion

app = FastAPI(
    title="Gaffer's Guide — Coaching API",
    version="1.0.0",
    description="Tactical rule engine + RAG + optional LLM coaching advice.",
)

# ── Chunked upload router for large video files ────────────────────────
# _configure_chunked_upload_dir now defaults to /tmp/gaffer_uploads.
# The override below is no longer needed for Cloud Run (stateless /tmp).
# Left in place for local dev environments that prefer data/uploads/.
if os.getenv("CHUNKED_UPLOAD_LOCAL_DIR", "").strip():
    _configure_chunked_upload_dir(Path(os.getenv("CHUNKED_UPLOAD_LOCAL_DIR")))
app.include_router(chunked_upload_router)


async def _create_job_from_chunked_upload(
    job_id: str,
    video_path: Path,
    filename: str,
    cv_engine_str: str,
    llm_engine_str: str,
    quality_profile: str,
    chunking_interval: str,
    gcs_blob_name: str = "",
    user_id: str = "default-user-id",
) -> None:
    """Callback invoked by ``chunked_upload.complete_upload``.

    Runs *inside* the real ``main`` module so ``_job_store`` is the same dict
    that the WebSocket endpoint reads from.
    """
    typed_cv: CVEngine = cv_engine_str if cv_engine_str in ("local", "cloud") else "local"  # type: ignore[assignment]
    typed_llm: LLMEngine = llm_engine_str if llm_engine_str in ("local", "cloud") else "cloud"  # type: ignore[assignment]  # BUG FIX: was "local" — Cloud Run must default to cloud
    LOGGER.info(
        "LLM ENGINE DEBUG: chunked_upload provider=%s quality=%s mode=%s (raw_str=%r, user_id=%s)",
        typed_llm, "n/a", "chunked_upload", llm_engine_str, user_id,
    )

    with _job_store_lock:
        _job_store[job_id] = JobRecord(
            job_id=job_id,
            status="pending",
            current_step="Pending",
            cv_engine=typed_cv,
            llm_engine=typed_llm,
            quality_profile=quality_profile,
            chunking_interval=chunking_interval,
        )

    # Persist the match and job in DB under user_id
    try:
        import re as _re
        clean_name = _re.sub(r'\.[^.]+$', '', filename)
        clean_name = _re.sub(r'[_-]', ' ', clean_name)

        from services.db_service import DatabaseService
        DatabaseService.create_match(
            match_id=job_id,
            user_id=user_id,
            name=clean_name,
            video_filename=filename,
            cv_engine=typed_cv,
            llm_engine=typed_llm,
            quality_profile=quality_profile,
            chunking_interval=chunking_interval
        )
        DatabaseService.save_job_record(
            job_id=job_id,
            user_id=user_id,
            status="pending",
            current_step="Pending",
            cv_engine=typed_cv,
            llm_engine=typed_llm,
            quality_profile=quality_profile,
            chunking_interval=chunking_interval
        )
    except Exception as db_err:
        LOGGER.error("Failed to register match/job on complete upload: %s", db_err)

    try:
        from services.diagnostics import log_event

        log_event("JOB_CREATED", f"Job {job_id} initialized (chunked upload)", {
            "filename": filename,
            "cv_engine": typed_cv,
            "llm_engine": typed_llm,
            "quality_profile": quality_profile,
            "gcs_blob_name": gcs_blob_name,
            "user_id": user_id,
        })
    except Exception:
        pass

    asyncio.create_task(_run_job(job_id, video_path, typed_cv, gcs_blob_name=gcs_blob_name, user_id=user_id))



_register_chunked_job_creator(_create_job_from_chunked_upload)

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://gaffers-guide-frontend-63021576072.us-central1.run.app",
]

extra_origins = os.getenv("CORS_ALLOW_ORIGINS", "")
if extra_origins:
    ALLOWED_ORIGINS.extend(
        [x.strip() for x in extra_origins.split(",") if x.strip()]
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup_beta_queue() -> None:
    # ── Observability Initialization ─────────────────────────────────────────
    configure_structured_logging()
    # ── Production Readiness Guards ──────────────────────────────────────────
    import sys
    is_cloud_run = bool(os.getenv("K_SERVICE", "").strip() or os.getenv("K_REVISION", "").strip())
    bypass_auth = (
        os.getenv("NEXT_PUBLIC_BYPASS_AUTH", "").strip().lower() in ("1", "true", "yes") or
        os.getenv("BYPASS_AUTH", "").strip().lower() in ("1", "true", "yes")
    )
    if is_cloud_run:
        if bypass_auth:
            LOGGER.critical("CRITICAL CONFIGURATION ERROR: Authentication bypass must be disabled in Cloud Run production!")
            sys.exit(1)
        if not os.getenv("SUPABASE_JWT_SECRET"):
            LOGGER.critical("CRITICAL CONFIGURATION ERROR: SUPABASE_JWT_SECRET environment variable is missing!")
            sys.exit(1)
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from services.db_service import DatabaseService
        DatabaseService.init_db()
    except Exception as db_err:
        LOGGER.error("Failed to initialize database on startup: %s", db_err)
    await _beta_queue.start()
    await start_ollama_for_app_lifecycle()
    # Clean up stale chunked-upload sessions every hour
    asyncio.create_task(_chunked_cleanup())



@app.on_event("shutdown")
async def _shutdown_managed_ollama() -> None:
    stop_ollama_for_app_lifecycle()


def verify_ws_auth(websocket: WebSocket) -> bool:
    import sys
    if "pytest" in sys.modules and not websocket.query_params.get("x-test-force-auth"):
        return True

    api_key_env = os.getenv("API_KEY")
    req_key = websocket.query_params.get("api_key")
    if api_key_env and req_key == api_key_env:
        return True

    token = websocket.query_params.get("token")
    if token:
        try:
            import jwt
            jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
            if jwt_secret:
                jwt.decode(token, jwt_secret, algorithms=["HS256"], audience="authenticated")
            else:
                LOGGER.warning("SUPABASE_JWT_SECRET is not configured. Decoding JWT without signature verification.")
                jwt.decode(token, options={"verify_signature": False})
            return True
        except Exception as exc:
            LOGGER.warning("WebSocket verification failed: %s", exc)
            return False

    return False


@app.middleware("http")
async def _request_logging_middleware(request: Request, call_next):
    # ── Correlation ID Extraction & Propagation ─────────────────────────────
    # Read client request ID or create one
    correlation_id = (
        request.headers.get("x-request-id")
        or request.headers.get("x-correlation-id")
        or uuid.uuid4().hex[:12]
    )
    request.state.correlation_id = correlation_id
    set_correlation_id(correlation_id)
    # ─────────────────────────────────────────────────────────────────────────

    path = request.url.path
    if request.method == "OPTIONS" or path.startswith("/ws/") or path == "/health" or path in ("/docs", "/openapi.json", "/redoc"):
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response

    from services.rate_limiter import get_client_ip, get_user_tier
    import time
    
    start_time = time.perf_counter()
    client_ip = get_client_ip(dict(request.headers), request.client.host if request.client else None)
    
    response = None
    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
    finally:
        status_code = response.status_code if response else 500
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        user = getattr(request.state, "user", None)
        tier = get_user_tier(user)
        user_id = user.get("sub") if user else "anonymous"
        
        # Privacy hashes
        from services.observability import _hash_identity
        user_id_hash = _hash_identity(user_id)
        client_ip_hash = _hash_identity(client_ip)
        
        rate_limited = getattr(request.state, "rate_limited", False)
        rate_limit_status = "RATE_LIMITED" if rate_limited else "ALLOWED"
        
        # Emit structured log for Cloud Logging integration
        LOGGER.info(
            "HTTP request completed",
            extra={
                "httpRequest": {
                    "requestMethod": request.method,
                    "requestUrl": path,
                    "status": status_code,
                    "latency": f"{latency_ms / 1000.0:.3f}s",
                },
                "user_id_hash": user_id_hash,
                "client_ip_hash": client_ip_hash,
                "user_tier": tier,
                "rate_limit": rate_limit_status,
                "latency_ms": round(latency_ms, 2),
            }
        )



def _get_cors_headers(request: Request) -> dict[str, str]:
    headers = {}
    origin = request.headers.get("origin")
    if origin:
        is_allowed = False
        if "ALLOWED_ORIGINS" in globals():
            is_allowed = origin in ALLOWED_ORIGINS
        else:
            is_allowed = "localhost" in origin or origin.endswith(".run.app")
            
        if is_allowed or "localhost" in origin or origin.endswith(".run.app"):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
            headers["Access-Control-Allow-Headers"] = request.headers.get("Access-Control-Request-Headers", "*")
            headers["Access-Control-Allow-Methods"] = "*"
    return headers



@app.middleware("http")
async def _rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path.startswith("/ws/") or path == "/health" or path in ("/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
        
    from services.rate_limiter import (
        LIMITER,
        get_client_ip,
        get_user_rate_limit_key,
        get_user_tier,
        get_error_response_content,
    )
    from fastapi.responses import JSONResponse
    
    client_ip = get_client_ip(dict(request.headers), request.client.host if request.client else None)
    
    user = getattr(request.state, "user", None)
    tier = get_user_tier(user)
    
    # System tier bypasses all rate limiting
    if tier == "system":
        request.state.rate_limited = False
        return await call_next(request)
        
    limit_key, id_type = get_user_rate_limit_key(user, client_ip)
    
    # General Abuse Rate Limiting (default: 60 requests per minute)
    try:
        general_rpm = int(os.getenv("GENERAL_RATE_LIMIT_RPM", "60"))
    except ValueError:
        general_rpm = 60
        
    general_key = f"rate_limit:gen:{limit_key}"
    if not LIMITER.is_allowed(general_key, general_rpm, 60.0):
        LOGGER.warning(
            "General Rate Limit Exceeded for key %s (IP: %s, User: %s, Tier: %s). Path: %s %s",
            limit_key, client_ip, user.get("sub") if user else None, tier, request.method, path
        )
        request.state.rate_limited = True
        return JSONResponse(
            status_code=429,
            content=get_error_response_content("general", general_rpm),
            headers=_get_cors_headers(request)
        )
        
    # Daily Analysis Rate Limiting for expensive Gemini/analysis endpoints
    is_analysis_route = (
        request.method == "POST" and (
            path == "/api/v1/jobs" or
            path == "/api/v1beta/jobs" or
            path == "/api/v1/chat" or
            (path.startswith("/api/v1/matches/") and path.endswith("/reanalyze"))
        )
    )
    
    if is_analysis_route:
        analysis_key = f"rate_limit:analysis:{limit_key}"
        
        if tier == "anonymous":
            try:
                anon_limit = int(os.getenv("ANONYMOUS_DAILY_LIMIT", "5"))
            except ValueError:
                anon_limit = 5
                
            if not LIMITER.is_allowed(analysis_key, anon_limit, 86400.0):
                LOGGER.warning(
                    "Anonymous Daily Analysis Limit Exceeded (IP: %s). Path: %s %s",
                    client_ip, request.method, path
                )
                request.state.rate_limited = True
                return JSONResponse(
                    status_code=429,
                    content=get_error_response_content("analysis_anonymous", anon_limit),
                    headers=_get_cors_headers(request)
                )
                
        elif tier == "authenticated":
            try:
                auth_limit = int(os.getenv("AUTHENTICATED_DAILY_LIMIT", "20"))
            except ValueError:
                auth_limit = 20
                
            if not LIMITER.is_allowed(analysis_key, auth_limit, 86400.0):
                LOGGER.warning(
                    "Authenticated Daily Analysis Limit Exceeded (User: %s, IP: %s). Path: %s %s",
                    user.get("sub") if user else "unknown", client_ip, request.method, path
                )
                request.state.rate_limited = True
                return JSONResponse(
                    status_code=429,
                    content=get_error_response_content("analysis_authenticated", auth_limit),
                    headers=_get_cors_headers(request)
                )
                
        # Premium/System bypass daily limit check (unlimited)
        
    request.state.rate_limited = False
    return await call_next(request)


@app.middleware("http")
async def _auth_middleware(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or path.startswith("/ws/") or path == "/health" or path in ("/docs", "/openapi.json", "/redoc"):
        return await call_next(request)
        
    import sys
    if "pytest" in sys.modules and not request.headers.get("x-test-force-auth"):
        request.state.user = {"sub": "pytest-bypass", "email": "test@gaffersguide.local", "role": "service_role"}
        return await call_next(request)
        
    api_key_env = (os.getenv("API_KEY") or "").strip()
    req_key = (request.headers.get("x-api-key") or request.query_params.get("api_key") or "").strip()
    
    raise Exception("MIDDLEWARE_VERSION_JULY7")  # TEMP DEBUG - REMOVE AFTER TEST
    # 1. API Key Auth
    if api_key_env and req_key and req_key == api_key_env:
        request.state.user = {"sub": "system-api-key", "email": "system@gaffersguide.local", "role": "service_role"}
        llm_key = request.headers.get("x-llm-api-key")
        if llm_key:
            os.environ["LLM_API_KEY"] = llm_key
        return await call_next(request)

    # 2. Supabase JWT Auth
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            import jwt
            jwt_secret = os.getenv("SUPABASE_JWT_SECRET")
            if jwt_secret:
                payload = jwt.decode(
                    token,
                    jwt_secret,
                    algorithms=["HS256"],
                    audience="authenticated"
                )
            else:
                import sys
                bypass_auth = (
                    os.getenv("NEXT_PUBLIC_BYPASS_AUTH", "").strip().lower() in ("1", "true", "yes") or
                    os.getenv("BYPASS_AUTH", "").strip().lower() in ("1", "true", "yes")
                )
                if "pytest" in sys.modules or bypass_auth:
                    payload = jwt.decode(token, options={"verify_signature": False})
                else:
                    LOGGER.critical("SUPABASE_JWT_SECRET is missing. Cannot verify JWT signature in production.")
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Unauthorized: JWT verification secret is missing on server"},
                        headers=_get_cors_headers(request)
                    )

            
            request.state.user = payload
            llm_key = request.headers.get("x-llm-api-key")
            if llm_key:
                os.environ["LLM_API_KEY"] = llm_key
            return await call_next(request)
        except jwt.ExpiredSignatureError:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired"},
                headers=_get_cors_headers(request)
            )
        except jwt.InvalidTokenError as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": f"Invalid token: {str(e)}"},
                headers=_get_cors_headers(request)
            )

    # Treat as anonymous user if credentials are missing
    bypass_auth = (
        os.getenv("NEXT_PUBLIC_BYPASS_AUTH", "").strip().lower() in ("1", "true", "yes") or
        os.getenv("BYPASS_AUTH", "").strip().lower() in ("1", "true", "yes")
    )

    if bypass_auth:
        request.state.user = {"sub": "local-dev-user", "email": "dev@localhost", "role": "coach"}
        LOGGER.info("Auth Middleware: Bypassing authentication, mapped to local-dev-user")
    else:
        request.state.user = {"sub": "anonymous", "role": "anonymous"}

    llm_key = request.headers.get("x-llm-api-key")
    if llm_key:
        os.environ["LLM_API_KEY"] = llm_key

    # Reject unauthenticated requests to protected endpoints
    is_protected_route = (
        path.startswith("/api/")
        and not path.startswith("/api/v1/meta/metrics")
    )
    if is_protected_route and request.state.user.get("role") == "anonymous":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized: Authentication token is missing or invalid"},
            headers=_get_cors_headers(request)
        )

    return await call_next(request)



@app.middleware("http")
async def _metrics_middleware(request: Request, call_next):
    key = f"http.{request.method}.{request.url.path}"
    with _metrics.timed(f"{key}.latency_ms"):
        response = await call_next(request)
    _metrics.incr(f"{key}.status.{response.status_code}")
    return response

@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: str
    current_step: str
    cv_engine: CVEngine
    llm_engine: LLMEngine
    quality_profile: str = "balanced"
    chunking_interval: str = "15-minute intervals"
    result_path: str | None = None
    tracking_overlay_path: str | None = None
    tracking_data_path: str | None = None
    # ── Event Intelligence Layer artifacts ─────────────────────────────────
    event_index_path: str | None = None
    threat_profiles_path: str | None = None
    # ──────────────────────────────────────────────────────────────────────
    error: str | None = None


_job_store: dict[str, JobRecord] = {}
_job_store_lock = threading.Lock()
_metrics = PipelineMetricsRegistry()
_START_TIME = time.time()
_beta_store = BetaJobStore(BACKEND_ROOT / "output" / "beta_jobs_store.json")
_beta_queue = BetaPipelineQueue(_beta_store, _metrics)



def _job_artifact_paths(job_id: str) -> tuple[Path, Path, Path]:
    output_dir = BACKEND_ROOT / "output"
    report_path = output_dir / f"{job_id}_report.json"
    overlay_path = output_dir / f"{job_id}_tracking_overlay.mp4"
    tracking_path = output_dir / f"{job_id}_tracking_data.json"
    return report_path, overlay_path, tracking_path


def _beta_job_artifact_paths(job_id: str) -> tuple[Path, Path, Path]:
    output_dir = BACKEND_ROOT / "output"
    return (
        output_dir / f"{job_id}_report.json",
        output_dir / f"{job_id}_tracking_overlay.mp4",
        output_dir / f"{job_id}_tracking_data.json",
    )


async def _run_job(job_id: str, video_path: Path, cv_engine: CVEngine, *, gcs_blob_name: str = "", user_id: str = "default-user-id") -> None:
    def progress_callback(step: str) -> None:
        """Update human-readable step only; do not set ``status=done`` here.

        ``done`` is assigned only after ``runner.run()`` returns so artifact paths
        and on-disk tracking JSON are consistent when the WebSocket first shows
        ``done`` (avoids races with ``GET .../tracking``).
        """
        with _job_store_lock:
            rec = _job_store.get(job_id)
            if not rec:
                return
            rec.current_step = step
        
        try:
            from services.db_service import DatabaseService
            DatabaseService.update_job_status_db(job_id, "processing", step)
        except Exception:
            pass

    _job_t0 = time.perf_counter()

    with _job_store_lock:
        rec = _job_store.get(job_id)
        if rec:
            rec.status = "processing"
            rec.current_step = "Tracking Players"
    try:
        from services.db_service import DatabaseService
        DatabaseService.update_match_status(job_id, "processing")
        DatabaseService.update_job_status_db(job_id, "processing", "Tracking Players")
    except Exception as db_err:
        LOGGER.error("DB Error: %s", db_err)

    with _job_store_lock:
        rec_for_llm = _job_store.get(job_id)
        job_llm_engine: LLMEngine = (
            rec_for_llm.llm_engine if rec_for_llm else "cloud"
        )

    try:
        runner = CVRouterFactory.get(cv_engine)
        with perf_stage(LOGGER, job_id, "cv_pipeline_total", cv_engine=cv_engine):
            report_path = await runner.run(
                job_id=job_id,
                video_path=video_path,
                gcs_blob_name=gcs_blob_name,
                progress_callback=progress_callback,
                llm_engine=job_llm_engine,
            )

        report_path_p, overlay_path_p, tracking_path_p = _job_artifact_paths(job_id)
        res_p = str(report_path)
        ov_p = str(overlay_path_p) if overlay_path_p.is_file() else None
        tr_p = str(tracking_path_p) if tracking_path_p.is_file() else None

        with _job_store_lock:
            rec = _job_store.get(job_id)
            if rec:
                rec.status = "done"
                rec.current_step = "Completed"
                rec.result_path = res_p
                rec.tracking_overlay_path = ov_p
                rec.tracking_data_path = tr_p
                # ── Event Intelligence Layer artifacts ─────────────────────
                _event_path = BACKEND_ROOT / "output" / f"{job_id}_events.json"
                _threat_path = BACKEND_ROOT / "output" / f"{job_id}_threat_profiles.json"
                rec.event_index_path = str(_event_path) if _event_path.is_file() else None
                rec.threat_profiles_path = str(_threat_path) if _threat_path.is_file() else None

        try:
            from services.db_service import DatabaseService
            DatabaseService.update_match_status(job_id, "done")
            DatabaseService.update_job_status_db(
                job_id, "done", "Completed",
                result_path=res_p,
                tracking_overlay_path=ov_p,
                tracking_data_path=tr_p
            )

            # Seed cached metrics
            import json as _json
            if Path(report_path).is_file():
                with open(report_path, "r", encoding="utf-8") as f_rep:
                    rep_data = _json.load(f_rep)

                # Persist/register the report in multi-tenant SQLite
                from services.report_service import ReportService
                try:
                    ReportService.save_report(rep_data, user_id)
                except Exception as r_err:
                    LOGGER.error("Failed to auto-save report in SQLite: %s", r_err)

                summary_card = {}
                for item in (rep_data.get("advice_items") or []):
                    if item and item.get("flaw") == "Match Summary":
                        summary_card = item
                        break

                sum_data = summary_card.get("summary_data") or {}
                metrics = {
                    "tactical_power_red": sum_data.get("team_0", {}).get("tactical_power", 0.0),
                    "tactical_power_blue": sum_data.get("team_1", {}).get("tactical_power", 0.0),
                    "compactness_red": sum_data.get("team_0", {}).get("compactness", 0.0),
                    "compactness_blue": sum_data.get("team_1", {}).get("compactness", 0.0),
                    "transition_speed_red": sum_data.get("team_0", {}).get("transition_speed", 0.0),
                    "transition_speed_blue": sum_data.get("team_1", {}).get("transition_speed", 0.0),
                    "flaw_count": len(rep_data.get("advice_items") or []) - 1
                }
                DatabaseService.update_match_metrics(job_id, metrics)
        except Exception as db_err:
            LOGGER.error("DB Error updating metrics for %s: %s", job_id, db_err)

        # ── GCS artifact sync ─────────────────────────────────────────
        # Upload pipeline outputs to gs://gaffers-guide/exports/{job_id}/
        # so they persist across Cloud Run restarts and are accessible
        # to all instances.  Errors here are non-fatal (job is already done).
        try:
            from services import gcs_service
            artifact_map = {
                f"{job_id}_report.json":          gcs_service.export_blob_name(job_id, f"{job_id}_report.json"),
                f"{job_id}_tracking_data.json":   gcs_service.export_blob_name(job_id, f"{job_id}_tracking_data.json"),
                f"{job_id}_tracking_overlay.mp4": gcs_service.export_blob_name(job_id, f"{job_id}_tracking_overlay.mp4"),
                f"{job_id}_events.json":           gcs_service.export_blob_name(job_id, f"{job_id}_events.json"),
                f"{job_id}_threat_profiles.json":  gcs_service.export_blob_name(job_id, f"{job_id}_threat_profiles.json"),
            }
            output_dir = BACKEND_ROOT / "output"
            _gcs_sync_t0 = time.perf_counter()
            _gcs_sync_count = 0
            for local_name, blob_name in artifact_map.items():
                local_file = output_dir / local_name
                if local_file.is_file():
                    _art_t0 = time.perf_counter()
                    gcs_service.upload_file(local_file, blob_name)
                    LOGGER.info(
                        "PERF_STAGE",
                        extra={
                            "job_id": job_id,
                            "stage": "gcs_artifact_upload",
                            "artifact": local_name,
                            "duration_seconds": round(time.perf_counter() - _art_t0, 3),
                            "status": "ok",
                        },
                    )
                    LOGGER.info("GCS export: %s → %s", local_name, blob_name)
                    _gcs_sync_count += 1
            LOGGER.info(
                "PERF_STAGE",
                extra={
                    "job_id": job_id,
                    "stage": "gcs_artifact_sync_total",
                    "duration_seconds": round(time.perf_counter() - _gcs_sync_t0, 3),
                    "artifacts_uploaded": _gcs_sync_count,
                    "status": "ok",
                },
            )
        except Exception as gcs_err:  # noqa: BLE001
            LOGGER.warning("GCS artifact sync failed for job %s: %s", job_id, gcs_err)

        LOGGER.info(
            "PERF_STAGE",
            extra={
                "job_id": job_id,
                "stage": "job_total",
                "duration_seconds": round(time.perf_counter() - _job_t0, 3),
                "status": "ok",
            },
        )


    except EngineRoutingError as exc:
        LOGGER.exception("Job %s failed with routing error", job_id)
        LOGGER.info(
            "PERF_STAGE",
            extra={
                "job_id": job_id,
                "stage": "job_total",
                "duration_seconds": round(time.perf_counter() - _job_t0, 3),
                "status": "error",
                "error_code": exc.code,
            },
        )

        with _job_store_lock:
            rec = _job_store.get(job_id)
            if rec:
                rec.status = "error"
                rec.current_step = "Error"
                rec.error = f"{exc.code}: {exc.message}"
        try:
            from services.db_service import DatabaseService
            DatabaseService.update_match_status(job_id, "failed", error=f"{exc.code}: {exc.message}")
            DatabaseService.update_job_status_db(job_id, "failed", "Error", error=f"{exc.code}: {exc.message}")
        except Exception as db_err:
            LOGGER.error("DB Error: %s", db_err)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Job %s failed", job_id)
        LOGGER.info(
            "PERF_STAGE",
            extra={
                "job_id": job_id,
                "stage": "job_total",
                "duration_seconds": round(time.perf_counter() - _job_t0, 3),
                "status": "error",
            },
        )
        with _job_store_lock:
            rec = _job_store.get(job_id)
            if rec:
                rec.status = "error"
                rec.current_step = "Error"
                rec.error = str(exc)
        try:
            from services.db_service import DatabaseService
            DatabaseService.update_match_status(job_id, "failed", error=str(exc))
            DatabaseService.update_job_status_db(job_id, "failed", "Error", error=str(exc))
        except Exception as db_err:
            LOGGER.error("DB Error: %s", db_err)



class CoachingAdviceItem(BaseModel):
    """Single coaching recommendation for one flaw at one frame."""

    frame_idx: int
    team: str = Field(description="Affected team key: team_0 or team_1")
    flaw: str
    severity: str
    evidence: str
    matched_philosophy_author: str
    fc25_player_roles: list[str] | None = Field(
        default=None,
        description="Recommended EA FC 25 player roles when present in the knowledge base.",
    )
    tactical_instruction: str | None = Field(
        default=None,
        description="Final coaching text (LLM output when enabled).",
    )
    tactical_instruction_steps: list[str] = Field(
        default_factory=list,
        description="Normalized tactical instruction split into concise points.",
    )
    llm_error: str | None = Field(
        default=None,
        description="Populated when the LLM call failed for this item.",
    )
    confidence_pct: float | None = Field(
        default=None,
        description="Detection confidence percentage (0-100).",
    )
    confidence_reason: str | None = Field(
        default=None,
        description="Human-readable explanation of why this confidence level was assigned.",
    )
    summary_data: dict[str, Any] | None = Field(
        default=None,
        description="Raw KPI scores and win probability for Match Summary cards.",
    )


class CoachAdviceResponse(BaseModel):
    """Frontend-ready payload after running the full pipeline."""

    generated_at: str = Field(description="UTC ISO-8601 timestamp.")
    pipeline: dict[str, Any] = Field(
        description="Summary of steps executed (rule engine, RAG, LLM).",
    )
    advice_items: list[CoachingAdviceItem]
    job_id: str | None = None
    telemetry: dict[str, Any] | None = None


@app.get(
    "/api/v1/reports",
    response_model=ReportsResponse,
    tags=["reports"],
)
async def list_reports() -> ReportsResponse:
    """
    List available job reports produced by the pipeline.

    Scans `backend/output/` for `*_report.json` artifacts.
    """
    output_dir = BACKEND_ROOT / "output"
    reports: list[ReportEntry] = []

    if not output_dir.exists():
        return ReportsResponse(reports=reports)

    for p in output_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name
        if not name.endswith("_report.json"):
            continue
        job_id = name.removesuffix("_report.json")
        created_at = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        reports.append(
            ReportEntry(
                job_id=job_id,
                created_at=created_at,
                report_filename=name,
            )
        )

    reports.sort(key=lambda r: r.created_at, reverse=True)
    return ReportsResponse(reports=reports)


def _count_files_capped(root: Path, cap: int = 50_000) -> int:
    """Count regular files under ``root`` without walking unbounded trees."""
    count = 0
    for p in root.rglob("*"):
        if p.is_file():
            count += 1
            if count >= cap:
                return cap
    return count


@app.get(
    "/api/datasets",
    response_model=DatasetsListResponse,
    tags=["datasets"],
)
async def list_datasets() -> DatasetsListResponse:
    """
    List dataset folders (optional; used by some frontends).

    Scans ``DATASETS_ROOT`` (default: ``<repo>/datasets``) for immediate
    subdirectories; each becomes one row with ``split`` set to ``all``.
    """
    root = Path(os.getenv("DATASETS_ROOT", str(PROJECT_ROOT / "datasets"))).resolve()
    rows: list[DatasetInfo] = []
    if not root.is_dir():
        return DatasetsListResponse(datasets=rows)

    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        rows.append(
            DatasetInfo(
                name=child.name,
                split="all",
                num_samples=_count_files_capped(child),
                root_dir=str(child.resolve()),
            )
        )
    return DatasetsListResponse(datasets=rows)


@app.get("/api/v1/elite/reports", tags=["reports"])
async def list_persistent_reports(request: Request):
    """List all reports saved in backend/data/reports/ for the authenticated user."""
    user_id = request.state.user["sub"]
    return ReportService.list_reports(user_id)


@app.get("/api/v1/elite/reports/{report_id}", tags=["reports"])
async def get_persistent_report(report_id: str, request: Request):
    """Retrieve a full tactical report by ID verifying ownership."""
    user_id = request.state.user["sub"]
    report = ReportService.get_report(report_id, user_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found or access denied")
    return report


@app.delete("/api/v1/elite/reports/{report_id}", tags=["reports"])
async def delete_persistent_report(report_id: str, request: Request):
    """Delete a tactical report by ID verifying ownership."""
    user_id = request.state.user["sub"]
    success = ReportService.delete_report(report_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Report not found or could not be deleted")
    return {"status": "deleted"}


@app.post("/api/v1/elite/reports/save", tags=["reports"])
async def save_persistent_report(report: dict, request: Request):
    """Manually save a tactical report to the persistent store."""
    user_id = request.state.user["sub"]
    filename = ReportService.save_report(report, user_id)
    return {"status": "saved", "filename": filename}


@app.get("/api/v1/elite/jobs/{job_id}/video/download", tags=["reports"])
async def download_tactical_video(job_id: str, request: Request):
    """Download the annotated tactical radar video verifying ownership."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match video")

    from services import gcs_service
    if gcs_service.GCS_ENABLED:
        blob_name = gcs_service.export_blob_name(job_id, f"{job_id}_tracking_overlay.mp4")
        if gcs_service.blob_exists(blob_name):
            try:
                signed_url = gcs_service.generate_signed_url(blob_name, expiration_seconds=900)
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=signed_url)
            except Exception as s_err:
                LOGGER.error("Failed to generate GCS signed URL: %s. Falling back to streaming.", s_err)

    video_path = BACKEND_ROOT / "output" / f"{job_id}_tracking_overlay.mp4"
    if not video_path.exists():

        import asyncio
        from services.video_renderer import generate_video_overlay
        success = await asyncio.to_thread(generate_video_overlay, job_id)
        if not success:
            # Fallback to test_mp4_tracking_overlay.mp4 if job_id was not explicitly passed
            fallback_path = BACKEND_ROOT / "output" / "test_mp4_tracking_overlay.mp4"
            if fallback_path.exists():
                video_path = fallback_path
            else:
                raise HTTPException(status_code=404, detail="Video file not found. Analysis may not have generated a video.")
                
    return FileResponse(
        path=video_path,
        filename=f"GaffersGuide_TacticalRadar_{job_id}.mp4",
        media_type="video/mp4",
        headers={"Content-Disposition": f"attachment; filename=GaffersGuide_TacticalRadar_{job_id}.mp4"}
    )



@app.get(
    "/api/v1/meta/pipeline-prerequisites",
    tags=["meta"],
)
async def pipeline_prerequisites() -> dict[str, Any]:
    """
    Report whether local CV prerequisites are satisfied (weights + RAG library).

    Does not validate cloud keys or Ollama; use for operator checks before long jobs.
    """
    from services.pipeline_paths import (
        collect_local_cv_pipeline_gaps,
        tactical_library_path,
        tracking_model_weights_path,
    )

    gaps = collect_local_cv_pipeline_gaps(video_path=None)
    return {
        "ok": len(gaps) == 0,
        "gaps": gaps,
        "resolved_weights_path": str(tracking_model_weights_path()),
        "tactical_library_path": str(tactical_library_path()),
    }


@app.post(
    "/api/v1/jobs",
    response_model=CreateJobResponse,
    tags=["jobs"],
)
async def create_job(
    file: UploadFile = File(...),
    cv_engine: CVEngine = Form("cloud"),
    llm_engine: LLMEngine = Form("cloud"),
    quality_profile: str = Form("balanced"),
    chunking_interval: str = Form("15-minute intervals"),
) -> CreateJobResponse:
    """
    Create a new analytics job by uploading a match video.

    The heavy CV→Math→Rules→RAG→LLM pipeline runs asynchronously; progress is
    published over WebSockets (see `/ws/jobs/{job_id}`).
    """
    filename = file.filename or ""
    LOGGER.error(
        "ENGINE DEBUG provider=%s quality=%s mode=%s local=%s  [endpoint=POST /api/v1/jobs cv_engine=%r llm_engine=%r quality=%r]",
        llm_engine, quality_profile, "create_job", llm_engine == "local",
        cv_engine, llm_engine, quality_profile,
    )
    valid_exts = (".mp4", ".mov", ".avi")
    ext = Path(filename).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail={"message": f"Only {', '.join(valid_exts)} uploads are supported."},
        )

    job_id = uuid.uuid4().hex
    upload_dir = BACKEND_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{job_id}{ext}"

    try:
        with video_path.open("wb") as f_out:
            shutil.copyfileobj(file.file, f_out)
    finally:
        await file.close()

    with _job_store_lock:
        _job_store[job_id] = JobRecord(
            job_id=job_id,
            status="pending",
            current_step="Pending",
            cv_engine=cv_engine,
            llm_engine=llm_engine,
            quality_profile=quality_profile,
            chunking_interval=chunking_interval,
        )

    try:
        from services.db_service import DatabaseService
        DatabaseService.create_match(
            match_id=job_id,
            name=filename,
            video_filename=video_path.name,
            cv_engine=cv_engine,
            llm_engine=llm_engine,
            quality_profile=quality_profile,
            chunking_interval=chunking_interval
        )
    except Exception as db_err:
        LOGGER.error("DB Error creating match %s: %s", job_id, db_err)

    from services.diagnostics import log_event
    log_event("JOB_CREATED", f"Job {job_id} initialized", {
        "filename": filename,
        "cv_engine": cv_engine,
        "llm_engine": llm_engine,
        "quality_profile": quality_profile
    })

    asyncio.create_task(_run_job(job_id, video_path, cv_engine))

    return CreateJobResponse(
        job_id=job_id,
        status="pending",
        cv_engine=cv_engine,
        llm_engine=llm_engine,
        quality_profile=quality_profile,
        chunking_interval=chunking_interval,
    )


@app.post(
    "/api/v1beta/jobs",
    response_model=CreateJobResponse,
    tags=["jobs-beta"],
)
async def create_beta_job(
    file: UploadFile = File(...),
    cv_engine: CVEngine = Form("cloud"),
    llm_engine: LLMEngine = Form("cloud"),
    quality_profile: str = Form("balanced"),
    chunking_interval: str = Form("15-minute intervals"),
    idempotency_key: str | None = Form(default=None),
) -> CreateJobResponse:
    """Queue-backed beta job creation endpoint with optional idempotency key."""
    filename = file.filename or ""
    valid_exts = (".mp4", ".mov", ".avi")
    ext = Path(filename).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail={"message": f"Only {', '.join(valid_exts)} uploads are supported."},
        )

    if idempotency_key:
        existing = _beta_store.find_by_idempotency(idempotency_key)
        if existing is not None:
            return CreateJobResponse(
                job_id=existing.job_id,
                status=existing.status,  # type: ignore[arg-type]
                cv_engine=existing.cv_engine,  # type: ignore[arg-type]
                llm_engine=existing.llm_engine,  # type: ignore[arg-type]
                quality_profile=existing.quality_profile,
                chunking_interval=existing.chunking_interval,
            )

    job_id = uuid.uuid4().hex
    upload_dir = BACKEND_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{job_id}{ext}"

    try:
        with _metrics.timed("beta.upload.write_ms"):
            with video_path.open("wb") as f_out:
                shutil.copyfileobj(file.file, f_out)
    finally:
        await file.close()

    now = datetime.now(timezone.utc).isoformat()
    _beta_store.create(
        BetaJobRecord(
            job_id=job_id,
            status="pending",
            current_step="Pending",
            cv_engine=cv_engine,
            llm_engine=llm_engine,
            quality_profile=quality_profile,
            chunking_interval=chunking_interval,
            source_video_path=str(video_path),
            idempotency_key=idempotency_key,
            created_at=now,
            updated_at=now,
        )
    )
    await _beta_queue.enqueue(
        BetaQueueItem(
            job_id=job_id, 
            video_path=video_path, 
            cv_engine=cv_engine,
            llm_engine=llm_engine,
        )
    )
    _metrics.incr("beta.jobs.created")

    return CreateJobResponse(
        job_id=job_id,
        status="pending",
        cv_engine=cv_engine,
        llm_engine=llm_engine,
        quality_profile=quality_profile,
        chunking_interval=chunking_interval,
    )


@app.get("/api/v1beta/jobs/{job_id}", tags=["jobs-beta"])
async def get_beta_job(job_id: str) -> dict[str, Any]:
    rec = _beta_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": rec.job_id,
        "status": rec.status,
        "current_step": rec.current_step,
        "result_path": rec.result_path,
        "tracking_overlay_path": rec.tracking_overlay_path,
        "tracking_data_path": rec.tracking_data_path,
        "error": rec.error,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
    }


@app.get("/api/v1beta/jobs/{job_id}/artifacts", tags=["jobs-beta"])
async def get_beta_job_artifacts(job_id: str) -> dict[str, Any]:
    rec = _beta_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    report_path, overlay_path, tracking_path = _beta_job_artifact_paths(job_id)
    return {
        "job_id": job_id,
        "status": rec.status,
        "report_path": str(report_path) if report_path.is_file() else rec.result_path,
        "tracking_overlay_path": (
            str(overlay_path) if overlay_path.is_file() else rec.tracking_overlay_path
        ),
        "tracking_data_path": (
            str(tracking_path) if tracking_path.is_file() else rec.tracking_data_path
        ),
    }


@app.websocket("/ws/v1beta/jobs/{job_id}")
async def beta_job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    if not verify_ws_auth(websocket):
        await websocket.accept()
        await websocket.send_json(
            {
                "job_id": job_id,
                "status": "error",
                "current_step": "Unauthorized",
                "result_path": None,
                "error": "unauthorized",
            }
        )
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            rec = _beta_store.get(job_id)
            if rec is None:
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "status": "error",
                        "current_step": "Unknown job",
                        "result_path": None,
                        "error": "job_not_found",
                    }
                )
                return
            await websocket.send_json(
                {
                    "job_id": rec.job_id,
                    "status": rec.status,
                    "current_step": rec.current_step,
                    "result_path": rec.result_path,
                    "tracking_overlay_path": rec.tracking_overlay_path,
                    "tracking_data_path": rec.tracking_data_path,
                    "error": rec.error,
                }
            )
            if rec.status in ("done", "error"):
                return
            await asyncio.sleep(0.5)
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/api/v1/jobs/{job_id}/artifacts", tags=["jobs"])
async def get_job_artifacts(job_id: str) -> dict[str, Any]:
    with _job_store_lock:
        rec = _job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    report_path_p, overlay_path_p, tracking_path_p = _job_artifact_paths(job_id)
    event_path_p = BACKEND_ROOT / "output" / f"{job_id}_events.json"
    threat_path_p = BACKEND_ROOT / "output" / f"{job_id}_threat_profiles.json"
    return {
        "job_id": job_id,
        "status": rec.status,
        "report_path": str(report_path_p) if report_path_p.is_file() else rec.result_path,
        "tracking_overlay_path": (
            str(overlay_path_p)
            if overlay_path_p.is_file()
            else rec.tracking_overlay_path
        ),
        "tracking_data_path": (
            str(tracking_path_p)
            if tracking_path_p.is_file()
            else rec.tracking_data_path
        ),
        # ── Event Intelligence Layer artifacts ─────────────────────────────
        "event_index_path": str(event_path_p) if event_path_p.is_file() else rec.event_index_path,
        "threat_profiles_path": str(threat_path_p) if threat_path_p.is_file() else rec.threat_profiles_path,
    }


@app.get("/api/v1/jobs/{job_id}/report/pdf", tags=["jobs"])
async def download_report_pdf(job_id: str, request: Request):
    """Generate and return a professional 7-page PDF Tactical Coaching Report."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match report")

    try:
        from fastapi.responses import StreamingResponse
        from services.pdf_service import PDFService
        
        # Resolve match name
        match_name = f"Match_{job_id[:8]}"
        conn = DatabaseService.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM matches WHERE id = ?", (job_id,))
            row = cursor.fetchone()
            if row:
                match_name = row["name"]
        except Exception:
            pass
        finally:
            conn.close()
            
        pdf_buffer = await asyncio.to_thread(PDFService.generate_report_pdf, job_id, match_name)
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=GaffersGuide_TacticalReport_{job_id}.pdf"
            }
        )
    except Exception as exc:
        LOGGER.exception("Failed to generate PDF for job %s", job_id)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/jobs/{job_id}/events", tags=["jobs"])
async def get_job_events(job_id: str, request: Request) -> dict[str, Any]:
    """
    Return the Event Intelligence Layer output for a completed job verifying ownership.
    """
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match events")

    event_path = BACKEND_ROOT / "output" / f"{job_id}_events.json"
    threat_path = BACKEND_ROOT / "output" / f"{job_id}_threat_profiles.json"

    if not event_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Event index not found. Job may still be processing or event detection was skipped.",
        )

    import json as _json
    with event_path.open(encoding="utf-8") as f:
        event_data = _json.load(f)

    threat_data: list[dict] = []
    if threat_path.is_file():
        with threat_path.open(encoding="utf-8") as f:
            threat_data = _json.load(f)

    # Build summary stats
    from collections import Counter as _Counter
    events = event_data.get("events", [])
    by_category = dict(_Counter(e.get("category", "unknown") for e in events))
    by_type = dict(_Counter(e.get("event_type", "unknown") for e in events).most_common(20))

    # Top threats per team
    team_threats: dict[str, list[dict]] = {}
    for profile in sorted(threat_data, key=lambda p: p.get("threat_score", 0), reverse=True):
        tid = profile.get("team_id", "unknown")
        if tid not in team_threats:
            team_threats[tid] = []
        if len(team_threats[tid]) < 3:
            team_threats[tid].append({
                "player_id": profile["player_id"],
                "threat_score": profile["threat_score"],
                "threat_rank": profile["threat_rank"],
                "primary_threat_types": profile.get("primary_threat_types", []),
                "explanation": profile.get("explanation", ""),
            })

    return {
        "job_id": job_id,
        "event_stats": {
            "total_events": len(events),
            "by_category": by_category,
            "by_type": by_type,
            "players_with_events": len({e.get("player_id") for e in events if e.get("player_id") is not None}),
        },
        "threat_profiles": threat_data,
        "top_threats_by_team": team_threats,
    }


@app.get("/api/v1/jobs/{job_id}/report/enriched", tags=["jobs"])
async def get_job_report_enriched(job_id: str, request: Request) -> list[dict[str, Any]]:
    """
    Return the enriched tactical coaching cards for a completed job verifying ownership.
    """
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match report")

    enriched_report_path = BACKEND_ROOT / "output" / f"{job_id}_report_enriched.json"
    if not enriched_report_path.is_file():
        # Fallback to generating it on the fly if it hasn't been built yet
        # But only if the original report and events index exist.
        report_path = BACKEND_ROOT / "output" / f"{job_id}_report.json"
        event_path = BACKEND_ROOT / "output" / f"{job_id}_events.json"
        
        if not report_path.is_file():
            raise HTTPException(status_code=404, detail="Job report not found. The job may still be processing.")
        if not event_path.is_file():
            raise HTTPException(status_code=404, detail="Event index not found. Event detection may have been skipped.")
            
        try:
            from event_layer.enricher import enrich_report
            await asyncio.to_thread(
                enrich_report,
                report_path=report_path,
                job_id=job_id,
                output_dir=BACKEND_ROOT / "output",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to enrich report: {exc}")

    try:
        with enriched_report_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load enriched report: {exc}")


@app.get("/api/v1/jobs/{job_id}/tracking", tags=["jobs"])
async def get_job_tracking(job_id: str, request: Request) -> dict[str, Any]:
    """Return tracking data verifying ownership."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match tracking data")

    with _job_store_lock:
        rec = _job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    tracking_path = BACKEND_ROOT / "output" / f"{job_id}_tracking_data.json"
    if not tracking_path.is_file():
        # #region agent log
        _agent_debug_ndjson(
            "H1",
            "main.py:get_job_tracking",
            "tracking file missing (425)",
            {
                "job_id_prefix": job_id[:8],
                "rec_status": rec.status,
                "rec_current_step": rec.current_step,
            },
        )
        # #endregion
        raise HTTPException(
            status_code=425,
            detail="Tracking timeline not ready yet. Wait for job completion.",
        )
    with tracking_path.open("r", encoding="utf-8") as f_in:
        payload: dict[str, Any] = json.load(f_in)
    # #region agent log
    _agent_debug_ndjson(
        "H1",
        "main.py:get_job_tracking",
        "tracking served",
        {
            "job_id_prefix": job_id[:8],
            "rec_status": rec.status,
            "frame_keys": len(payload.get("frames", [])) if isinstance(payload.get("frames"), list) else -1,
        },
    )
    # #endregion
    return payload


@app.get("/api/v1/jobs/{job_id}/overlay", tags=["jobs"])
async def get_job_overlay_video(job_id: str, request: Request) -> FileResponse:
    """Return job overlay video path verifying ownership."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match overlay video")

    from services import gcs_service
    if gcs_service.GCS_ENABLED:
        blob_name = gcs_service.export_blob_name(job_id, f"{job_id}_tracking_overlay.mp4")
        if gcs_service.blob_exists(blob_name):
            try:
                signed_url = gcs_service.generate_signed_url(blob_name, expiration_seconds=900)
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=signed_url)
            except Exception as s_err:
                LOGGER.error("Failed to generate GCS signed URL: %s. Falling back to streaming.", s_err)

    with _job_store_lock:
        rec = _job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    overlay_path = BACKEND_ROOT / "output" / f"{job_id}_tracking_overlay.mp4"
    if not overlay_path.is_file():
        raise HTTPException(
            status_code=425,
            detail="Tracking overlay video not ready yet. Wait for job completion.",
        )
    return FileResponse(str(overlay_path), media_type="video/mp4", filename=overlay_path.name)




@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
    tags=["coaching"],
)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """
    Generate follow-up coaching text.

    If `job_id` is provided, include the job's report insights as context.
    When a job is active, the LLM always receives match telemetry — intent detection
    only modifies the prompt framing, not the context injection.
    """
    user_id = request.state.user["sub"]
    if req.job_id:
        from services.db_service import DatabaseService
        if not DatabaseService.check_match_ownership(req.job_id, user_id):
            raise HTTPException(status_code=403, detail="Forbidden: You do not own this match")

    from scripts.llm_router import detect_intent

    LOGGER.error(
        "ENGINE DEBUG provider=%s quality=%s mode=%s local=%s  [endpoint=POST /api/v1/chat req.llm_engine=%r req.job_id=%r]",
        req.llm_engine or "(not sent)", "n/a", "chat_entry", req.llm_engine == "local",
        req.llm_engine, req.job_id,
    )
    intent = await detect_intent(req.message)

    prompt_context = ""
    _is_cloud_run = bool(
        os.getenv("K_SERVICE", "").strip() or os.getenv("K_REVISION", "").strip()
    )
    # Default: use req.llm_engine if provided, else "cloud" on Cloud Run, "local" locally.
    selected_llm_engine: LLMEngine = req.llm_engine or ("cloud" if _is_cloud_run else "local")
    # Hard override: if Cloud Run AND "local" somehow arrived (stale localStorage, old client),
    # switch to "cloud". The `or` above doesn't catch this because "local" is truthy.
    if selected_llm_engine == "local" and _is_cloud_run:
        LOGGER.warning(
            "chat: llm_engine='local' received but K_SERVICE/K_REVISION is set "
            "(Cloud Run). Overriding to 'cloud' — Ollama is not available on Cloud Run."
        )
        selected_llm_engine = "cloud"
    LOGGER.info(
        "LLM ENGINE DEBUG: provider=%s quality=%s mode=%s (req_engine=%r cloud_run=%s)",
        selected_llm_engine, "n/a", "chat", req.llm_engine, _is_cloud_run,
    )

    evidence_attachment = None
    if intent in ("evidence_request", "threat_query") and req.job_id:
        from event_layer.chat_evidence import build_evidence_response
        evidence_attachment = await asyncio.to_thread(
            build_evidence_response,
            message=req.message,
            job_id=req.job_id,
            output_dir=BACKEND_ROOT / "output",
        )
        if evidence_attachment:
            clip_lines = []
            for i, clip in enumerate(evidence_attachment.clips, 1):
                clip_lines.append(
                    f"Clip {i}: {clip['label']} (confidence: {clip['confidence_pct']}%) [Frame {clip['start_frame']}-{clip['end_frame']}]"
                )
            
            threat_lines = []
            for t in evidence_attachment.top_threats:
                threat_lines.append(
                    f"Player {t['player_id']} (team: {t['team_id']}) - Threat Score: {t['threat_score']:.1f}/100. Explanation: {t['explanation']}"
                )
            
            evidence_str = "\n".join(clip_lines)
            threats_str = "\n".join(threat_lines)
            
            prompt_context += (
                f"\n[Retrieved Match Clips & Evidence]\n"
                f"{evidence_str if evidence_str else 'No direct clip matches found.'}\n"
                f"\n[Retrieved Player Threat Profiles]\n"
                f"{threats_str if threats_str else 'No threat profiles found.'}\n"
            )

    # Inject match telemetry whenever a job is active.
    # Even general questions get match context so the LLM can ground its response.
    if req.job_id:
        with _job_store_lock:
            rec = _job_store.get(req.job_id)
            job_status = rec.status if rec else None
            report_path = rec.result_path if rec else None

        if rec and job_status in ("done", "processing", "error"):
            if req.llm_engine is None:
                selected_llm_engine = rec.llm_engine  # type: ignore[assignment]

            if not report_path:
                report_path = str(BACKEND_ROOT / "output" / f"{req.job_id}_report.json")

            try:
                with Path(report_path).open("r", encoding="utf-8") as f:
                    report_cards: list[dict[str, Any]] = json.load(f)
                
                top = report_cards[:6]
                lines: list[str] = []
                for item in top:
                    team = str(item.get("team", "?"))
                    flaw = str(item.get("flaw", "?"))
                    severity = str(item.get("severity", ""))
                    evidence = str(item.get("evidence", "")).strip()
                    instruction = str(item.get("tactical_instruction") or "").strip()
                    lines.append(f"- {team}: {flaw} ({severity}) — {evidence}")
                    if instruction:
                        lines.append(f"  Coaching: {instruction[:150]}")

                if intent == "general":
                    # For general questions, still anchor to the match but frame as education.
                    prompt_context = (
                        f"[Current match analysis available — use for context]\n"
                        + "\n".join(lines)
                        + "\n\nThe user is asking a general question. Answer it in the context of this specific match where relevant."
                        + ("\n" + prompt_context if prompt_context else "")
                    )
                elif intent in ("evidence_request", "threat_query"):
                    prompt_context = (
                        "Match-specific tactical intelligence:\n"
                        + "\n".join(lines)
                        + "\n\nRetrieved match evidence context for this visual request:\n"
                        + prompt_context
                    )
                else:
                    prompt_context = "Match-specific tactical intelligence:\n" + "\n".join(lines)
            except Exception:
                # Fallback if report is missing — continue with intent-only mode
                if intent == "tactical" or intent in ("evidence_request", "threat_query"):
                    prompt_context = (
                        f"[Job {req.job_id[:8]} report not available yet. "
                        "Answer tactically based on the user's question.]"
                        + ("\n" + prompt_context if prompt_context else "")
                    )

    full_prompt = build_structured_coaching_prompt(
        user_prompt=req.message,
        context=prompt_context,
        history=req.history,
    )

    try:
        reply = await get_tactical_advice(full_prompt, selected_llm_engine)
        if req.job_id:
            try:
                from services.db_service import DatabaseService
                import uuid as _uuid
                chat_msg_id = _uuid.uuid4().hex
                DatabaseService.save_chat_message(
                    id=chat_msg_id,
                    user_id=user_id,
                    job_id=req.job_id,
                    message=req.message,
                    reply=reply
                )
            except Exception as db_err:
                LOGGER.error("Failed to save chat message to SQLite: %s", db_err)
    except EngineRoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Chat completion failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(reply=reply, evidence=evidence_attachment)



@app.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """
    Stream job progress updates over WebSockets.

    Sends JSON messages shaped like:
      { job_id, status: pending|processing|done|error, current_step, result_path?, error? }
    """
    if not verify_ws_auth(websocket):
        await websocket.accept()
        await websocket.send_json(
            {
                "job_id": job_id,
                "status": "error",
                "current_step": "Unauthorized",
                "result_path": None,
                "error": "unauthorized",
            }
        )
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        while True:
            with _job_store_lock:
                rec = _job_store.get(job_id)

            if rec is None:
                await websocket.send_json(
                    {
                        "job_id": job_id,
                        "status": "error",
                        "current_step": "Unknown job",
                        "result_path": None,
                        "error": "job_not_found",
                    }
                )
                return

            await websocket.send_json(
                {
                    "job_id": rec.job_id,
                    "status": rec.status,
                    "current_step": rec.current_step,
                    "result_path": rec.result_path,
                    "tracking_overlay_path": rec.tracking_overlay_path,
                    "tracking_data_path": rec.tracking_data_path,
                    "error": rec.error,
                }
            )

            if rec.status in ("done", "error"):
                return

            await asyncio.sleep(0.5)
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("WebSocket error for job %s: %s", job_id, exc)
        try:
            await websocket.send_json(
                {
                    "job_id": job_id,
                    "status": "error",
                    "current_step": "WebSocket error",
                    "result_path": None,
                    "error": str(exc),
                }
            )
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def _resolve_llm_credentials() -> tuple[str | None, str, str | None]:
    """Return (api_key, model, base_url) for OpenAI-compatible APIs."""

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    return api_key, model, base_url


async def _complete_coaching_instruction(
    client: AsyncOpenAI | None,
    model: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Return (content, error_message)."""

    if client is None:
        return None, None

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.35,
                max_tokens=600,
            )
            choice = response.choices[0].message.content
            return (choice.strip() if choice else None, None)
        except Exception as exc:
            LOGGER.exception("LLM completion failed")
            return None, str(exc)


async def _complete_coaching_instruction_gemini(
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> tuple[str | None, str | None]:
    """Return (content, error_message) using Google Gemini."""

    async with semaphore:
        try:
            text = await asyncio.to_thread(generate_coaching_advice, user_prompt)
            return (text if text else None, None)
        except Exception as exc:
            LOGGER.exception("Gemini completion failed")
            return None, str(exc)


@app.get("/health", tags=["meta"])
async def health() -> dict[str, Any]:
    """Deep production health check for Cloud Run deployment and liveness probes."""
    checks = {
        "database": "unknown",
        "gemini": "not_configured",
        "gcs": "not_configured",
        "ollama": "unknown",
    }
    status = "ok"

    # 1. Database Connectivity (Critical check)
    try:
        from services.db_service import DatabaseService
        conn = DatabaseService.get_connection()
        conn.execute("SELECT 1;")
        conn.close()
        checks["database"] = "ok"
    except Exception as db_err:
        LOGGER.error("Health check failure (database): %s", db_err)
        checks["database"] = "error"
        status = "unhealthy"  # Hard failure for local database issues

    # 2. Gemini configuration (Degraded check)
    if gemini_is_configured():
        checks["gemini"] = "configured"
    else:
        # Not configured is acceptable if using local Ollama, so degraded instead of unhealthy
        checks["gemini"] = "not_configured"
        if status == "ok":
            status = "degraded"

    # 3. GCS Sync status (Degraded check)
    try:
        from services import gcs_service
        if gcs_service.GCS_ENABLED:
            # Force GCS client validation
            gcs_service._get_client()
            checks["gcs"] = "ok"
        else:
            checks["gcs"] = "local_fallback"
    except Exception as gcs_err:
        LOGGER.warning("Health check warning (GCS): %s", gcs_err)
        checks["gcs"] = "error"
        if status == "ok":
            status = "degraded"

    # 4. Ollama liveness (Soft / local-only check)
    if os.getenv("K_SERVICE", "").strip():
        # Cloud Run does not run Ollama locally
        checks["ollama"] = "cloud_run_skip"
    else:
        try:
            # Direct quick socket check to avoid slow timeouts
            import socket
            with socket.create_connection(("127.0.0.1", 11434), timeout=0.1):
                checks["ollama"] = "running"
        except Exception:
            checks["ollama"] = "not_running"

    # Calculate active job count
    active_jobs = 0
    with _job_store_lock:
        active_jobs = sum(1 for rec in _job_store.values() if rec.status in ("pending", "processing"))

    uptime = time.time() - _START_TIME

    # Set response code
    response_status = 200
    if status == "unhealthy":
        from fastapi import Response
        # Return 503 Service Unavailable for critical failures
        return Response(
            status_code=503,
            content=json.dumps({
                "status": "unhealthy",
                "checks": checks,
                "uptime_seconds": round(uptime, 1),
            }),
            media_type="application/json",
        )

    return {
        "status": status,
        "checks": checks,
        "metrics": {
            "gemini_failure_rate_5m_pct": gemini_monitor.failure_rate_pct(),
            "upload_failure_rate_5m_pct": upload_monitor.failure_rate_pct(),
            "active_jobs": active_jobs,
        },
        "uptime_seconds": round(uptime, 1),
        "revision": os.getenv("K_REVISION", "local"),
        "correlation_id": get_correlation_id(),
    }


@app.get("/api/v1/meta/metrics", tags=["meta"])
async def production_metrics(request: Request) -> dict[str, Any]:
    """Exposes all pipeline performance counters and failure rate monitors.

    Requires authentication via API key.
    """
    # Simple api key verification (similar to _auth_middleware)
    api_key_env = (os.getenv("API_KEY") or "").strip()
    req_key = (request.headers.get("x-api-key") or request.query_params.get("api_key") or "").strip()
    if api_key_env and req_key != api_key_env:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")

    snapshot = _metrics.snapshot()
    active_jobs = 0
    with _job_store_lock:
        active_jobs = sum(1 for rec in _job_store.values() if rec.status in ("pending", "processing"))

    return {
        "metrics": snapshot,
        "monitors": {
            "gemini_failure_rate_5m_pct": gemini_monitor.failure_rate_pct(),
            "gemini_total_calls_5m": gemini_monitor.total_in_window(),
            "upload_failure_rate_5m_pct": upload_monitor.failure_rate_pct(),
            "upload_total_events_5m": upload_monitor.total_in_window(),
        },
        "active_jobs": active_jobs,
        "uptime_seconds": round(time.time() - _START_TIME, 1),
    }


@app.get("/api/v1beta/metrics", tags=["meta-beta"])
async def beta_metrics() -> dict[str, Any]:
    """Return beta pipeline metrics snapshot for baseline and promotion gates."""
    snapshot = _metrics.snapshot()
    counters = snapshot.get("counters", {})
    succeeded = int(counters.get("beta.jobs.succeeded", 0)) if isinstance(counters, dict) else 0
    failed = int(counters.get("beta.jobs.failed", 0)) if isinstance(counters, dict) else 0
    total = succeeded + failed
    success_rate = (succeeded / total * 100.0) if total else 0.0
    gates = {
        "job_success_rate_pct": round(success_rate, 2),
        "minimum_required_success_rate_pct": 95.0,
        "pass": success_rate >= 95.0 if total else False,
    }
    return {"snapshot": snapshot, "promotion_gate": gates}



def _card_needs_local_llm_refresh(card: dict[str, Any]) -> bool:
    """True when the card has a prompt but no successful coaching text yet."""
    prompt = card.get("llm_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    tip = card.get("tactical_instruction")
    if isinstance(tip, str) and tip.strip():
        return False
    return True


async def _refresh_job_report_cards_with_local_llm(
    report_cards: list[dict[str, Any]],
    *,
    llm_concurrency: int,
) -> list[dict[str, Any]]:
    """Re-run LLM for job report rows using Ollama when the pipeline skipped cloud keys."""
    # #region agent log
    _t0 = datetime.now(timezone.utc)
    _agent_debug_ndjson(
        "C",
        "main.py:_refresh_job_report_cards_with_local_llm",
        "refresh_enter",
        {"card_count": len(report_cards), "llm_concurrency": llm_concurrency},
    )
    # #endregion
    # ── Cloud Run guard ──────────────────────────────────────────────────────
    # On Cloud Run (K_SERVICE is set) there is no local Ollama daemon.
    # Return cards unchanged so the API does not crash — use cloud LLM engine.
    from services.ollama_client import OLLAMA_AUTO_START_IN_CLOUD_ENV, _env_truthy
    if os.getenv("K_SERVICE", "").strip() and not _env_truthy(OLLAMA_AUTO_START_IN_CLOUD_ENV):
        LOGGER.info(
            "_refresh_job_report_cards_with_local_llm: Cloud Run detected; "
            "skipping local Ollama refresh (set OLLAMA_AUTO_START_IN_CLOUD=1 to override)."
        )
        return report_cards
    # ────────────────────────────────────────────────────────────────────────
    await ensure_ollama_available()
    semaphore = asyncio.Semaphore(llm_concurrency)

    async def _one(card: dict[str, Any]) -> dict[str, Any]:
        if not _card_needs_local_llm_refresh(card):
            return card
        prompt = card["llm_prompt"]
        if not isinstance(prompt, str):
            return card
        async with semaphore:
            try:
                text = await get_tactical_advice(prompt, "local")
                return {**card, "tactical_instruction": text, "llm_error": None}
            except EngineRoutingError as exc:
                return {
                    **card,
                    "llm_error": f"{exc.code}: {exc.message}",
                }
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Local Ollama completion failed for job report card")
                return {**card, "llm_error": str(exc)}

    out = list(await asyncio.gather(*[_one(c) for c in report_cards]))
    # #region agent log
    _ms = int((datetime.now(timezone.utc) - _t0).total_seconds() * 1000)
    _agent_debug_ndjson(
        "C",
        "main.py:_refresh_job_report_cards_with_local_llm",
        "refresh_exit",
        {"elapsed_ms": _ms, "card_count": len(out)},
    )
    # #endregion
    return out


@app.get(
    "/api/v1/coach/advice",
    response_model=CoachAdviceResponse,
    tags=["coaching"],
)
async def get_coach_advice(
    job_id: Annotated[
        str | None,
        Query(description="If provided, load advice from the job's report file."),
    ] = None,
    skip_llm: Annotated[
        bool,
        Query(description="If true, build prompts only and skip remote LLM calls."),
    ] = False,
    llm_concurrency: Annotated[
        int,
        Query(ge=1, le=16, description="Max parallel LLM requests."),
    ] = 4,
    llm_engine: Annotated[
        LLMEngine,
        Query(description="Route LLM completion to cloud API or local Ollama."),
    ] = "cloud",
) -> CoachAdviceResponse:
    """
    Run the tactical pipeline: metrics → triggers → RAG prompts → optional LLM completions.

    Requires ``backend/output/tactical_metrics.json`` from upstream analytics.
    Set ``GEMINI_API_KEY`` for Google Gemini (preferred), or ``LLM_API_KEY`` /
    ``OPENAI_API_KEY`` for OpenAI-compatible APIs.

    When ``job_id`` is set and ``llm_engine=local``, cards that have ``llm_prompt``
    but no ``tactical_instruction`` (e.g. cloud keys were missing at job time) are
    completed with local Ollama on read.
    """

    generated_at = datetime.now(timezone.utc).isoformat()
    pipeline: dict[str, Any] = {
        "rule_engine": "pending",
        "rag_synthesizer": "pending",
        "llm": "skipped" if skip_llm else "pending",
    }

    # Job mode: load already computed report cards for this job id.
    if job_id:
        with _job_store_lock:
            rec = _job_store.get(job_id)
            status = rec.status if rec else None
            report_path = rec.result_path if rec else None

        if not report_path:
            report_path = str(BACKEND_ROOT / "output" / f"{job_id}_report.json")

        report_file = Path(report_path)
        if not report_file.is_file():
            if status in (None, "pending"):
                raise HTTPException(status_code=404, detail="Job report not found yet.")
            raise HTTPException(status_code=425, detail="Job report not ready yet.")

        with report_file.open("r", encoding="utf-8") as f:
            report_cards: list[dict[str, Any]] = json.load(f)

        llm_skip_reason: str | None = None
        needs_local_refresh = any(_card_needs_local_llm_refresh(c) for c in report_cards)
        # #region agent log
        _agent_debug_ndjson(
            "D",
            "main.py:get_coach_advice",
            "job_report_loaded",
            {
                "job_id_prefix": job_id[:8],
                "skip_llm": skip_llm,
                "llm_engine": llm_engine,
                "cards_len": len(report_cards),
                "needs_local_refresh": needs_local_refresh,
                "will_run_refresh": (
                    not skip_llm and llm_engine == "local" and needs_local_refresh
                ),
            },
        )
        # #endregion
        if not skip_llm and llm_engine == "local" and needs_local_refresh and not os.getenv("K_SERVICE", "").strip():
            try:
                report_cards = await _refresh_job_report_cards_with_local_llm(
                    report_cards,
                    llm_concurrency=llm_concurrency,
                )
            except EngineRoutingError as exc:
                llm_skip_reason = f"{exc.code}: {exc.message}"
                LOGGER.warning(
                    "Job %s: local LLM refresh skipped (%s)",
                    job_id,
                    llm_skip_reason,
                )

        # Determine whether LLM produced text.
        llm_ok = any(bool(c.get("tactical_instruction")) for c in report_cards)
        if llm_skip_reason:
            pipeline_llm = f"skipped ({llm_skip_reason})"
        elif llm_ok:
            pipeline_llm = f"ok ({llm_engine})"
        else:
            pipeline_llm = "skipped"
        pipeline = {
            "rule_engine": "ok",
            "rag_synthesizer": "ok",
            "llm": pipeline_llm,
        }

        advice_items: list[CoachingAdviceItem] = []
        for card in report_cards:
            advice_items.append(
                CoachingAdviceItem(
                    frame_idx=int(card.get("frame_idx", 0)),
                    team=str(card.get("team", "")),
                    flaw=str(card.get("flaw", "")),
                    severity=str(card.get("severity", "")),
                    evidence=str(card.get("evidence", "")),
                    matched_philosophy_author=str(
                        card.get("matched_philosophy_author", "")
                    ),
                    fc25_player_roles=card.get("fc_role_recommendations"),
                    tactical_instruction=format_numbered_steps(
                        normalize_instruction_steps(card.get("tactical_instruction")),
                        card.get("tactical_instruction"),
                    ),
                    tactical_instruction_steps=normalize_instruction_steps(
                        card.get("tactical_instruction")
                    ),
                    llm_error=card.get("llm_error"),
                    confidence_pct=card.get("confidence_pct"),
                    confidence_reason=card.get("confidence_reason"),
                    summary_data=card.get("summary_data"),
                )
            )

        # Fetch telemetry to include in the report for Radar restoration
        telemetry_data = None
        if job_id:
            tracking_path = JOBS_DIR / f"{job_id}_tracking_data.json"
            if tracking_path.exists():
                try:
                    with open(tracking_path, "r") as f:
                        telemetry_data = json.load(f)
                except Exception:
                    pass

        response = CoachAdviceResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            pipeline=pipeline,
            advice_items=advice_items,
            job_id=job_id,
            telemetry=telemetry_data
        )
        
        # Auto-save report on completion if it's a job
        if job_id:
            ReportService.save_report(response.model_dump())
            
        return response

    try:
        await asyncio.to_thread(run_engine)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "message": str(exc),
                "hint": "Produce tactical_metrics.json (e.g. via your analytics pipeline) under backend/output/.",
            },
        ) from exc

    pipeline["rule_engine"] = "success"

    try:
        records = await asyncio.to_thread(run_rag_synthesizer)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    pipeline["rag_synthesizer"] = "success"

    # ── Cloud Run safety guard ─────────────────────────────────────────────
    # On Cloud Run (K_SERVICE or K_REVISION set) there is no local Ollama daemon.
    # If the request still arrives with llm_engine="local" (e.g. stale frontend
    # default, cached localStorage value), override silently to "cloud" and skip
    # the Ollama preflight entirely.  Never call ensure_ollama_available() here.
    _on_cloud_run = bool(
        os.getenv("K_SERVICE", "").strip() or os.getenv("K_REVISION", "").strip()
    )
    if llm_engine == "local" and _on_cloud_run:
        LOGGER.warning(
            "get_coach_advice: llm_engine='local' received but K_SERVICE/K_REVISION is set "
            "(Cloud Run environment). Overriding to 'cloud' — Ollama is not available on "
            "Cloud Run. Set OLLAMA_AUTO_START_IN_CLOUD=1 only for sidecar Ollama setups."
        )
        llm_engine = "cloud"
    # ──────────────────────────────────────────────────────────────────────
    if not skip_llm and llm_engine == "local":
        try:
            await ensure_ollama_available()
        except EngineRoutingError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc

    semaphore = asyncio.Semaphore(llm_concurrency)

    async def _complete_routed(
        user_prompt: str,
        gate: asyncio.Semaphore,
    ) -> tuple[str | None, str | None]:
        async with gate:
            try:
                text = await get_tactical_advice(user_prompt, llm_engine)
                return text, None
            except EngineRoutingError as exc:
                return None, f"{exc.code}: {exc.message}"
            except Exception as exc:  # noqa: BLE001
                return None, str(exc)

    if skip_llm:
        llm_results: list[tuple[str | None, str | None]] = [
            (None, None) for _ in records
        ]
    else:
        llm_tasks = [_complete_routed(rec.llm_prompt, semaphore) for rec in records]
        llm_results = await asyncio.gather(*llm_tasks)

    if skip_llm:
        pipeline["llm"] = "skipped_by_query"
    else:
        pipeline["llm"] = f"ok ({llm_engine})"

    advice_items: list[CoachingAdviceItem] = []
    for rec, (instruction, err) in zip(records, llm_results, strict=True):
        instruction_steps = normalize_instruction_steps(instruction)
        advice_items.append(
            CoachingAdviceItem(
                frame_idx=rec.frame_idx,
                team=rec.team,
                flaw=rec.flaw,
                severity=rec.severity,
                evidence=rec.evidence,
                matched_philosophy_author=rec.matched_philosophy_author,
                fc25_player_roles=rec.fc_role_recommendations,
                tactical_instruction=format_numbered_steps(
                    instruction_steps,
                    instruction,
                ),
                tactical_instruction_steps=instruction_steps,
                llm_error=err,
            )
        )

    return CoachAdviceResponse(
        generated_at=generated_at,
        pipeline=pipeline,
        advice_items=advice_items,
    )


# ── Match History & Archive Endpoints ──────────────────────────────────────────

@app.get("/api/v1/jobs/{job_id}/timeline", tags=["matches"])
async def get_tactical_timeline(job_id: str, request: Request):
    """Generates and returns the segmented tactical timeline for a completed job."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match")

    try:
        from services.timeline_service import TimelineService
        segments = TimelineService.generate_timeline(job_id)
        if not segments:
            raise HTTPException(status_code=404, detail="Timeline not found or metrics file missing. Ensure job is completed.")
        return segments
    except HTTPException:
        raise
    except Exception as err:
        LOGGER.error("Error generating timeline for job %s: %s", job_id, err)
        raise HTTPException(status_code=500, detail=str(err))


@app.get("/api/v1/matches", tags=["matches"])
async def list_matches(request: Request, search: str = None, sort: str = "newest"):
    """Returns paginated, searchable, sorted matches from SQLite database."""
    user_id = request.state.user["sub"]
    try:
        from services.db_service import DatabaseService
        return DatabaseService.list_matches(user_id=user_id, search=search, sort_by=sort)
    except Exception as db_err:
        LOGGER.error("DB Error listing matches: %s", db_err)
        raise HTTPException(status_code=500, detail=str(db_err))


@app.delete("/api/v1/matches/{match_id}", tags=["matches"])
async def delete_match(match_id: str, request: Request):
    """Deletes a match record and unlinks all associated files on disk."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(match_id, user_id):
        raise HTTPException(status_code=403, detail="Forbidden: You do not own this match")

    try:
        success = DatabaseService.delete_match(match_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Match not found")

        # Unlink output files on disk
        report_p, overlay_p, tracking_p = _job_artifact_paths(match_id)
        for p in (report_p, overlay_p, tracking_p):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

        # Unlink other event files
        for suffix in ("_events.json", "_threat_profiles.json", "_report_enriched.json"):
            p = BACKEND_ROOT / "output" / f"{match_id}{suffix}"
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as db_err:
        LOGGER.error("DB Error deleting match %s: %s", match_id, db_err)
        raise HTTPException(status_code=500, detail=str(db_err))


@app.post("/api/v1/matches/{match_id}/reanalyze", tags=["matches"])
async def reanalyze_match(match_id: str, request: Request):
    """Re-runs the analysis pipeline using preserved video uploads."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    match = DatabaseService.get_match(match_id, user_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found or access denied")

    try:
        video_filename = match["video_filename"]
        # Find video path in uploads or main data directory
        video_path = BACKEND_ROOT / "data" / "uploads" / video_filename
        if not video_path.is_file():
            video_path = BACKEND_ROOT / "data" / video_filename
            if not video_path.is_file():
                raise HTTPException(
                    status_code=400,
                    detail=f"Source video file '{video_filename}' not found. Cannot re-analyze."
                )

        # Reset database status
        DatabaseService.update_match_status(match_id, "pending")
        DatabaseService.update_job_status_db(match_id, "pending", "Pending")

        # Sync memory store
        with _job_store_lock:
            _job_store[match_id] = JobRecord(
                job_id=match_id,
                status="pending",
                current_step="Pending",
                cv_engine=match["cv_engine"],
                llm_engine=match["llm_engine"],
                quality_profile=match["quality_profile"],
                chunking_interval=match["chunking_interval"],
            )

        # Trigger parallel CV pipeline execution task
        asyncio.create_task(_run_job(match_id, video_path, match["cv_engine"], user_id=user_id))
        return {"status": "reanalysis_started", "job_id": match_id}
    except HTTPException:
        raise
    except Exception as db_err:
        LOGGER.error("DB Error starting reanalysis for match %s: %s", match_id, db_err)
        raise HTTPException(status_code=500, detail=str(db_err))


# ── Player Mappings Endpoints ────────────────────────────────────────────────

@app.get("/api/v1/player-mappings/{job_id}", tags=["player-mappings"])
async def get_player_mappings(job_id: str, request: Request):
    """Retrieve player mapping identities for a specific match owned by user."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    # Scoping security: check if match belongs to user
    if not DatabaseService.check_match_ownership(job_id, user_id):
         raise HTTPException(status_code=403, detail="Forbidden: You do not own this match mapping")

    mapping = DatabaseService.get_player_mapping(job_id, user_id)
    if not mapping:
        return {"job_id": job_id, "mappings": {}}
    
    import json as _json
    try:
        data = _json.loads(mapping["mapping_data"])
        return data
    except Exception:
        return {"job_id": job_id, "mappings": {}}


@app.post("/api/v1/player-mappings/{job_id}", tags=["player-mappings"])
async def save_player_mappings(job_id: str, payload: dict, request: Request):
    """Save player mapping identities for a specific match verifying ownership."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
         raise HTTPException(status_code=403, detail="Forbidden: You do not own this match mapping")

    import json as _json
    mapping_str = _json.dumps(payload)
    DatabaseService.save_player_mapping(job_id, user_id, mapping_str)
    return {"status": "saved", "job_id": job_id}


# ── Chat History Endpoints ───────────────────────────────────────────────────

@app.get("/api/v1/chat/history/{job_id}", tags=["chat"])
async def get_chat_logs(job_id: str, request: Request):
    """List persisted chat interaction history for a match owned by user."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    if not DatabaseService.check_match_ownership(job_id, user_id):
         raise HTTPException(status_code=403, detail="Forbidden: You do not own this match chat logs")

    logs = DatabaseService.get_chat_history(job_id, user_id)
    return {"history": logs}


# ── Jobs DB Status Endpoint ──────────────────────────────────────────────────

@app.get("/api/v1/jobs/{job_id}/status", tags=["jobs"])
async def get_job_status(job_id: str, request: Request):
    """Query current pipeline job state verifying ownership."""
    user_id = request.state.user["sub"]
    from services.db_service import DatabaseService
    
    # 1. Try DB first (persisted jobs)
    job = DatabaseService.get_job(job_id, user_id)
    if job:
        return {
            "job_id": job["id"],
            "status": job["status"],
            "current_step": job["current_step"],
            "error": job["error"],
            "result_path": job["result_path"],
            "tracking_overlay_path": job["tracking_overlay_path"],
            "tracking_data_path": job["tracking_data_path"]
        }
        
    # 2. Fallback to in-memory store if not yet flushed to DB (e.g. legacy/fast checks)
    with _job_store_lock:
        rec = _job_store.get(job_id)
        
    # Owner verification check on memory fallback: check if match belongs to user
    if rec and DatabaseService.check_match_ownership(job_id, user_id):
        return {
            "job_id": rec.job_id,
            "status": rec.status,
            "current_step": rec.current_step,
            "error": rec.error,
            "result_path": rec.result_path,
            "tracking_overlay_path": rec.tracking_overlay_path,
            "tracking_data_path": rec.tracking_data_path
        }

    raise HTTPException(status_code=404, detail="Job not found or access denied")



