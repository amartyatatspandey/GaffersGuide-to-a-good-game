"""
FastAPI entrypoint for the AI Coaching Engine pipeline.

Run from the ``backend`` directory::

    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from fastapi import File, FastAPI, Form, HTTPException, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from llm_service import gemini_is_configured, generate_coaching_advice
from models import ChatRequest, ChatResponse, CreateJobResponse, ReportEntry, ReportsResponse
from services.cv_router import CVEngine, CVRouterFactory
from services.errors import EngineRoutingError
from services.llm_router import LLMEngine, ensure_ollama_available, get_tactical_advice
from scripts.rag_coach import run as run_rag_synthesizer
from scripts.tactical_rule_engine import run_engine

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_ROOT / ".env")

LOGGER = logging.getLogger(__name__)

app = FastAPI(
    title="Gaffer's Guide — Coaching API",
    version="1.0.0",
    description="Tactical rule engine + RAG + optional LLM coaching advice.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@dataclass(slots=True)
class JobRecord:
    job_id: str
    status: str
    current_step: str
    cv_engine: CVEngine
    llm_engine: LLMEngine
    result_path: str | None = None
    tracking_overlay_path: str | None = None
    tracking_data_path: str | None = None
    error: str | None = None


_job_store: dict[str, JobRecord] = {}
_job_store_lock = threading.Lock()


def _job_artifact_paths(job_id: str) -> tuple[Path, Path, Path]:
    output_dir = BACKEND_ROOT / "output"
    report_path = output_dir / f"{job_id}_report.json"
    overlay_path = output_dir / f"{job_id}_tracking_overlay.mp4"
    tracking_path = output_dir / f"{job_id}_tracking_data.json"
    return report_path, overlay_path, tracking_path


async def _run_job(job_id: str, video_path: Path, cv_engine: CVEngine) -> None:
    def progress_callback(step: str) -> None:
        with _job_store_lock:
            rec = _job_store.get(job_id)
            if not rec:
                return
            rec.current_step = step
            if step == "Completed":
                rec.status = "done"

    with _job_store_lock:
        rec = _job_store.get(job_id)
        if rec:
            rec.status = "processing"
            rec.current_step = "Tracking Players"

    try:
        runner = CVRouterFactory.get(cv_engine)
        report_path = await runner.run(
            job_id=job_id,
            video_path=video_path,
            progress_callback=progress_callback,
        )

        with _job_store_lock:
            rec = _job_store.get(job_id)
            if rec:
                report_path_p, overlay_path_p, tracking_path_p = _job_artifact_paths(job_id)
                rec.status = "done"
                rec.current_step = "Completed"
                rec.result_path = str(report_path)
                rec.tracking_overlay_path = (
                    str(overlay_path_p) if overlay_path_p.is_file() else None
                )
                rec.tracking_data_path = (
                    str(tracking_path_p) if tracking_path_p.is_file() else None
                )
    except EngineRoutingError as exc:
        LOGGER.exception("Job %s failed with routing error", job_id)
        with _job_store_lock:
            rec = _job_store.get(job_id)
            if rec:
                rec.status = "error"
                rec.current_step = "Error"
                rec.error = f"{exc.code}: {exc.message}"
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Job %s failed", job_id)
        with _job_store_lock:
            rec = _job_store.get(job_id)
            if rec:
                rec.status = "error"
                rec.current_step = "Error"
                rec.error = str(exc)


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


class CoachAdviceResponse(BaseModel):
    """Frontend-ready payload after running the full pipeline."""

    generated_at: str = Field(description="UTC ISO-8601 timestamp.")
    pipeline: dict[str, Any] = Field(
        description="Summary of steps executed (rule engine, RAG, LLM).",
    )
    advice_items: list[CoachingAdviceItem]


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


@app.post(
    "/api/v1/jobs",
    response_model=CreateJobResponse,
    tags=["jobs"],
)
async def create_job(
    file: UploadFile = File(...),
    cv_engine: CVEngine = Form("cloud"),
    llm_engine: LLMEngine = Form("cloud"),
) -> CreateJobResponse:
    """
    Create a new analytics job by uploading a match video.

    The heavy CV→Math→Rules→RAG→LLM pipeline runs asynchronously; progress is
    published over WebSockets (see `/ws/jobs/{job_id}`).
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(
            status_code=400,
            detail={"message": "Only .mp4 uploads are supported for now."},
        )

    job_id = uuid.uuid4().hex
    upload_dir = BACKEND_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / f"{job_id}.mp4"

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
        )

    asyncio.create_task(_run_job(job_id, video_path, cv_engine))

    return CreateJobResponse(
        job_id=job_id,
        status="pending",
        cv_engine=cv_engine,
        llm_engine=llm_engine,
    )


@app.get("/api/v1/jobs/{job_id}/artifacts", tags=["jobs"])
async def get_job_artifacts(job_id: str) -> dict[str, Any]:
    with _job_store_lock:
        rec = _job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    report_path_p, overlay_path_p, tracking_path_p = _job_artifact_paths(job_id)
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
    }


@app.get("/api/v1/jobs/{job_id}/tracking", tags=["jobs"])
async def get_job_tracking(job_id: str) -> dict[str, Any]:
    with _job_store_lock:
        rec = _job_store.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    tracking_path = BACKEND_ROOT / "output" / f"{job_id}_tracking_data.json"
    if not tracking_path.is_file():
        raise HTTPException(
            status_code=425,
            detail="Tracking timeline not ready yet. Wait for job completion.",
        )
    with tracking_path.open("r", encoding="utf-8") as f_in:
        payload: dict[str, Any] = json.load(f_in)
    return payload


@app.get("/api/v1/jobs/{job_id}/overlay", tags=["jobs"])
async def get_job_overlay_video(job_id: str) -> FileResponse:
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
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Generate follow-up coaching text.

    If `job_id` is provided, include the job's report insights as context.
    """
    prompt_context = ""
    selected_llm_engine: LLMEngine = (req.llm_engine or "cloud")
    if req.job_id:
        with _job_store_lock:
            rec = _job_store.get(req.job_id)
            job_status = rec.status if rec else None
            report_path = rec.result_path if rec else None

        if not rec or job_status not in ("done", "processing", "error"):
            raise HTTPException(status_code=404, detail="Job not found.")
        if req.llm_engine is None:
            selected_llm_engine = rec.llm_engine

        if not report_path:
            report_path = str(BACKEND_ROOT / "output" / f"{req.job_id}_report.json")

        try:
            import json

            with Path(report_path).open("r", encoding="utf-8") as f:
                report_cards: list[dict[str, Any]] = json.load(f)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Failed to read job report for chat context.",
                    "error": str(exc),
                },
            ) from exc

        top = report_cards[:5]
        lines: list[str] = []
        for item in top:
            team = str(item.get("team", "?"))
            flaw = str(item.get("flaw", "?"))
            severity = str(item.get("severity", ""))
            evidence = str(item.get("evidence", "")).strip()
            lines.append(f"- {team}: {flaw} ({severity}) — {evidence}")

        prompt_context = "Job insights:\n" + "\n".join(lines)

    full_prompt = f"""You are an elite football coach and tactician.

{prompt_context}

User question:
{req.message}

Output requirements:
1. Provide exactly 3 numbered tactical steps.
2. Keep it under 150 words.
3. Focus on the user question and stay consistent with the job insights when provided.
"""

    try:
        reply = await get_tactical_advice(full_prompt, selected_llm_engine)
    except EngineRoutingError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.to_detail()) from exc
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Chat completion failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ChatResponse(reply=reply)


@app.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str) -> None:
    """
    Stream job progress updates over WebSockets.

    Sends JSON messages shaped like:
      { job_id, status: pending|processing|done|error, current_step, result_path?, error? }
    """
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
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _card_needs_local_llm_refresh(card: dict[str, Any]) -> bool:
    """True when the card has a prompt but no successful coaching text yet."""
    prompt = card.get("llm_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    tip = card.get("tactical_instruction")
    if isinstance(tip, str) and tip.strip():
        return False
    return True


def _normalize_instruction_steps(text: str | None) -> list[str]:
    """Convert model output into a clean list of at most 3 tactical points."""
    if not isinstance(text, str):
        return []
    cleaned = text.strip()
    if not cleaned:
        return []

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    steps: list[str] = []
    bullet_prefix = re.compile(r"^\s*(?:\d+[.)]\s*|[-*]\s+)")
    for line in lines:
        line_step = bullet_prefix.sub("", line).strip()
        if line_step:
            steps.append(line_step)

    if len(steps) <= 1:
        chunks = re.split(r"(?<=[.!?])\s+", cleaned)
        steps = [c.strip() for c in chunks if c.strip()]

    deduped: list[str] = []
    for step in steps:
        normalized = " ".join(step.split())
        if normalized and normalized not in deduped:
            deduped.append(normalized)
        if len(deduped) == 3:
            break
    return deduped


def _format_numbered_steps(steps: list[str], fallback_text: str | None) -> str | None:
    """Return user-facing numbered text so older clients still render readable points."""
    if steps:
        return "\n".join([f"{idx + 1}. {step}" for idx, step in enumerate(steps)])
    if isinstance(fallback_text, str):
        value = fallback_text.strip()
        return value or None
    return None


async def _refresh_job_report_cards_with_local_llm(
    report_cards: list[dict[str, Any]],
    *,
    llm_concurrency: int,
) -> list[dict[str, Any]]:
    """Re-run LLM for job report rows using Ollama when the pipeline skipped cloud keys."""
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

    return list(await asyncio.gather(*[_one(c) for c in report_cards]))


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

        if not skip_llm and llm_engine == "local":
            report_cards = await _refresh_job_report_cards_with_local_llm(
                report_cards,
                llm_concurrency=llm_concurrency,
            )

        # Determine whether LLM produced text.
        llm_ok = any(bool(c.get("tactical_instruction")) for c in report_cards)
        pipeline_llm = (
            f"ok ({llm_engine})" if llm_ok else "skipped"
        )
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
                    tactical_instruction=_format_numbered_steps(
                        _normalize_instruction_steps(card.get("tactical_instruction")),
                        card.get("tactical_instruction"),
                    ),
                    tactical_instruction_steps=_normalize_instruction_steps(
                        card.get("tactical_instruction")
                    ),
                    llm_error=card.get("llm_error"),
                )
            )

        return CoachAdviceResponse(
            generated_at=datetime.now(timezone.utc).isoformat(),
            pipeline=pipeline,
            advice_items=advice_items,
        )

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
        instruction_steps = _normalize_instruction_steps(instruction)
        advice_items.append(
            CoachingAdviceItem(
                frame_idx=rec.frame_idx,
                team=rec.team,
                flaw=rec.flaw,
                severity=rec.severity,
                evidence=rec.evidence,
                matched_philosophy_author=rec.matched_philosophy_author,
                fc25_player_roles=rec.fc_role_recommendations,
                tactical_instruction=_format_numbered_steps(
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
