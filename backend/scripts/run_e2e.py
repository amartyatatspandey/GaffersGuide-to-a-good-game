from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from openai import AsyncOpenAI
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT_LOCAL = SCRIPT_DIR.parent

# Ensure backend root is importable so `from models import ...` works.
if str(BACKEND_ROOT_LOCAL) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT_LOCAL))

from generate_analytics import TacticalAnalyzer
from rag_coach import (
    GeneratedPromptRecord,
    TacticalLibrary,
    load_json,
    process_insights,
)
from models import ChunkTacticalInsight
from tactical_rule_engine import evaluate_timeline as evaluate_chunk_insights
from track_teams import (
    BACKEND_ROOT,
    CLASS_BALL,
    CLASS_PLAYER,
    HybridIDHealer,
    MODEL_PATH,
    TacticalRadar,
    TeamClassifier,
)


@dataclass(slots=True)
class TacticalPlayer:
    id: int | None
    team: str
    radar_pt: list[float] | None


@dataclass(slots=True)
class TacticalFrame:
    frame_idx: int
    players: list[TacticalPlayer]


def _print_step(message: str) -> None:
    print(f"[✓] {message}")


def _resolve_video_path(video_name: str) -> Path:
    candidate = Path(video_name)
    if candidate.is_file():
        return candidate
    backend_data_path = BACKEND_ROOT / "data" / video_name
    if backend_data_path.is_file():
        return backend_data_path
    if video_name == "test.mp4":
        alias = BACKEND_ROOT / "data" / "match_test.mp4"
        if alias.is_file():
            print("[i] test.mp4 not found, using backend/data/match_test.mp4")
            return alias
    raise FileNotFoundError(
        f"Video not found: {video_name}. Tried local path and {backend_data_path}"
    )


def _prediction_to_team(prediction: str) -> str | None:
    if prediction == "team_0":
        return "team_0"
    if prediction == "team_1":
        return "team_1"
    return None


def run_cv_tracking(video_path: Path) -> list[TacticalFrame]:
    """
    Run CV tracking in-memory and return TacticalFrame timeline.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Tracking model not found: {MODEL_PATH}")

    model: YOLO = YOLO(str(MODEL_PATH))
    tracker = sv.ByteTrack()
    classifier = TeamClassifier()
    healer = HybridIDHealer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    radar = TacticalRadar(video_res=(width, height))

    frames_out: list[TacticalFrame] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results: list[Any] = model(frame, conf=0.3, verbose=False)
            if not results:
                frames_out.append(TacticalFrame(frame_idx=frame_idx, players=[]))
                frame_idx += 1
                continue

            detections = sv.Detections.from_ultralytics(results[0])
            detections = tracker.update_with_detections(detections)
            radar.update_camera_angle(frame_idx)

            radar_pts: list[tuple[int, int] | None] = [
                radar.map_to_2d(detections.xyxy[i]) for i in range(len(detections))
            ]
            tracker_ids = healer.process_and_heal(detections, frame, radar_pts, frame_idx)
            if tracker_ids is None:
                tracker_ids = getattr(detections, "tracker_id", None)

            frame_data: list[dict[str, Any]] = []
            for i in range(len(detections)):
                tid: int | None = None
                if tracker_ids is not None and i < len(tracker_ids):
                    raw_id = tracker_ids[i]
                    tid = int(raw_id) if raw_id is not None else None
                frame_data.append(
                    {
                        "id": tid,
                        "bbox": detections.xyxy[i],
                        "cid": int(detections.class_id[i]),
                        "radar_pt": radar_pts[i],
                    }
                )

            role_mapping = classifier.predict_frame(frame, frame_data, frame_idx)
            tactical_players: list[TacticalPlayer] = []
            for row in frame_data:
                if row["cid"] in (CLASS_BALL,):
                    continue
                if row["cid"] != CLASS_PLAYER:
                    continue
                prediction = role_mapping.get(row["id"], "unknown")
                team = _prediction_to_team(prediction)
                if team is None:
                    continue
                pt = row["radar_pt"]
                pt_out: list[float] | None = None
                if pt is not None:
                    pt_out = [float(pt[0]), float(pt[1])]
                tactical_players.append(
                    TacticalPlayer(id=row["id"], team=team, radar_pt=pt_out)
                )

            frames_out.append(TacticalFrame(frame_idx=frame_idx, players=tactical_players))
            frame_idx += 1
    finally:
        cap.release()

    return frames_out


def build_metrics_timeline(raw_frames: list[TacticalFrame]) -> list[dict[str, Any]]:
    """
    Compute tactical metrics from in-memory TacticalFrame data.
    """
    analyzer = TacticalAnalyzer()
    timeline: list[dict[str, Any]] = []

    for frame in raw_frames:
        t0_pts: list[list[float]] = []
        t1_pts: list[list[float]] = []
        t0_speeds: list[float] = []
        t1_speeds: list[float] = []

        for p in frame.players:
            if p.radar_pt is None or p.id is None:
                continue
            speed = analyzer.calc_speed(p.id, frame.frame_idx, p.radar_pt)
            if p.team == "team_0":
                t0_pts.append(p.radar_pt)
                t0_speeds.append(speed)
            elif p.team == "team_1":
                t1_pts.append(p.radar_pt)
                t1_speeds.append(speed)

        metrics_0 = analyzer.analyze_team_spatial(t0_pts)
        metrics_1 = analyzer.analyze_team_spatial(t1_pts)
        if not metrics_0 or not metrics_1:
            continue

        t0_pct, t1_pct = analyzer.calculate_pitch_control(t0_pts, t1_pts)
        t0_def_mid, t0_mid_att = analyzer.calculate_line_gaps(t0_pts)
        t1_def_mid, t1_mid_att = analyzer.calculate_line_gaps(t1_pts)

        metrics_0.update(
            {
                "pitch_control_pct": t0_pct,
                "pressure_index_m": analyzer.calculate_pressure_index(t0_pts, t1_pts),
                "line_gap_def_mid_m": t0_def_mid,
                "line_gap_mid_att_m": t0_mid_att,
                "avg_speed_kmh": float(np.mean(t0_speeds)) if t0_speeds else 0.0,
                "max_speed_kmh": float(np.max(t0_speeds)) if t0_speeds else 0.0,
            }
        )
        metrics_1.update(
            {
                "pitch_control_pct": t1_pct,
                "pressure_index_m": analyzer.calculate_pressure_index(t1_pts, t0_pts),
                "line_gap_def_mid_m": t1_def_mid,
                "line_gap_mid_att_m": t1_mid_att,
                "avg_speed_kmh": float(np.mean(t1_speeds)) if t1_speeds else 0.0,
                "max_speed_kmh": float(np.max(t1_speeds)) if t1_speeds else 0.0,
            }
        )
        timeline.append(
            {"frame_idx": frame.frame_idx, "team_0": metrics_0, "team_1": metrics_1}
        )

    return timeline


def synthesize(
    triggers: list[dict[str, Any]], library_path: Path
) -> list[GeneratedPromptRecord]:
    """
    Build RAG prompt payloads from trigger timeline and tactical library.
    """
    raw_library = load_json(library_path)
    library = TacticalLibrary.model_validate(raw_library)
    insights = [ChunkTacticalInsight.model_validate(row) for row in triggers]
    return process_insights(insights=insights, library=library)


def _resolve_llm_credentials() -> tuple[str | None, str, str | None]:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
    return api_key, model, base_url


async def _complete_prompt(
    client: AsyncOpenAI, model: str, prompt: str
) -> tuple[str | None, str | None]:
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
            max_tokens=600,
        )
        content = response.choices[0].message.content
        return (content.strip() if content else None, None)
    except Exception as exc:  # noqa: BLE001
        return (None, str(exc))


async def run_llm(records: list[GeneratedPromptRecord]) -> list[dict[str, Any]]:
    """
    Run Cloud LLM completions for each generated prompt record.
    """
    api_key, model, base_url = _resolve_llm_credentials()
    if not api_key:
        return [
            {
                **r.model_dump(),
                "tactical_instruction": None,
                "llm_error": "Missing LLM_API_KEY/OPENAI_API_KEY; skipped cloud completion.",
            }
            for r in records
        ]

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = AsyncOpenAI(**kwargs)

    tasks = [_complete_prompt(client, model, r.llm_prompt) for r in records]
    results = await asyncio.gather(*tasks) if tasks else []

    out: list[dict[str, Any]] = []
    for record, (instruction, llm_error) in zip(records, results, strict=True):
        payload = record.model_dump()
        payload["tactical_instruction"] = instruction
        payload["llm_error"] = llm_error
        out.append(payload)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run complete CV→Math→Rules→RAG→LLM E2E pipeline."
    )
    parser.add_argument(
        "video",
        type=str,
        help="Video filename or path (example: test.mp4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = _resolve_video_path(args.video)

    raw_frames = run_cv_tracking(video_path)
    _print_step(f"CV Tracking Complete: Processed {len(raw_frames)} frames.")

    metrics = build_metrics_timeline(raw_frames)
    triggers = evaluate_chunk_insights(metrics)
    flaw_count = len(triggers)
    _print_step(
        f"Analytics Complete: Built {len(metrics)} metric frames and found {flaw_count} chunk-level tactical flaws."
    )

    library_path = BACKEND_ROOT / "data" / "tactical_library.json"
    prompt_records = synthesize(triggers, library_path=library_path)
    _print_step(f"RAG Complete: Generated {len(prompt_records)} coaching prompts.")

    final_cards = asyncio.run(run_llm(prompt_records))

    output_path = BACKEND_ROOT / "output" / "test_mp4_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_cards, f, indent=2, ensure_ascii=False)

    _print_step(f"LLM Complete: Report saved to {output_path}.")


if __name__ == "__main__":
    main()

