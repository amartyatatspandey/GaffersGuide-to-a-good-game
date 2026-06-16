from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Reuse existing pipeline components
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

# Import from the existing cloud pipeline
import scripts.run_e2e_cloud as cloud_pipeline
from scripts.e2e_shared import (
    MIN_BALL_CONFIDENCE,
    FPS,
    COUNTER_ATTACK_WINDOW_FRAMES,
    PRESS_SUCCESS_WINDOW_FRAMES,
    build_metrics_timeline,
    compute_ball_visibility_ratio,
    apply_ball_metrics_gate,
    evaluate_chunk_insights,
    synthesize,
    run_llm,
    _resolve_video_path,
    ensure_core_pipeline_directories,
    collect_local_cv_pipeline_gaps,
    format_pipeline_prerequisite_errors,
    _print_step,
    _write_tracking_artifact,
    _print_data_guard_reliability,
    _final_cards_llm_skipped_low_reliability,
)
from scripts.global_refiner import GlobalRefiner
from scripts.track_teams import TacticalRadar
from scripts.legacy.run_e2e_legacy import TacticalFrame, TacticalPlayer, CVTelemetry

# Import the new ZSL module
from scripts.zsl_classifier import ZSLTacticalClassifier

LOGGER = logging.getLogger(__name__)

def run_e2e_with_zsl(
    video: str | Path,
    *,
    output_prefix: str = "test_mp4",
    progress_callback: Callable[[str], None] | None = None,
    batch_size: int = cloud_pipeline.DEFAULT_BATCH_SIZE,
    flow_max_width: int = cloud_pipeline.DEFAULT_FLOW_MAX_WIDTH,
    device: str | None = None,
    llm_engine: Literal["local", "cloud"] = "cloud",
    enable_zsl: bool = False,
) -> Path:
    """
    Parallel version of run_e2e_cloud with ZSL branch support.
    """
    if progress_callback is not None:
        progress_callback("Pending")

    video_path = video if isinstance(video, Path) else _resolve_video_path(video)

    ensure_core_pipeline_directories()
    prereq_gaps = collect_local_cv_pipeline_gaps(video_path=video_path)
    if prereq_gaps:
        raise FileNotFoundError(format_pipeline_prerequisite_errors(prereq_gaps))

    # Setup paths
    out_dir = BACKEND_ROOT / "output"
    tracking_overlay_path = out_dir / f"{output_prefix}_tracking_overlay.mp4"
    tracking_data_path = out_dir / f"{output_prefix}_tracking_data.json"
    metrics_output_path = out_dir / f"{output_prefix}_tactical_metrics.json"
    report_output_path = out_dir / f"{output_prefix}_report.json"

    # Ensure no stale artifacts are interpreted as new output.
    for p in [tracking_overlay_path, tracking_data_path, metrics_output_path, report_output_path]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    if progress_callback is not None:
        progress_callback("Tracking Players")
    
    # Run the core CV tracking
    raw_frames, telemetry, tracking_frames = cloud_pipeline.run_cv_tracking_batched(
        video_path,
        batch_size=batch_size,
        flow_max_width=flow_max_width,
        device=device,
        progress_callback=progress_callback,
    )
    
    _print_step(f"CV Tracking Complete: Processed {len(raw_frames)} frames.")
    _write_tracking_artifact(
        tracking_data_path,
        video_path=video_path,
        overlay_path=tracking_overlay_path,
        telemetry=telemetry,
        frames=tracking_frames,
    )

    # Spatial Math & Refinement
    if progress_callback is not None:
        progress_callback("Spatial Math")
    
    refiner = GlobalRefiner()
    refined_frames = refiner.refine(
        raw_frames,
        frame_factory=lambda frame_idx, players: TacticalFrame(
            frame_idx=frame_idx,
            players=players,
            ball_xy=None,
            possession_team_id=None,
        ),
        player_factory=lambda pid, team, radar_pt: TacticalPlayer(
            id=pid, team=team, radar_pt=radar_pt
        ),
    )
    
    # Restore ball and possession data to refined frames
    raw_ball_by_frame = {f.frame_idx: (f.ball_xy, f.possession_team_id) for f in raw_frames}
    for frame in refined_frames:
        ball_xy, possession_team_id = raw_ball_by_frame.get(frame.frame_idx, (None, None))
        frame.ball_xy = ball_xy
        frame.possession_team_id = possession_team_id

    metrics = build_metrics_timeline(refined_frames)
    visibility_ratio = compute_ball_visibility_ratio(refined_frames)
    metrics, ball_data_quality = apply_ball_metrics_gate(
        metrics,
        refined_frames,
        visibility_ratio=visibility_ratio,
        min_ball_confidence=MIN_BALL_CONFIDENCE,
        fps=FPS,
        counter_attack_window_frames=COUNTER_ATTACK_WINDOW_FRAMES,
        press_success_window_frames=PRESS_SUCCESS_WINDOW_FRAMES,
    )

    with metrics_output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --- Tactical Branching ---
    if progress_callback is not None:
        progress_callback("Tactical Engine")
    
    if enable_zsl:
        _print_step("Running Zero-Shot Learning tactical branch exclusively...")
        triggers = []
        try:
            from scripts.zsl_classifier import ZSLTacticalClassifier
            zsl_classifier = ZSLTacticalClassifier()
            zsl_insights_0 = zsl_classifier.analyze_chunk(refined_frames, "team_0")
            zsl_insights_1 = zsl_classifier.analyze_chunk(refined_frames, "team_1")
            
            zsl_count = 0
            for insight in zsl_insights_0 + zsl_insights_1:
                triggers.append(insight.model_dump())
                zsl_count += 1
            
            _print_step(f"ZSL branch complete. Extracted {zsl_count} neural tactical insights.")
        except Exception as e:
            LOGGER.warning(f"ZSL branch failed, falling back to Rule Engine: {e}")
            triggers = evaluate_chunk_insights(metrics)
    else:
        _print_step("Running Standard Rule Engine...")
        triggers = evaluate_chunk_insights(metrics)

    # Synthesize & LLM
    reliability_pct, guard_status = _print_data_guard_reliability(len(metrics), len(raw_frames))
    library_path = BACKEND_ROOT / "data" / ("zsl_tactics.json" if enable_zsl else "tactical_library.json")
    
    if progress_callback is not None:
        progress_callback("Synthesizing Advice")
    
    prompt_records = synthesize(
        triggers,
        library_path=library_path,
        ball_data_quality=ball_data_quality,
    )

    if guard_status == "abort":
        final_cards = _final_cards_llm_skipped_low_reliability(
            prompt_records, reliability_pct, len(metrics), len(raw_frames)
        )
    else:
        # NOTE: This function runs inside asyncio.to_thread(), so we cannot call
        # asyncio.run() (it would raise RuntimeError: event loop already running).
        # Instead, create a dedicated new loop in a separate thread for each LLM call.
        import concurrent.futures

        def _run_async_in_new_loop(coro):
            """Execute an async coroutine from a synchronous context safely."""
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(asyncio.run, coro).result()

        if llm_engine == "local":
            from scripts.llm_router import ensure_ollama_available
            _run_async_in_new_loop(ensure_ollama_available())
            from scripts.e2e_llm_local import run_llm_local
            final_cards = _run_async_in_new_loop(run_llm_local(prompt_records))
        else:
            final_cards = _run_async_in_new_loop(run_llm(prompt_records))

    # Safety net: if the report is still empty, synthesize a baseline advisory from match metrics.
    if not final_cards and metrics:
        LOGGER.warning("Pipeline produced 0 advice cards; injecting baseline match summary.")
        rule_engine = __import__("scripts.tactical_rule_engine", fromlist=["RuleEngine"]).RuleEngine
        re = rule_engine()
        scores = re.calculate_tactical_scores(metrics)
        final_cards = [{
            "frame_idx": 0,
            "team": "global",
            "flaw": "Match Summary",
            "severity": "Info",
            "frequency_pct": 100.0,
            "confidence_pct": 100.0,
            "confidence_reason": "Baseline injection from direct telemetry scores.",
            "evidence": (
                f"Tactical Power: Red {scores['team_0'].get('tactical_power', 0):.1f} vs Blue {scores['team_1'].get('tactical_power', 0):.1f}. "
                f"Win Probability: Red {scores['team_0'].get('win_prob', 50)}% | Blue {scores['team_1'].get('win_prob', 50)}%. "
                f"Compactness: Red {scores['team_0'].get('compactness', 0):.0f} / Blue {scores['team_1'].get('compactness', 0):.0f}. "
                f"Transition Speed: Red {scores['team_0'].get('transition_speed', 0):.0f} / Blue {scores['team_1'].get('transition_speed', 0):.0f}."
            ),
            "summary_data": scores,
            "matched_philosophy_author": "GAFFER Match Engine",
            "matched_quote_excerpt": "Structured analysis based on spatial telemetry.",
            "fc_role_recommendations": [],
            "tactical_instruction": (
                "1. Analyze the win probability distribution to identify which team needs tactical adjustments. "
                "2. Focus on the team with lower compactness scores — they need to tighten their defensive shape. "
                "3. The team with higher pressing intensity should be encouraged to sustain pressure in the final third."
            ),
            "llm_error": None,
        }]

    with report_output_path.open("w", encoding="utf-8") as f:
        json.dump(final_cards, f, indent=2, ensure_ascii=False)

    if progress_callback is not None:
        progress_callback("Completed")
    
    return report_output_path

def main():
    parser = argparse.ArgumentParser(description="GAFFER E2E Pipeline with optional ZSL support.")
    parser.add_argument("video", type=str, help="Video filename or path.")
    parser.add_argument("--output-prefix", type=str, default="test_mp4")
    parser.add_argument("--batch-size", type=int, default=cloud_pipeline.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--llm-engine", type=str, choices=("local", "cloud"), default="cloud")
    parser.add_argument("--enable-zsl", action="store_true", help="Enable the Zero-Shot Learning tactical branch.")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    run_e2e_with_zsl(
        args.video,
        output_prefix=args.output_prefix,
        batch_size=args.batch_size,
        device=args.device,
        llm_engine=args.llm_engine,
        enable_zsl=args.enable_zsl
    )

if __name__ == "__main__":
    main()
