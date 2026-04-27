from __future__ import annotations

from gaffers_guide.pipeline.e2e_shared_impl import BALL_INTERPOLATION_MAX_GAP
from gaffers_guide.pipeline.e2e_shared_impl import COUNTER_ATTACK_WINDOW_FRAMES
from gaffers_guide.pipeline.e2e_shared_impl import FPS
from gaffers_guide.pipeline.e2e_shared_impl import HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD
from gaffers_guide.pipeline.e2e_shared_impl import MIN_BALL_CONFIDENCE
from gaffers_guide.pipeline.e2e_shared_impl import PRESS_SUCCESS_WINDOW_FRAMES
from gaffers_guide.pipeline.e2e_shared_impl import CVTelemetry
from gaffers_guide.pipeline.e2e_shared_impl import TacticalFrame
from gaffers_guide.pipeline.e2e_shared_impl import TacticalPlayer
from gaffers_guide.pipeline.e2e_shared_impl import TrackingFrameArtifact
from gaffers_guide.pipeline.e2e_shared_impl import _final_cards_llm_skipped_low_reliability
from gaffers_guide.pipeline.e2e_shared_impl import _fallback_project_from_camera_shift
from gaffers_guide.pipeline.e2e_shared_impl import _homography_confidence
from gaffers_guide.pipeline.e2e_shared_impl import _prediction_to_team
from gaffers_guide.pipeline.e2e_shared_impl import _print_data_guard_reliability
from gaffers_guide.pipeline.e2e_shared_impl import _print_step
from gaffers_guide.pipeline.e2e_shared_impl import _resolve_primary_ball_class_ids
from gaffers_guide.pipeline.e2e_shared_impl import _resolve_video_path
from gaffers_guide.pipeline.e2e_shared_impl import _write_tracking_artifact
from gaffers_guide.pipeline.e2e_shared_impl import apply_ball_metrics_gate
from gaffers_guide.pipeline.e2e_shared_impl import build_metrics_timeline
from gaffers_guide.pipeline.e2e_shared_impl import compute_ball_visibility_ratio
from gaffers_guide.pipeline.e2e_shared_impl import evaluate_chunk_insights
from gaffers_guide.pipeline.e2e_shared_impl import run_llm
from gaffers_guide.pipeline.e2e_shared_impl import synthesize

__all__ = [
    "BALL_INTERPOLATION_MAX_GAP",
    "COUNTER_ATTACK_WINDOW_FRAMES",
    "FPS",
    "HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD",
    "MIN_BALL_CONFIDENCE",
    "PRESS_SUCCESS_WINDOW_FRAMES",
    "CVTelemetry",
    "TacticalFrame",
    "TacticalPlayer",
    "TrackingFrameArtifact",
    "_final_cards_llm_skipped_low_reliability",
    "_fallback_project_from_camera_shift",
    "_homography_confidence",
    "_prediction_to_team",
    "_print_data_guard_reliability",
    "_print_step",
    "_resolve_primary_ball_class_ids",
    "_resolve_video_path",
    "_write_tracking_artifact",
    "apply_ball_metrics_gate",
    "build_metrics_timeline",
    "compute_ball_visibility_ratio",
    "evaluate_chunk_insights",
    "run_llm",
    "synthesize",
]