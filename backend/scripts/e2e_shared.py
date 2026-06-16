from __future__ import annotations

from services.pipeline_paths import (
    collect_local_cv_pipeline_gaps,
    ensure_core_pipeline_directories,
    format_pipeline_prerequisite_errors,
)

from scripts.legacy.run_e2e_legacy import (
    BALL_INTERPOLATION_MAX_GAP,
    COUNTER_ATTACK_WINDOW_FRAMES,
    FPS,
    HOMOGRAPHY_CONFIDENCE_FALLBACK_THRESHOLD,
    MIN_BALL_CONFIDENCE,
    PRESS_SUCCESS_WINDOW_FRAMES,
    CVTelemetry,
    TacticalFrame,
    TacticalPlayer,
    TrackingFrameArtifact,
    _fallback_project_from_camera_shift,
    _final_cards_llm_skipped_low_reliability,
    _homography_confidence,
    _prediction_to_team,
    _print_data_guard_reliability,
    _print_step,
    _resolve_primary_ball_class_ids,
    _resolve_video_path,
    _write_tracking_artifact,
    apply_ball_metrics_gate,
    build_metrics_timeline,
    compute_ball_visibility_ratio,
    evaluate_chunk_insights,
    run_llm,
    synthesize,
)

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
    "_fallback_project_from_camera_shift",
    "_final_cards_llm_skipped_low_reliability",
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
    "ensure_core_pipeline_directories",
    "collect_local_cv_pipeline_gaps",
    "format_pipeline_prerequisite_errors",
    "interpolate_tracking_data",
]


def interpolate_tracking_data(
    frames_out: list[TacticalFrame],
    frame_artifacts: list[TrackingFrameArtifact],
    frame_step: int,
) -> None:
    """
    Linearly interpolates player and ball positions in place for skipped frames.

    Args:
        frames_out: List of TacticalFrame objects containing sparse keyframes.
        frame_artifacts: List of TrackingFrameArtifact objects containing sparse keyframes.
        frame_step: The sampling step size used (e.g. 3).
    """
    if frame_step <= 1:
        return

    n = len(frames_out)
    if n <= 1:
        return

    for i in range(0, n, frame_step):
        next_i = i + frame_step
        if next_i >= n:
            # For the last incomplete gap, copy the last keyframe's data
            for j in range(i + 1, n):
                frames_out[j].players = [
                    TacticalPlayer(id=p.id, team=p.team, radar_pt=p.radar_pt)
                    for p in frames_out[i].players
                ]
                frames_out[j].ball_xy = frames_out[i].ball_xy
                frames_out[j].possession_team_id = frames_out[i].possession_team_id
                
                # Copy artifacts
                frame_artifacts[j].players = [dict(p) for p in frame_artifacts[i].players]
                frame_artifacts[j].ball_xy = frame_artifacts[i].ball_xy
                frame_artifacts[j].ball_canvas = frame_artifacts[i].ball_canvas
                frame_artifacts[j].homography_confidence = frame_artifacts[i].homography_confidence
                frame_artifacts[j].used_optical_flow_fallback = frame_artifacts[i].used_optical_flow_fallback
                frame_artifacts[j].camera_shift_xy = frame_artifacts[i].camera_shift_xy
                frame_artifacts[j].possession_team_id = frame_artifacts[i].possession_team_id
            break

        # Map players by ID for fast lookup
        players_i = {p.id: p for p in frames_out[i].players if p.id is not None}
        players_next = {p.id: p for p in frames_out[next_i].players if p.id is not None}
        
        art_players_i = {p["id"]: p for p in frame_artifacts[i].players if p["id"] is not None}
        art_players_next = {p["id"]: p for p in frame_artifacts[next_i].players if p["id"] is not None}

        # Interpolate each frame in the gap
        for j in range(i + 1, next_i):
            w = (j - i) / float(frame_step)  # interpolation weight: 0 < w < 1
            
            # 1. Interpolate TacticalFrame players
            interp_players = []
            for pid, p_start in players_i.items():
                if pid in players_next:
                    p_end = players_next[pid]
                    if p_start.radar_pt is not None and p_end.radar_pt is not None:
                        rx = p_start.radar_pt[0] * (1.0 - w) + p_end.radar_pt[0] * w
                        ry = p_start.radar_pt[1] * (1.0 - w) + p_end.radar_pt[1] * w
                        radar_pt = [rx, ry]
                    else:
                        radar_pt = p_start.radar_pt if w < 0.5 else p_end.radar_pt
                    interp_players.append(
                        TacticalPlayer(id=pid, team=p_start.team, radar_pt=radar_pt)
                    )
                else:
                    # Player disappeared; keep them for the first half of the gap
                    if w < 0.5:
                        interp_players.append(
                            TacticalPlayer(id=pid, team=p_start.team, radar_pt=p_start.radar_pt)
                        )
            
            # Include players that appeared in next_i but weren't in i
            for pid, p_end in players_next.items():
                if pid not in players_i:
                    if w >= 0.5:
                        interp_players.append(
                            TacticalPlayer(id=pid, team=p_end.team, radar_pt=p_end.radar_pt)
                        )
                        
            frames_out[j].players = interp_players

            # 2. Interpolate TacticalFrame ball_xy
            if frames_out[i].ball_xy is not None and frames_out[next_i].ball_xy is not None:
                bx = frames_out[i].ball_xy[0] * (1.0 - w) + frames_out[next_i].ball_xy[0] * w
                by = frames_out[i].ball_xy[1] * (1.0 - w) + frames_out[next_i].ball_xy[1] * w
                frames_out[j].ball_xy = [bx, by]
            else:
                frames_out[j].ball_xy = frames_out[i].ball_xy if w < 0.5 else frames_out[next_i].ball_xy

            # Possession is copied from closest keyframe
            frames_out[j].possession_team_id = (
                frames_out[i].possession_team_id if w < 0.5 else frames_out[next_i].possession_team_id
            )

            # 3. Interpolate TrackingFrameArtifact players list
            interp_art_players = []
            for pid, ap_start in art_players_i.items():
                if pid in art_players_next:
                    ap_end = art_players_next[pid]
                    # Interpolate x_pitch, y_pitch
                    x_p = None
                    if ap_start.get("x_pitch") is not None and ap_end.get("x_pitch") is not None:
                        x_p = ap_start["x_pitch"] * (1.0 - w) + ap_end["x_pitch"] * w
                    y_p = None
                    if ap_start.get("y_pitch") is not None and ap_end.get("y_pitch") is not None:
                        y_p = ap_start["y_pitch"] * (1.0 - w) + ap_end["y_pitch"] * w
                        
                    # Interpolate x_canvas, y_canvas
                    x_c = ap_start["x_canvas"] * (1.0 - w) + ap_end["x_canvas"] * w
                    y_c = ap_start["y_canvas"] * (1.0 - w) + ap_end["y_canvas"] * w
                    
                    # Interpolate bbox
                    bbox = []
                    if ap_start.get("bbox") is not None and ap_end.get("bbox") is not None:
                        bbox = [
                            ap_start["bbox"][k] * (1.0 - w) + ap_end["bbox"][k] * w
                            for k in range(4)
                        ]
                    else:
                        bbox = ap_start.get("bbox") if w < 0.5 else ap_end.get("bbox")
                        
                    # Interpolate speed
                    speed = 0.0
                    if ap_start.get("speed_kmh") is not None and ap_end.get("speed_kmh") is not None:
                        speed = ap_start["speed_kmh"] * (1.0 - w) + ap_end["speed_kmh"] * w
                        
                    interp_art_players.append({
                        "id": pid,
                        "team_id": ap_start["team_id"],
                        "x_pitch": x_p,
                        "y_pitch": y_p,
                        "x_canvas": x_c,
                        "y_canvas": y_c,
                        "bbox": bbox,
                        "speed_kmh": round(float(speed), 2)
                    })
                else:
                    if w < 0.5:
                        interp_art_players.append(dict(ap_start))
                        
            for pid, ap_end in art_players_next.items():
                if pid not in art_players_i:
                    if w >= 0.5:
                        interp_art_players.append(dict(ap_end))
                        
            frame_artifacts[j].players = interp_art_players

            # 4. Interpolate TrackingFrameArtifact ball_xy
            if frame_artifacts[i].ball_xy is not None and frame_artifacts[next_i].ball_xy is not None:
                bx = frame_artifacts[i].ball_xy[0] * (1.0 - w) + frame_artifacts[next_i].ball_xy[0] * w
                by = frame_artifacts[i].ball_xy[1] * (1.0 - w) + frame_artifacts[next_i].ball_xy[1] * w
                frame_artifacts[j].ball_xy = [bx, by]
            else:
                frame_artifacts[j].ball_xy = frame_artifacts[i].ball_xy if w < 0.5 else frame_artifacts[next_i].ball_xy

            # 5. Interpolate TrackingFrameArtifact ball_canvas
            if frame_artifacts[i].ball_canvas is not None and frame_artifacts[next_i].ball_canvas is not None:
                cx = frame_artifacts[i].ball_canvas[0] * (1.0 - w) + frame_artifacts[next_i].ball_canvas[0] * w
                cy = frame_artifacts[i].ball_canvas[1] * (1.0 - w) + frame_artifacts[next_i].ball_canvas[1] * w
                frame_artifacts[j].ball_canvas = [cx, cy]
            else:
                frame_artifacts[j].ball_canvas = frame_artifacts[i].ball_canvas if w < 0.5 else frame_artifacts[next_i].ball_canvas

            # 6. Interpolate TrackingFrameArtifact metadata
            frame_artifacts[j].homography_confidence = (
                frame_artifacts[i].homography_confidence * (1.0 - w) + frame_artifacts[next_i].homography_confidence * w
            )
            frame_artifacts[j].used_optical_flow_fallback = (
                frame_artifacts[i].used_optical_flow_fallback if w < 0.5 else frame_artifacts[next_i].used_optical_flow_fallback
            )
            cx_shift = frame_artifacts[i].camera_shift_xy[0] * (1.0 - w) + frame_artifacts[next_i].camera_shift_xy[0] * w
            cy_shift = frame_artifacts[i].camera_shift_xy[1] * (1.0 - w) + frame_artifacts[next_i].camera_shift_xy[1] * w
            frame_artifacts[j].camera_shift_xy = (cx_shift, cy_shift)
            frame_artifacts[j].possession_team_id = (
                frame_artifacts[i].possession_team_id if w < 0.5 else frame_artifacts[next_i].possession_team_id
            )

