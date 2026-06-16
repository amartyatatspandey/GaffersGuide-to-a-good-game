from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_PATH = BACKEND_ROOT / "output" / "tactical_metrics.json"
DEFAULT_OUTPUT_PATH = BACKEND_ROOT / "output" / "tactical_triggers.json"


class RuleEngine:
    def __init__(self):
        # Professional Tactical Thresholds
        self.MAX_LINE_GAP = 20.0  # meters
        self.HIGH_LINE_X = 20.0  # meters past center
        self.LOW_BLOCK_X = -25.0  # meters behind center
        self.POOR_PRESS_DIST = 8.0  # meters (increased from 6.0 for sensitivity)
        self.MAX_TEAM_LENGTH = 65.0  # meters (increased from 55.0 to prevent false over-stretch flags)
        self.MIN_PITCH_CONTROL = 40.0  # percentage

    def calculate_tactical_scores(self, metrics_timeline: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Elite-grade tactical KPI scorecard with professional coaching standards.
        Includes: compactness (H/V/midfield), press resistance, width utilization,
        line staggering, overload score, transition speed, and confidence percentages.
        """
        def safe_mean(frames: list[dict], key: str, default: float) -> float:
            vals = [float(f.get(key, default) or default) for f in frames]
            return statistics.mean(vals) if vals else default

        def aggregate(tid: str) -> dict[str, float]:
            frames = [f.get(tid, {}) for f in metrics_timeline if tid in f]
            if not frames:
                return {}

            team_len = safe_mean(frames, "team_length_m", 50.0)
            team_wid = safe_mean(frames, "team_width_m", 40.0)
            pitch_ctrl = safe_mean(frames, "pitch_control_pct", 50.0)
            press_idx = safe_mean(frames, "pressure_index_m", 10.0)
            line_gap = safe_mean(frames, "line_gap_def_mid_m", 15.0)
            mid_gap = safe_mean(frames, "line_gap_mid_att_m", 15.0)
            avg_speed = safe_mean(frames, "avg_speed_mps", 4.0)
            area = safe_mean(frames, "area_sq_meters", 800.0)
            deepest_x = safe_mean(frames, "deepest_x", 0.0)

            # --- Compactness KPIs ---
            # Vertical: how tight front-to-back (lower team length = better)
            compactness_v = max(0.0, min(100.0, 100.0 - (team_len - 20.0) * 1.5))
            # Horizontal: how contained side-to-side (40m width is optimal)
            compactness_h = max(0.0, min(100.0, 100.0 - abs(team_wid - 40.0) * 1.2))
            # Midfield compactness: gap between def and mid lines
            compactness_mid = max(0.0, min(100.0, 100.0 - (line_gap * 3.5)))
            # Overall weighted compactness
            compactness = compactness_v * 0.4 + compactness_h * 0.3 + compactness_mid * 0.3

            # --- Advanced KPIs ---
            # Press resistance: ability to cope under pressure (higher pitch control + lower press dist = better)
            press_resistance = max(0.0, min(100.0, (pitch_ctrl * 0.6) + max(0.0, (12.0 - press_idx) * 3.5)))

            # Width utilization: optimal width is 38-48m; extremes are penalized
            if 38.0 <= team_wid <= 48.0:
                width_utilization = min(100.0, 85.0 + (team_wid - 38.0) * 1.5)
            else:
                width_utilization = max(0.0, min(100.0, 85.0 - abs(team_wid - 43.0) * 2.5))

            # Line staggering: measures how well the three lines are spaced (optimal is 15-20m each)
            ideal_gap = 17.5
            stagger_dev = abs(line_gap - ideal_gap) + abs(mid_gap - ideal_gap)
            line_staggering = max(0.0, min(100.0, 100.0 - stagger_dev * 2.5))

            # Overload score: pitch control + compactness = ability to create numerical advantages
            overload_score = max(0.0, min(100.0, (pitch_ctrl * 0.55) + (compactness * 0.45)))

            # Transition speed: how fast the team moves (5-8 m/s optimal)
            if 4.5 <= avg_speed <= 7.5:
                transition_speed = min(100.0, 60.0 + (avg_speed - 4.5) * 13.3)
            else:
                transition_speed = max(0.0, 100.0 - abs(avg_speed - 6.0) * 15.0)

            # Midfield control
            midfield_control = max(0.0, min(100.0, pitch_ctrl))

            # Pressing intensity
            pressing = max(0.0, min(100.0, 100.0 - (press_idx * 5.0)))

            # Defensive shape
            defensive_shape = max(0.0, min(100.0, 100.0 - (line_gap * 2.0)))

            return {
                "compactness": round(compactness, 1),
                "compactness_vertical": round(compactness_v, 1),
                "compactness_horizontal": round(compactness_h, 1),
                "compactness_midfield": round(compactness_mid, 1),
                "midfield_control": round(midfield_control, 1),
                "pressing": round(pressing, 1),
                "defensive_shape": round(defensive_shape, 1),
                "press_resistance": round(press_resistance, 1),
                "width_utilization": round(width_utilization, 1),
                "line_staggering": round(line_staggering, 1),
                "overload_score": round(overload_score, 1),
                "transition_speed": round(transition_speed, 1),
            }

        red = aggregate("team_0")
        blue = aggregate("team_1")

        def weighted_power(m: dict) -> float:
            """Professional Tactical Power Index — weighted across all KPIs."""
            if not m:
                return 0.0
            return (
                m.get("compactness", 0) * 0.20 +
                m.get("midfield_control", 0) * 0.18 +
                m.get("pressing", 0) * 0.12 +
                m.get("defensive_shape", 0) * 0.14 +
                m.get("press_resistance", 0) * 0.10 +
                m.get("width_utilization", 0) * 0.08 +
                m.get("line_staggering", 0) * 0.08 +
                m.get("overload_score", 0) * 0.06 +
                m.get("transition_speed", 0) * 0.04
            )

        red_power = weighted_power(red)
        blue_power = weighted_power(blue)
        total = max(red_power + blue_power, 1.0)

        red_win_prob = round(max(0.0, min(100.0, (red_power / total) * 100.0)), 1)
        blue_win_prob = round(max(0.0, min(100.0, (blue_power / total) * 100.0)), 1)

        # Confidence: based on how many frames had data
        frames_with_t0 = sum(1 for f in metrics_timeline if "team_0" in f)
        frames_with_t1 = sum(1 for f in metrics_timeline if "team_1" in f)
        total_frames = max(len(metrics_timeline), 1)
        confidence_red = round(min(100.0, (frames_with_t0 / total_frames) * 100.0), 1)
        confidence_blue = round(min(100.0, (frames_with_t1 / total_frames) * 100.0), 1)

        # Aggregate zonal data
        zonal_aggregated = []
        all_zonal_frames = [f.get("zonal_data", []) for f in metrics_timeline if "zonal_data" in f]
        if all_zonal_frames:
            num_zones = len(all_zonal_frames[0])
            for z_idx in range(num_zones):
                zone_frames = [frame[z_idx] for frame in all_zonal_frames if len(frame) > z_idx]
                if not zone_frames:
                    continue
                
                z_0_densities = [f["team_0_density"] for f in zone_frames]
                z_1_densities = [f["team_1_density"] for f in zone_frames]
                z_ctrl = [f["control_pct"] for f in zone_frames]
                z_press = [f["pressure_index"] for f in zone_frames]
                z_vuln = [f["vulnerability"] for f in zone_frames]
                z_threat = [f["threat_level"] for f in zone_frames]
                z_overload = [f.get("overload", 0) for f in zone_frames]
                z_comp = [f.get("compactness", 0) for f in zone_frames]
                z_trans = [f.get("transition_potential", 0) for f in zone_frames]

                zonal_aggregated.append({
                    "zone_id": z_idx,
                    "x_range": zone_frames[0]["x_range"],
                    "y_range": zone_frames[0]["y_range"],
                    "avg_team_0_density": round(statistics.mean(z_0_densities), 2),
                    "avg_team_1_density": round(statistics.mean(z_1_densities), 2),
                    "avg_control_pct": round(statistics.mean(z_ctrl), 1),
                    "avg_pressure_index": round(statistics.mean(z_press), 1),
                    "has_ball_frequency": round(sum(1 for f in zone_frames if f["has_ball"]) / len(zone_frames), 2),
                    "vulnerability": round(statistics.mean(z_vuln), 1),
                    "threat_level": round(statistics.mean(z_threat), 1),
                    "overload_score": round(statistics.mean(z_overload), 1),
                    "local_compactness": round(statistics.mean(z_comp), 1),
                    "transition_potential": round(statistics.mean(z_trans), 1),
                })

        return {
            "team_0": {
                "compactness": red.get("compactness", 0),
                "compactness_vertical": red.get("compactness_vertical", 0),
                "compactness_horizontal": red.get("compactness_horizontal", 0),
                "compactness_midfield": red.get("compactness_midfield", 0),
                "control": red.get("midfield_control", 0),
                "intensity": red.get("pressing", 0),
                "defensive_shape": red.get("defensive_shape", 0),
                "press_resistance": red.get("press_resistance", 0),
                "width_utilization": red.get("width_utilization", 0),
                "line_staggering": red.get("line_staggering", 0),
                "overload_score": red.get("overload_score", 0),
                "transition_speed": red.get("transition_speed", 0),
                "tactical_power": round(red_power, 1),
                "win_prob": red_win_prob,
                "confidence_pct": confidence_red,
            },
            "team_1": {
                "compactness": blue.get("compactness", 0),
                "compactness_vertical": blue.get("compactness_vertical", 0),
                "compactness_horizontal": blue.get("compactness_horizontal", 0),
                "compactness_midfield": blue.get("compactness_midfield", 0),
                "control": blue.get("midfield_control", 0),
                "intensity": blue.get("pressing", 0),
                "defensive_shape": blue.get("defensive_shape", 0),
                "press_resistance": blue.get("press_resistance", 0),
                "width_utilization": blue.get("width_utilization", 0),
                "line_staggering": blue.get("line_staggering", 0),
                "overload_score": blue.get("overload_score", 0),
                "transition_speed": blue.get("transition_speed", 0),
                "tactical_power": round(blue_power, 1),
                "win_prob": blue_win_prob,
                "confidence_pct": confidence_blue,
            },
            "team_red_score": round(red_power, 2),
            "team_blue_score": round(blue_power, 2),
            "win_probability": {
                "team_red": red_win_prob,
                "team_blue": blue_win_prob,
            },
            "zonal_analytics": zonal_aggregated,
        }

    def evaluate_team(
        self, frame_idx: int, team_name: str, metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Backwards-compatible frame-by-frame evaluator.

        The production pipeline now uses `evaluate_timeline()` for chunk-level aggregation.
        """
        triggers: list[dict[str, Any]] = []

        # Rule 1: Midfield Disconnect
        gap_def_mid = metrics.get("line_gap_def_mid_m", 0.0)
        gap_mid_att = metrics.get("line_gap_mid_att_m", 0.0)
        if gap_def_mid > self.MAX_LINE_GAP or gap_mid_att > self.MAX_LINE_GAP:
            max_gap = max(float(gap_def_mid), float(gap_mid_att))
            triggers.append(
                {
                    "flaw": "Midfield Disconnect",
                    "severity": "High",
                    "evidence": f"Massive {max_gap:.1f}m gap between lines.",
                }
            )

        # Rule 2: Suicidal High Line
        deepest_x = float(metrics.get("deepest_x", 0.0))
        control = float(metrics.get("pitch_control_pct", 50.0))
        if deepest_x > self.HIGH_LINE_X and control < self.MIN_PITCH_CONTROL:
            triggers.append(
                {
                    "flaw": "Suicidal High Line",
                    "severity": "Critical",
                    "evidence": (
                        f"Defense is pushed up to {deepest_x:.1f}m, but controlling only {control:.1f}% of the pitch."
                    ),
                }
            )

        # Rule 3: Parked Bus (Extreme Low Block)
        if (
            deepest_x < self.LOW_BLOCK_X
            and float(metrics.get("area_sq_meters", 1000.0)) < 600.0
        ):
            triggers.append(
                {
                    "flaw": "Parked Bus",
                    "severity": "Medium",
                    "evidence": (
                        f"Deepest defender at {deepest_x:.1f}m with highly compressed area."
                    ),
                }
            )

        # Rule 4: Lethargic Press
        pressure = float(metrics.get("pressure_index_m", 0.0))
        if pressure > self.POOR_PRESS_DIST:
            triggers.append(
                {
                    "flaw": "Lethargic Press",
                    "severity": "Medium",
                    "evidence": (
                        f"Average distance to nearest opponent is a loose {pressure:.1f}m."
                    ),
                }
            )

        # Rule 5: Over-Stretched Formation
        length = float(metrics.get("team_length_m", 0.0))
        if length > self.MAX_TEAM_LENGTH:
            triggers.append(
                {
                    "flaw": "Over-Stretched Formation",
                    "severity": "High",
                    "evidence": (
                        f"Team length is {length:.1f}m, making them easy to play through."
                    ),
                }
            )

        return triggers


def _severity_rank(severity: str) -> int:
    # Higher rank = more severe.
    return {
        "Critical": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1,
    }.get(severity, 0)


def _evidence_text(frequency_pct: float, average_metric_m: float) -> str:
    # Keep this stable for automated parsing/tests.
    return (
        f"Team spent {frequency_pct:.1f}% of the time with this flaw, "
        f"averaging a {average_metric_m:.1f}m gap."
    )


def evaluate_timeline(
    metrics_timeline: list[dict[str, Any]],
    *,
    min_frequency_pct: float = 10.0,
    top_k_per_team: int = 3,
) -> list[dict[str, Any]]:
    """
    Evaluate tactical rules cumulatively across the full metrics timeline.

    Produces chunk-level aggregated insights rather than frame-by-frame triggers.
    """
    total_frames = len(metrics_timeline)
    if total_frames == 0:
        return []

    engine = RuleEngine()
    team_ids: tuple[Literal["team_0", "team_1"], Literal["team_0", "team_1"]] = (
        "team_0",
        "team_1",
    )

    # 1. Calculate Tactical Scores and Win Probability
    summary = engine.calculate_tactical_scores(metrics_timeline)

    # Data coverage confidence
    frames_t0 = sum(1 for f in metrics_timeline if "team_0" in f)
    frames_t1 = sum(1 for f in metrics_timeline if "team_1" in f)
    coverage_t0 = round((frames_t0 / total_frames) * 100.0, 1)
    coverage_t1 = round((frames_t1 / total_frames) * 100.0, 1)
    summary_confidence = round((coverage_t0 + coverage_t1) / 2.0, 1)

    t0 = summary["team_0"]
    t1 = summary["team_1"]

    # 2. Inject Match Summary with full KPI data
    global_insights: list[dict[str, Any]] = [
        {
            "team_id": "global",
            "flaw": "Match Summary",
            "severity": "Info",
            "frequency_pct": 100.0,
            "confidence_pct": summary_confidence,
            "confidence_reason": f"Based on {total_frames} frames with {coverage_t0:.0f}%/{coverage_t1:.0f}% data coverage for Red/Blue.",
            "evidence": (
                f"Tactical Power: Red {t0.get('tactical_power', 0):.1f} vs Blue {t1.get('tactical_power', 0):.1f}. "
                f"Win Probability: Red {t0.get('win_prob', 50)}% | Blue {t1.get('win_prob', 50)}%. "
                f"Compactness: Red {t0.get('compactness', 0):.0f} / Blue {t1.get('compactness', 0):.0f}. "
                f"Press Resistance: Red {t0.get('press_resistance', 0):.0f} / Blue {t1.get('press_resistance', 0):.0f}. "
                f"Width Utilization: Red {t0.get('width_utilization', 0):.0f} / Blue {t1.get('width_utilization', 0):.0f}. "
                f"Line Staggering: Red {t0.get('line_staggering', 0):.0f} / Blue {t1.get('line_staggering', 0):.0f}. "
                f"Transition Speed: Red {t0.get('transition_speed', 0):.0f} / Blue {t1.get('transition_speed', 0):.0f}."
            ),
            "summary_data": summary,
        }
    ]

    output: list[dict[str, Any]] = []

    for team_id in team_ids:
        current_team_insights: list[dict[str, Any]] = []

        # --- Rule 1: Midfield Disconnect
        mid_gap_values: list[float] = []
        for frame in metrics_timeline:
            metrics = frame.get(team_id, {}) or {}
            gap_def_mid = float(metrics.get("line_gap_def_mid_m", 0.0) or 0.0)
            gap_mid_att = float(metrics.get("line_gap_mid_att_m", 0.0) or 0.0)
            if gap_def_mid > engine.MAX_LINE_GAP or gap_mid_att > engine.MAX_LINE_GAP:
                mid_gap_values.append(max(gap_def_mid, gap_mid_att))
        if mid_gap_values:
            frequency_pct = (len(mid_gap_values) / total_frames) * 100.0
            if frequency_pct > min_frequency_pct:
                average_metric_m = statistics.mean(mid_gap_values)
                max_gap = max(mid_gap_values)
                # Confidence: frequency certainty × threshold margin × data coverage
                threshold_margin = min(1.0, (average_metric_m - engine.MAX_LINE_GAP) / engine.MAX_LINE_GAP)
                frames_with_data = sum(1 for f in metrics_timeline if team_id in f)
                data_coverage = frames_with_data / total_frames
                confidence_pct = round(min(99.0, (frequency_pct / 100.0) * 0.45 + threshold_margin * 0.35 + data_coverage * 0.20) * 100.0, 1)
                current_team_insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Midfield Disconnect",
                        "severity": "High",
                        "frequency_pct": frequency_pct,
                        "confidence_pct": confidence_pct,
                        "confidence_reason": f"Avg gap {average_metric_m:.1f}m (threshold {engine.MAX_LINE_GAP}m), detected in {frequency_pct:.0f}% of frames.",
                        "evidence": (
                            f"Avg line gap {average_metric_m:.1f}m (peak {max_gap:.1f}m) — "
                            f"threshold is {engine.MAX_LINE_GAP}m. Violated in {frequency_pct:.0f}% of frames."
                        ),
                    }
                )

        # --- Rule 2: Suicidal High Line
        suicidal_high_values: list[float] = []
        for frame in metrics_timeline:
            metrics = frame.get(team_id, {}) or {}
            deepest_x = float(metrics.get("deepest_x", 0.0) or 0.0)
            pitch_control = float(metrics.get("pitch_control_pct", 50.0) or 50.0)
            if deepest_x > engine.HIGH_LINE_X and pitch_control < engine.MIN_PITCH_CONTROL:
                suicidal_high_values.append(deepest_x)
        if suicidal_high_values:
            frequency_pct = (len(suicidal_high_values) / total_frames) * 100.0
            if frequency_pct > min_frequency_pct:
                average_metric_m = statistics.mean(suicidal_high_values)
                threshold_margin = min(1.0, (average_metric_m - engine.HIGH_LINE_X) / engine.HIGH_LINE_X)
                frames_with_data = sum(1 for f in metrics_timeline if team_id in f)
                data_coverage = frames_with_data / total_frames
                confidence_pct = round(min(99.0, (frequency_pct / 100.0) * 0.50 + threshold_margin * 0.30 + data_coverage * 0.20) * 100.0, 1)
                current_team_insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Suicidal High Line",
                        "severity": "Critical",
                        "frequency_pct": frequency_pct,
                        "confidence_pct": confidence_pct,
                        "confidence_reason": f"Defensive line avg {average_metric_m:.1f}m high with <{engine.MIN_PITCH_CONTROL}% pitch control for {frequency_pct:.0f}% of frames.",
                        "evidence": (
                            f"Defensive line pushed to avg {average_metric_m:.1f}m (threshold {engine.HIGH_LINE_X}m) "
                            f"with insufficient pitch control. Vulnerable in {frequency_pct:.0f}% of frames."
                        ),
                    }
                )

        # --- Rule 3: Parked Bus (Extreme Low Block)
        parked_bus_values: list[float] = []
        for frame in metrics_timeline:
            metrics = frame.get(team_id, {}) or {}
            deepest_x = float(metrics.get("deepest_x", 0.0) or 0.0)
            area_sq_meters = float(metrics.get("area_sq_meters", 1000.0) or 1000.0)
            if deepest_x < engine.LOW_BLOCK_X and area_sq_meters < 600.0:
                # Represent as a positive "depth distance".
                parked_bus_values.append(abs(deepest_x))
        if parked_bus_values:
            frequency_pct = (len(parked_bus_values) / total_frames) * 100.0
            if frequency_pct > min_frequency_pct:
                average_metric_m = statistics.mean(parked_bus_values)
                threshold_margin = min(1.0, (engine.LOW_BLOCK_X - average_metric_m) / max(engine.LOW_BLOCK_X, 1.0))
                frames_with_data = sum(1 for f in metrics_timeline if team_id in f)
                data_coverage = frames_with_data / total_frames
                confidence_pct = round(min(99.0, (frequency_pct / 100.0) * 0.45 + threshold_margin * 0.35 + data_coverage * 0.20) * 100.0, 1)
                current_team_insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Parked Bus",
                        "severity": "Medium",
                        "frequency_pct": frequency_pct,
                        "confidence_pct": confidence_pct,
                        "confidence_reason": f"Deepest defender avg {average_metric_m:.1f}m deep with compressed area, in {frequency_pct:.0f}% of frames.",
                        "evidence": (
                            f"Team entrenched — deepest defender avg {average_metric_m:.1f}m from goal. "
                            f"Highly compressed formation area (<600 sq.m) in {frequency_pct:.0f}% of frames."
                        ),
                    }
                )

        # --- Rule 4: Lethargic Press
        lethargic_press_values: list[float] = []
        for frame in metrics_timeline:
            metrics = frame.get(team_id, {}) or {}
            pressure = float(metrics.get("pressure_index_m", 0.0) or 0.0)
            if pressure > engine.POOR_PRESS_DIST:
                lethargic_press_values.append(pressure)
        if lethargic_press_values:
            frequency_pct = (len(lethargic_press_values) / total_frames) * 100.0
            if frequency_pct > min_frequency_pct:
                average_metric_m = statistics.mean(lethargic_press_values)
                threshold_margin = min(1.0, (average_metric_m - engine.POOR_PRESS_DIST) / engine.POOR_PRESS_DIST)
                frames_with_data = sum(1 for f in metrics_timeline if team_id in f)
                data_coverage = frames_with_data / total_frames
                confidence_pct = round(min(99.0, (frequency_pct / 100.0) * 0.50 + threshold_margin * 0.30 + data_coverage * 0.20) * 100.0, 1)
                current_team_insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Lethargic Press",
                        "severity": "Medium",
                        "frequency_pct": frequency_pct,
                        "confidence_pct": confidence_pct,
                        "confidence_reason": f"Avg pressing distance {average_metric_m:.1f}m — {average_metric_m - engine.POOR_PRESS_DIST:.1f}m beyond the {engine.POOR_PRESS_DIST}m threshold.",
                        "evidence": (
                            f"Average pressing distance {average_metric_m:.1f}m to nearest opponent "
                            f"({engine.POOR_PRESS_DIST}m threshold exceeded in {frequency_pct:.0f}% of frames). "
                            f"Opposition able to build without pressure."
                        ),
                    }
                )

        # --- Rule 5: Over-Stretched Formation
        over_stretched_values: list[float] = []
        for frame in metrics_timeline:
            metrics = frame.get(team_id, {}) or {}
            length = float(metrics.get("team_length_m", 0.0) or 0.0)
            if length > engine.MAX_TEAM_LENGTH:
                over_stretched_values.append(length)
        if over_stretched_values:
            frequency_pct = (len(over_stretched_values) / total_frames) * 100.0
            if frequency_pct > min_frequency_pct:
                average_metric_m = statistics.mean(over_stretched_values)
                threshold_margin = min(1.0, (average_metric_m - engine.MAX_TEAM_LENGTH) / engine.MAX_TEAM_LENGTH)
                frames_with_data = sum(1 for f in metrics_timeline if team_id in f)
                data_coverage = frames_with_data / total_frames
                confidence_pct = round(min(99.0, (frequency_pct / 100.0) * 0.45 + threshold_margin * 0.35 + data_coverage * 0.20) * 100.0, 1)
                current_team_insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Over-Stretched Formation",
                        "severity": "High",
                        "frequency_pct": frequency_pct,
                        "confidence_pct": confidence_pct,
                        "confidence_reason": f"Team length avg {average_metric_m:.1f}m ({average_metric_m - engine.MAX_TEAM_LENGTH:.1f}m beyond the {engine.MAX_TEAM_LENGTH}m safe limit).",
                        "evidence": (
                            f"Team vertically stretched to avg {average_metric_m:.1f}m end-to-end "
                            f"(safe limit: {engine.MAX_TEAM_LENGTH}m). Creates passable channels in {frequency_pct:.0f}% of frames."
                        ),
                    }
                )

        # Return top 1-3 most severe/frequent insights per team.
        current_team_insights.sort(
            key=lambda ins: (
                _severity_rank(str(ins.get("severity", ""))),
                float(ins.get("frequency_pct", 0.0)),
            ),
            reverse=True,
        )

        output.extend(current_team_insights[: max(1, top_k_per_team)])

    # Prepend the global summary to the output
    return global_insights + output


def run_engine(
    input_path: Path | None = None,
    output_path: Path | None = None,
    *,
    write_output: bool = True,
) -> list[dict[str, Any]]:
    """
    Evaluate tactical metrics and produce chunk-level aggregated insights.

    Returns a list of chunk insights (one per team+flaw) rather than frame-by-frame triggers.
    """
    input_path = input_path or DEFAULT_INPUT_PATH
    output_path = output_path or DEFAULT_OUTPUT_PATH

    if not input_path.is_file():
        raise FileNotFoundError(f"Tactical metrics not found: {input_path}")

    with input_path.open(encoding="utf-8") as f:
        timeline: list[dict[str, Any]] = json.load(f)

    chunk_insights = evaluate_timeline(timeline)

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(chunk_insights, f, indent=4, ensure_ascii=False)
        LOGGER.info(
            "Rule engine produced %s chunk-level insight(s); wrote %s",
            len(chunk_insights),
            output_path,
        )

    return chunk_insights


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run_engine()
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)


if __name__ == "__main__":
    main()
