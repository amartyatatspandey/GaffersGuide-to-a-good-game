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
        self.POOR_PRESS_DIST = 6.0  # meters
        self.MAX_TEAM_LENGTH = 45.0  # meters
        self.MIN_PITCH_CONTROL = 40.0  # percentage

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

    output: list[dict[str, Any]] = []

    for team_id in team_ids:
        insights: list[dict[str, Any]] = []

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
                insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Midfield Disconnect",
                        "severity": "High",
                        "frequency_pct": frequency_pct,
                        "evidence": _evidence_text(
                            frequency_pct=frequency_pct,
                            average_metric_m=average_metric_m,
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
                insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Suicidal High Line",
                        "severity": "Critical",
                        "frequency_pct": frequency_pct,
                        "evidence": _evidence_text(
                            frequency_pct=frequency_pct,
                            average_metric_m=average_metric_m,
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
                insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Parked Bus",
                        "severity": "Medium",
                        "frequency_pct": frequency_pct,
                        "evidence": _evidence_text(
                            frequency_pct=frequency_pct,
                            average_metric_m=average_metric_m,
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
                insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Lethargic Press",
                        "severity": "Medium",
                        "frequency_pct": frequency_pct,
                        "evidence": _evidence_text(
                            frequency_pct=frequency_pct,
                            average_metric_m=average_metric_m,
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
                insights.append(
                    {
                        "team_id": team_id,
                        "flaw": "Over-Stretched Formation",
                        "severity": "High",
                        "frequency_pct": frequency_pct,
                        "evidence": _evidence_text(
                            frequency_pct=frequency_pct,
                            average_metric_m=average_metric_m,
                        ),
                    }
                )

        # Return top 1-3 most severe/frequent insights per team.
        insights.sort(
            key=lambda ins: (
                _severity_rank(str(ins.get("severity", ""))),
                float(ins.get("frequency_pct", 0.0)),
            ),
            reverse=True,
        )

        output.extend(insights[: max(1, top_k_per_team)])

    return output


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
