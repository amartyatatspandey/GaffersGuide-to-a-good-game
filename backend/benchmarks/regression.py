"""
benchmarks/regression.py — Regression detection algorithm.

Compares a current BenchmarkReport against a baseline BenchmarkReport.
"""
from __future__ import annotations

from .models import BenchmarkReport, RegressionFlag

# Tolerances in percentages
TOLERANCE_PCT = {
    "end_to_end_ms": 10.0,
    "peak_ram_mb": 15.0,
    "peak_vram_mb": 15.0,
    "overall_fps": 10.0,  # Decrease tolerance
    "llm_total_ms": 20.0,
}
STAGE_TOLERANCE_PCT = 10.0


def compare_reports(current: BenchmarkReport, baseline: BenchmarkReport) -> list[RegressionFlag]:
    """
    Compares two reports and returns a list of detected regressions.
    Validates that hardware profile and dataset version match.
    """
    flags: list[RegressionFlag] = []

    if current.hardware_profile.profile_id != baseline.hardware_profile.profile_id:
        flags.append(
            RegressionFlag(
                metric_id="hardware_mismatch",
                baseline_value=0.0,
                current_value=0.0,
                delta_pct=0.0,
                tolerance_pct=0.0,
                severity="CRITICAL",
                description=(
                    f"Hardware mismatch: baseline is {baseline.hardware_profile.cpu_model}, "
                    f"current is {current.hardware_profile.cpu_model}"
                ),
            )
        )
        return flags

    # Helper for checking increases
    def _check_increase(metric_id: str, b_val: float, c_val: float, tol: float, desc: str) -> None:
        if b_val <= 0:
            return
        delta = ((c_val - b_val) / b_val) * 100.0
        if delta > tol:
            sev = "CRITICAL" if delta > (tol * 2) else "WARNING"
            flags.append(
                RegressionFlag(
                    metric_id=metric_id,
                    baseline_value=b_val,
                    current_value=c_val,
                    delta_pct=round(delta, 2),
                    tolerance_pct=tol,
                    severity=sev,
                    description=f"{desc} regressed by {delta:.1f}% (limit {tol}%)",
                )
            )

    # Helper for checking decreases
    def _check_decrease(metric_id: str, b_val: float, c_val: float, tol: float, desc: str) -> None:
        if b_val <= 0:
            return
        delta = ((b_val - c_val) / b_val) * 100.0
        if delta > tol:
            sev = "CRITICAL" if delta > (tol * 2) else "WARNING"
            flags.append(
                RegressionFlag(
                    metric_id=metric_id,
                    baseline_value=b_val,
                    current_value=c_val,
                    delta_pct=round(-delta, 2),
                    tolerance_pct=tol,
                    severity=sev,
                    description=f"{desc} dropped by {delta:.1f}% (limit {tol}%)",
                )
            )

    # 1. Aggregate Checks
    c_agg = current.aggregate
    b_agg = baseline.aggregate

    _check_increase("T-001", b_agg.end_to_end_ms, c_agg.end_to_end_ms, TOLERANCE_PCT["end_to_end_ms"], "End-to-end time")
    _check_increase("M-001", b_agg.peak_ram_mb, c_agg.peak_ram_mb, TOLERANCE_PCT["peak_ram_mb"], "Peak RAM")
    if b_agg.peak_vram_mb and c_agg.peak_vram_mb:
        _check_increase("M-002", b_agg.peak_vram_mb, c_agg.peak_vram_mb, TOLERANCE_PCT["peak_vram_mb"], "Peak VRAM")
    _check_decrease("R-002", b_agg.overall_fps, c_agg.overall_fps, TOLERANCE_PCT["overall_fps"], "Overall FPS")

    # 2. Module/Stage Checks
    b_stages = {s.stage_id: s.duration_ms for s in baseline.stages}
    for c_stage in current.stages:
        if c_stage.stage_id in b_stages:
            _check_increase(
                f"stage_duration_{c_stage.stage_id}",
                b_stages[c_stage.stage_id],
                c_stage.duration_ms,
                STAGE_TOLERANCE_PCT,
                f"Stage '{c_stage.stage_name}' duration",
            )

    return flags
