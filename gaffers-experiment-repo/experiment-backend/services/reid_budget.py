from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ReIDBudgetStats:
    invocations: int
    reid_ms: float
    id_switch_rate: float


def run_reid_budget_controller(
    *,
    frames_processed: int,
    quality_mode: str,
) -> ReIDBudgetStats:
    if quality_mode == "fast":
        invocations = max(1, frames_processed // 300)
    elif quality_mode == "high":
        invocations = max(1, frames_processed // 120)
    else:
        invocations = max(1, frames_processed // 180)
    reid_ms = float(invocations) * 0.8
    id_switch_rate = max(0.0, 0.06 - (invocations * 0.0005))
    return ReIDBudgetStats(
        invocations=invocations,
        reid_ms=reid_ms,
        id_switch_rate=round(id_switch_rate, 4),
    )
