"""Tactical metric calculators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gaffers_guide.core.types import MatchState
from gaffers_guide.pipeline.generate_analytics import TacticalAnalyzer


@dataclass
class MetricsCalculatorAdapter:
    """Adapter around existing TacticalAnalyzer implementation."""

    analyzer: TacticalAnalyzer = TacticalAnalyzer()

    def calculate(self, states: list[MatchState]) -> dict[str, Any]:
        # Existing analyzer expects point-series API; provide a stable placeholder
        # contract until full tactical extraction is complete.
        return {"frames": len(states)}
