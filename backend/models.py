from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChunkTacticalInsight(BaseModel):
    """
    Aggregated tactical flaw over an entire video chunk (macro-trend).

    frequency_pct is the percent of frames in which the rule was violated.
    """

    team_id: Literal["team_0", "team_1"]
    flaw: str
    severity: str
    frequency_pct: float = Field(..., ge=0.0, le=100.0)
    evidence: str

