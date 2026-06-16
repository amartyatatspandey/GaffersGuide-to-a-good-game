"""
Event Intelligence Layer — Data Models
=======================================

All Pydantic schemas used by the Event Intelligence Layer.
Designed to be:
  - Self-describing (no external context needed to read a record)
  - LLM-ready (description field is natural language)
  - Evidence-first (every record contains enough info to retrieve its clip)
  - JSON-serialisable with zero custom encoders
"""
from __future__ import annotations

from typing import Literal, Any
from pydantic import BaseModel, Field
import uuid


# ──────────────────────────────────────────────────────────────────────────────
# Core event record
# ──────────────────────────────────────────────────────────────────────────────

EventCategory = Literal["movement", "positional", "threat", "shape", "transition"]

ConfidenceLabel = Literal["high", "medium", "low", "very_low"]

PitchZone = Literal[
    "defensive_third",
    "middle_third",
    "final_third",
    "left_channel",
    "right_channel",
    "half_space_left",
    "half_space_right",
    "central_corridor",
    "box",
    "wide_left",
    "wide_right",
    "unknown",
]


class EventRecord(BaseModel):
    """A single detected football event.

    All spatial coordinates are in metres, origin at centre circle.
    X is positive toward attacking goal, Y is positive toward right touchline.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str  # Ontology code, e.g. "THR_001"
    event_name: str  # Human label, e.g. "Dangerous Run"
    category: EventCategory

    # ── Attribution ───────────────────────────────────────────────────────────
    player_id: int | None = None  # None for team-level shape/transition events
    team_id: str  # "team_0" | "team_1"

    # ── Temporal anchors ──────────────────────────────────────────────────────
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float

    # ── Spatial context ───────────────────────────────────────────────────────
    pitch_zone: PitchZone = "unknown"
    start_radar_pt: tuple[float, float] | None = None
    end_radar_pt: tuple[float, float] | None = None
    peak_radar_pt: tuple[float, float] | None = None  # Most extreme point

    # ── Scoring ───────────────────────────────────────────────────────────────
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(ge=0.0, le=1.0)
    threat_contribution: float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def confidence_label(self) -> ConfidenceLabel:
        if self.confidence >= 0.85:
            return "high"
        if self.confidence >= 0.65:
            return "medium"
        if self.confidence >= 0.40:
            return "low"
        return "very_low"

    # ── Natural language ──────────────────────────────────────────────────────
    description: str = ""  # Auto-generated, LLM-ready
    tags: list[str] = Field(default_factory=list)

    # ── Evidence retrieval ────────────────────────────────────────────────────
    clip_id: str | None = None
    clip_start_frame: int = 0
    clip_end_frame: int = 0

    # ── Metadata ──────────────────────────────────────────────────────────────
    job_id: str = ""
    detected_at: str = ""
    detector_version: str = "event_layer_v1.0"


# ──────────────────────────────────────────────────────────────────────────────
# Threat profiles
# ──────────────────────────────────────────────────────────────────────────────

class PlayerThreatProfile(BaseModel):
    """Aggregated threat record for one player across a full analysis job."""

    player_id: int
    team_id: str
    job_id: str

    # Raw event counts and IDs per type
    event_counts: dict[str, int] = Field(default_factory=dict)
    event_ids: dict[str, list[str]] = Field(default_factory=dict)

    # Computed scores
    threat_score: float = Field(default=0.0, ge=0.0, le=100.0)
    threat_rank: int = 0  # 1 = highest threat on team

    # Explanation
    primary_threat_types: list[str] = Field(default_factory=list)
    explanation: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Clip records
# ──────────────────────────────────────────────────────────────────────────────

AnnotationType = Literal["run_trail", "zone_highlight", "arrow", "player_circle"]


class ClipRecord(BaseModel):
    """Video clip extracted to support a tactical observation."""

    clip_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    job_id: str
    event_id: str  # Primary event this clip supports

    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float

    file_path: str | None = None
    thumbnail_path: str | None = None

    # Visual annotations to apply at render time
    highlight_player_ids: list[int] = Field(default_factory=list)
    annotation_type: AnnotationType = "player_circle"
    annotation_data: dict[str, Any] = Field(default_factory=dict)

    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Human-readable context for the coach UI
    label: str = ""  # E.g. "Dangerous Run — left channel uncontested"
    confidence_pct: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Evidence bundles
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceBundle(BaseModel):
    """Curated set of clips attached to a single tactical observation."""

    bundle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tactical_observation: str
    query_intent: str

    clips: list[ClipRecord] = Field(default_factory=list)
    supporting_event_ids: list[str] = Field(default_factory=list)

    generated_at: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Evidence query
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceQuery(BaseModel):
    """Structured query for retrieving supporting clips for a tactical observation."""

    observation_text: str

    # Extracted entities
    player_ids: list[int] | None = None
    team_id: str | None = None
    event_types: list[str] = Field(default_factory=list)
    time_window: tuple[float, float] | None = None  # (start_s, end_s)

    # Retrieval preferences
    min_confidence: float = 0.65
    min_importance: float = 0.50
    max_results: int = 3
    prefer_variety: bool = True  # Avoid selecting 3 clips of the same event type


# ──────────────────────────────────────────────────────────────────────────────
# Full event index (persisted per job)
# ──────────────────────────────────────────────────────────────────────────────

class EventIndex(BaseModel):
    """Container for all events detected in a single analysis job."""

    job_id: str
    detector_version: str = "event_layer_v1.0"
    total_frames: int = 0
    fps: float = 25.0
    events: list[EventRecord] = Field(default_factory=list)
    generated_at: str = ""

    def filter(
        self,
        *,
        player_ids: list[int] | None = None,
        team_id: str | None = None,
        event_types: list[str] | None = None,
        categories: list[str] | None = None,
        min_confidence: float = 0.0,
        min_importance: float = 0.0,
    ) -> list[EventRecord]:
        """Return filtered events. All criteria are ANDed together."""
        result = self.events
        if player_ids is not None:
            result = [e for e in result if e.player_id in player_ids]
        if team_id is not None:
            result = [e for e in result if e.team_id == team_id]
        if event_types:
            result = [e for e in result if e.event_type in event_types]
        if categories:
            result = [e for e in result if e.category in categories]
        result = [e for e in result if e.confidence >= min_confidence]
        result = [e for e in result if e.importance >= min_importance]
        return result

    def stats(self) -> dict[str, Any]:
        """Return a summary of the index contents."""
        from collections import Counter
        type_counts = Counter(e.event_type for e in self.events)
        cat_counts = Counter(e.category for e in self.events)
        return {
            "total_events": len(self.events),
            "by_type": dict(type_counts.most_common()),
            "by_category": dict(cat_counts),
            "players_with_events": len({e.player_id for e in self.events if e.player_id is not None}),
        }
