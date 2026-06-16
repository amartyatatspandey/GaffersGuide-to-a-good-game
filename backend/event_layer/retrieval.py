"""
Event Intelligence Layer — Evidence Retriever
=============================================

Retrieves supporting video clips for any tactical observation.

Pipeline:
  Tactical Observation (text)
    → EvidenceQuery construction (keyword → event type mapping)
    → Event matching (filter EventIndex)
    → Candidate ranking (relevance scoring)
    → Clip selection (top-N with variety enforcement)
    → Clip deduplication (merge overlapping clips)
    → EvidenceBundle assembly
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from event_layer.models import (
    ClipRecord,
    EvidenceBundle,
    EvidenceQuery,
    EventIndex,
    EventRecord,
)
from event_layer.ontology import keywords_to_event_types, THRESHOLDS

LOGGER = logging.getLogger(__name__)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Annotation builders
# ──────────────────────────────────────────────────────────────────────────────

_THREAT_COLOR = "#FF3B30"   # Red for threat events
_SHAPE_COLOR  = "#FF9500"   # Orange for shape events
_MOV_COLOR    = "#30D158"   # Green for movement events
_POS_COLOR    = "#0A84FF"   # Blue for positional events
_TRN_COLOR    = "#BF5AF2"   # Purple for transition events

_CATEGORY_COLORS: dict[str, str] = {
    "threat": _THREAT_COLOR,
    "shape": _SHAPE_COLOR,
    "movement": _MOV_COLOR,
    "positional": _POS_COLOR,
    "transition": _TRN_COLOR,
}


def _build_annotation(event: EventRecord) -> dict[str, Any]:
    color = _CATEGORY_COLORS.get(event.category, "#FFFFFF")

    if event.category in ("threat", "movement") and event.player_id is not None:
        return {
            "type": "run_trail",
            "player_id": event.player_id,
            "highlight_color": color,
            "trail_start_frame": event.start_frame,
            "trail_end_frame": event.end_frame,
            "show_defender_positions": event.category == "threat",
            "show_zone_overlay": True,
        }
    elif event.category == "shape":
        return {
            "type": "zone_highlight",
            "team_id": event.team_id,
            "zone": event.pitch_zone,
            "highlight_color": color,
            "show_compactness_box": True,
        }
    elif event.category == "positional" and event.player_id is not None:
        return {
            "type": "player_circle",
            "player_id": event.player_id,
            "highlight_color": color,
            "show_zone_overlay": True,
        }
    else:
        return {
            "type": "player_circle",
            "player_id": event.player_id,
            "highlight_color": color,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Relevance scoring
# ──────────────────────────────────────────────────────────────────────────────

def _recency_bonus(event: EventRecord, total_frames: int) -> float:
    """Return 1.0 if event is in last 30% of match, else 0.5."""
    threshold = 0.70 * total_frames
    return 1.0 if event.start_frame >= threshold else 0.5


def _relevance_score(event: EventRecord, total_frames: int) -> float:
    return (
        0.40 * event.importance
        + 0.30 * event.threat_contribution
        + 0.20 * event.confidence
        + 0.10 * _recency_bonus(event, total_frames)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Clip deduplication
# ──────────────────────────────────────────────────────────────────────────────

def _overlap_fraction(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Return what fraction of the shorter clip is covered by the other."""
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    overlap = max(0, hi - lo)
    shorter = min(a[1] - a[0], b[1] - b[0])
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _deduplicate_clips(
    clips: list[tuple[EventRecord, float]],
    merge_threshold: float = THRESHOLDS.CLIP_OVERLAP_MERGE_FRACTION,
) -> list[tuple[EventRecord, float]]:
    """
    Remove clips that overlap with a higher-ranked clip by >= merge_threshold.
    Input is already sorted by relevance descending.
    """
    kept: list[tuple[EventRecord, float]] = []
    for event, score in clips:
        a = (event.clip_start_frame, event.clip_end_frame)
        dominated = False
        for kept_event, _ in kept:
            b = (kept_event.clip_start_frame, kept_event.clip_end_frame)
            if _overlap_fraction(a, b) >= merge_threshold:
                dominated = True
                break
        if not dominated:
            kept.append((event, score))
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# Main retriever class
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceRetriever:
    """
    Retrieves supporting clips for tactical observations from an EventIndex.

    Usage:
        retriever = EvidenceRetriever(index)
        bundle = retriever.retrieve(query)
        retriever.save_bundle(bundle, output_dir)
    """

    def __init__(self, index: EventIndex) -> None:
        self.index = index
        self.total_frames = index.total_frames or 1

    def build_query(
        self,
        observation: str,
        *,
        player_ids: list[int] | None = None,
        team_id: str | None = None,
        time_window: tuple[float, float] | None = None,
        max_results: int = 3,
    ) -> EvidenceQuery:
        """
        Construct an EvidenceQuery from a natural language observation.

        The keyword→event_type mapping from ontology.py is used to extract
        relevant event types automatically.
        """
        event_types = keywords_to_event_types(observation)

        # If no keyword match but player specified, default to all threat events
        if not event_types and player_ids:
            from event_layer.ontology import THREAT_WEIGHTS
            event_types = list(THREAT_WEIGHTS.keys())

        return EvidenceQuery(
            observation_text=observation,
            player_ids=player_ids,
            team_id=team_id,
            event_types=event_types,
            time_window=time_window,
            max_results=max_results,
        )

    def retrieve(self, query: EvidenceQuery) -> EvidenceBundle:
        """
        Execute the evidence retrieval pipeline and return an EvidenceBundle.
        """
        # Step 1: Match events against query filters
        candidates = self._match_events(query)

        if not candidates:
            LOGGER.info("EvidenceRetriever: no candidates matched query: %s", query.observation_text[:80])
            return EvidenceBundle(
                tactical_observation=query.observation_text,
                query_intent="no_match",
                clips=[],
                supporting_event_ids=[],
                generated_at=_now_utc(),
            )

        # Step 2: Score each candidate
        scored = [
            (event, _relevance_score(event, self.total_frames))
            for event in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Step 3: Apply variety penalty if requested
        if query.prefer_variety:
            scored = self._apply_variety_penalty(scored)
            scored.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Deduplicate overlapping clips
        scored = _deduplicate_clips(scored)

        # Step 5: Take top N
        top = scored[: query.max_results]

        # Step 6: Build ClipRecord objects
        clips = [self._build_clip(event, score) for event, score in top]
        supporting_ids = [event.event_id for event, _ in top]

        bundle = EvidenceBundle(
            tactical_observation=query.observation_text,
            query_intent=self._classify_intent(query),
            clips=clips,
            supporting_event_ids=supporting_ids,
            generated_at=_now_utc(),
        )

        LOGGER.info(
            "EvidenceRetriever: returned %d clips for observation: %s",
            len(clips), query.observation_text[:60],
        )
        return bundle

    # ── Step 1: Match ─────────────────────────────────────────────────────────

    def _match_events(self, query: EvidenceQuery) -> list[EventRecord]:
        events = self.index.events

        if query.player_ids is not None:
            events = [e for e in events if e.player_id in query.player_ids]

        if query.team_id is not None:
            events = [e for e in events if e.team_id == query.team_id]

        if query.event_types:
            events = [e for e in events if e.event_type in query.event_types]

        if query.time_window:
            start_s, end_s = query.time_window
            events = [
                e for e in events
                if e.start_time_s >= start_s and e.end_time_s <= end_s
            ]

        events = [e for e in events if e.confidence >= query.min_confidence]
        events = [e for e in events if e.importance >= query.min_importance]

        return events

    # ── Step 3: Variety ───────────────────────────────────────────────────────

    def _apply_variety_penalty(
        self, scored: list[tuple[EventRecord, float]]
    ) -> list[tuple[EventRecord, float]]:
        """
        Penalise events of the same type after the first occurrence.
        2nd+ events of the same event_type get −0.15 on their relevance score.
        """
        type_seen: dict[str, int] = {}
        result = []
        for event, score in scored:
            seen_count = type_seen.get(event.event_type, 0)
            if seen_count > 0:
                score = max(0.0, score - 0.15 * seen_count)
            type_seen[event.event_type] = seen_count + 1
            result.append((event, score))
        return result

    # ── Step 6: Clip building ─────────────────────────────────────────────────

    def _build_clip(self, event: EventRecord, relevance: float) -> ClipRecord:
        annotation = _build_annotation(event)
        highlight_ids = []
        if event.player_id is not None:
            highlight_ids = [event.player_id]

        t_s = event.start_time_s
        label = (
            f"{event.event_name} — "
            f"{int(t_s // 60):02d}:{int(t_s % 60):02d} "
            f"({event.pitch_zone.replace('_', ' ')})"
        )

        return ClipRecord(
            job_id=self.index.job_id,
            event_id=event.event_id,
            start_frame=event.clip_start_frame,
            end_frame=event.clip_end_frame,
            start_time_s=round(event.clip_start_frame / self.index.fps, 2),
            end_time_s=round(event.clip_end_frame / self.index.fps, 2),
            highlight_player_ids=highlight_ids,
            annotation_type=annotation["type"],
            annotation_data=annotation,
            relevance_score=round(relevance, 3),
            label=label,
            confidence_pct=round(event.confidence * 100, 1),
        )

    def _classify_intent(self, query: EvidenceQuery) -> str:
        if query.player_ids:
            return "player_threat_explanation"
        if query.team_id and not query.player_ids:
            return "team_pattern_explanation"
        return "general_observation"


# ──────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def save_evidence_bundle(bundle: EvidenceBundle, output_dir: Path, job_id: str) -> Path:
    """Append an EvidenceBundle to the job's evidence file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{job_id}_evidence_bundles.json"

    existing: list[dict] = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    existing.append(bundle.model_dump())

    with path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    return path


def retrieve_evidence_for_observation(
    observation: str,
    *,
    index: EventIndex,
    player_ids: list[int] | None = None,
    team_id: str | None = None,
    max_results: int = 3,
) -> EvidenceBundle:
    """
    One-shot convenience: build query, retrieve, and return bundle.
    """
    retriever = EvidenceRetriever(index)
    query = retriever.build_query(
        observation,
        player_ids=player_ids,
        team_id=team_id,
        max_results=max_results,
    )
    return retriever.retrieve(query)


def bundle_to_coach_text(bundle: EvidenceBundle) -> str:
    """
    Format an EvidenceBundle as coach-readable text for inclusion in reports.

    Example output:
      Observation: "Player 7 repeatedly exploits space behind the fullback."

      Supporting Evidence (3 clips):
        Clip 1 [34:21 – 34:28] — Dangerous Run — left channel (Confidence: 91%)
        Clip 2 [51:44 – 51:52] — Channel Exploitation — right channel (Confidence: 87%)
        Clip 3 [73:09 – 73:16] — Box Entry — box (Confidence: 79%)
    """
    lines = [
        f'Observation: "{bundle.tactical_observation}"',
        "",
        f"Supporting Evidence ({len(bundle.clips)} clip{'s' if len(bundle.clips) != 1 else ''}):",
    ]

    for i, clip in enumerate(bundle.clips, 1):
        start_m = int(clip.start_time_s // 60)
        start_s = int(clip.start_time_s % 60)
        end_m   = int(clip.end_time_s // 60)
        end_s   = int(clip.end_time_s % 60)
        lines.append(
            f"  Clip {i} [{start_m:02d}:{start_s:02d} – {end_m:02d}:{end_s:02d}] "
            f"— {clip.label} (Confidence: {clip.confidence_pct:.0f}%)"
        )

    return "\n".join(lines)
