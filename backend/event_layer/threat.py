"""
Event Intelligence Layer — Threat Attributor
=============================================

Computes per-player threat scores, ranks players, generates explanations,
and persists PlayerThreatProfile objects to disk.

Threat Score Formula
--------------------
raw_score = Σ (event_count × weight × mean_importance × temporal_recency_factor)
normalized_score = 100 × (raw_score / max_raw_score_in_team)

Temporal Recency: events in the last 25% of total frames are weighted 1.5×.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from event_layer.models import EventIndex, EventRecord, PlayerThreatProfile
from event_layer.ontology import (
    THREAT_WEIGHTS,
    TRANSITION_ONLY_PENALTY,
    MIN_THREAT_SCORE_FOR_REPORT,
)

LOGGER = logging.getLogger(__name__)

# Event types that count toward threat scoring (threat category events only)
THREAT_EVENT_TYPES = set(THREAT_WEIGHTS.keys())

# Template descriptions for dominant event types
_DOMINANT_TEMPLATES: dict[str, str] = {
    "THR_001": "Player {pid} repeatedly finds space behind the defensive line through timing and movement.",
    "THR_003": "Player {pid} consistently reaches the box, indicating a structured attacking run pattern.",
    "THR_006": "Player {pid} is exploiting the channel between the centre-back and full-back.",
    "THR_007": "Player {pid} is winning 1v1 duels in wide areas, creating crossing and cutting opportunities.",
    "THR_002": "Player {pid} frequently penetrates the final third, maintaining constant forward pressure.",
    "THR_005": "Player {pid} receives in dangerous areas and creates problems when in possession.",
    "THR_004": "Player {pid} is highly involved in transitions, participating in most attacking sequences.",
}

# Human names for event types used in explanation bullets
_EVENT_NAMES: dict[str, str] = {
    "THR_001": "Dangerous Runs",
    "THR_002": "Final-Third Entries",
    "THR_003": "Box Entries",
    "THR_004": "Transition Involvements",
    "THR_005": "Dangerous Receptions",
    "THR_006": "Channel Exploitations",
    "THR_007": "Isolated Defender Exploits",
}


class ThreatAttributor:
    """
    Computes threat scores and ranks players for both teams.

    Usage:
        attributor = ThreatAttributor(index)
        profiles = attributor.compute_profiles()
        attributor.save(profiles, output_path)
    """

    def __init__(self, index: EventIndex) -> None:
        self.index = index
        self.total_frames = index.total_frames or 1
        self.fps = index.fps or 25.0

    def compute_profiles(self) -> list[PlayerThreatProfile]:
        """
        Compute a PlayerThreatProfile for every player who has at least
        one threat event. Returns all profiles, unsuppressed.
        """
        # Collect threat events per player
        threat_events_by_player: dict[int, list[EventRecord]] = defaultdict(list)
        team_map: dict[int, str] = {}

        for event in self.index.events:
            if event.event_type not in THREAT_EVENT_TYPES:
                continue
            if event.player_id is None:
                continue
            threat_events_by_player[event.player_id].append(event)
            if event.player_id not in team_map:
                team_map[event.player_id] = event.team_id

        if not threat_events_by_player:
            return []

        # Compute raw scores per player
        raw_scores: dict[int, float] = {}
        for pid, events in threat_events_by_player.items():
            raw_scores[pid] = self._compute_raw_score(pid, events)

        # Apply transition-only penalty
        for pid, events in threat_events_by_player.items():
            non_transition = [e for e in events if e.event_type != "THR_004"]
            if not non_transition:
                raw_scores[pid] *= TRANSITION_ONLY_PENALTY

        # Normalize per team
        team_players: dict[str, list[int]] = defaultdict(list)
        for pid, tid in team_map.items():
            team_players[tid].append(pid)

        profiles: list[PlayerThreatProfile] = []
        for team_id, pids in team_players.items():
            team_raw = {pid: raw_scores[pid] for pid in pids}
            max_raw = max(team_raw.values()) if team_raw else 1.0
            max_raw = max(max_raw, 1e-9)

            normalized = {
                pid: round(100.0 * raw / max_raw, 1)
                for pid, raw in team_raw.items()
            }

            # Rank players (1 = highest threat)
            ranked = sorted(pids, key=lambda p: normalized[p], reverse=True)

            for rank, pid in enumerate(ranked, start=1):
                events = threat_events_by_player[pid]
                profile = self._build_profile(
                    pid=pid,
                    team_id=team_id,
                    events=events,
                    threat_score=normalized[pid],
                    rank=rank,
                )
                profiles.append(profile)

        # Set threat_contribution back on the index events
        self._backfill_threat_contributions(profiles)

        return profiles

    def _compute_raw_score(self, pid: int, events: list[EventRecord]) -> float:
        """Compute the weighted raw threat score for a player."""
        # Late-match recency threshold: last 25% of frames
        late_start = int(0.75 * self.total_frames)

        score = 0.0
        for event_type, weight in THREAT_WEIGHTS.items():
            type_events = [e for e in events if e.event_type == event_type]
            if not type_events:
                continue

            count = len(type_events)
            mean_importance = sum(e.importance for e in type_events) / count

            # Temporal recency: events in last 25% get 1.5× weight
            late_count = sum(1 for e in type_events if e.start_frame >= late_start)
            early_count = count - late_count
            recency_factor = (late_count * 1.5 + early_count * 1.0) / count

            score += count * weight * mean_importance * recency_factor

        return score

    def _build_profile(
        self,
        pid: int,
        team_id: str,
        events: list[EventRecord],
        threat_score: float,
        rank: int,
    ) -> PlayerThreatProfile:
        # Event counts and IDs
        event_counts: dict[str, int] = defaultdict(int)
        event_ids: dict[str, list[str]] = defaultdict(list)

        for event in events:
            event_counts[event.event_type] += 1
            event_ids[event.event_type].append(event.event_id)

        # Top 3 contributing event types (by weighted contribution)
        contributions = {
            et: event_counts[et] * THREAT_WEIGHTS.get(et, 0)
            for et in event_counts
        }
        top_types = sorted(contributions, key=contributions.get, reverse=True)[:3]  # type: ignore[arg-type]

        explanation = self._generate_explanation(pid, team_id, threat_score, event_counts, top_types)

        return PlayerThreatProfile(
            player_id=pid,
            team_id=team_id,
            job_id=self.index.job_id,
            event_counts=dict(event_counts),
            event_ids=dict(event_ids),
            threat_score=threat_score,
            threat_rank=rank,
            primary_threat_types=top_types,
            explanation=explanation,
        )

    def _generate_explanation(
        self,
        pid: int,
        team_id: str,
        threat_score: float,
        event_counts: dict[str, int],
        top_types: list[str],
    ) -> str:
        """Generate a structured 3-part explanation for a player threat profile."""
        # Part 1: Verdict
        verdict = (
            f"Player {pid} ({team_id}) threat score: {threat_score:.1f}/100."
        )

        # Part 2: Evidence bullets
        bullets = []
        for et in top_types:
            count = event_counts.get(et, 0)
            name = _EVENT_NAMES.get(et, et)
            bullets.append(f"  • {count}× {name}")
        evidence = "\n".join(bullets)

        # Part 3: Tactical pattern (from dominant event type)
        dominant = top_types[0] if top_types else None
        pattern = ""
        if dominant and dominant in _DOMINANT_TEMPLATES:
            pattern = _DOMINANT_TEMPLATES[dominant].format(pid=pid)

        parts = [verdict]
        if evidence:
            parts.append(f"Primary contributions:\n{evidence}")
        if pattern:
            parts.append(pattern)

        return "\n\n".join(parts)

    def _backfill_threat_contributions(self, profiles: list[PlayerThreatProfile]) -> None:
        """Set threat_contribution on matching EventRecord objects in the index."""
        # Map player_id → normalized score (0-1)
        score_map: dict[int, float] = {
            p.player_id: p.threat_score / 100.0 for p in profiles
        }
        for event in self.index.events:
            if event.player_id in score_map:
                event.threat_contribution = round(score_map[event.player_id], 3)

    def top_threats(
        self,
        profiles: list[PlayerThreatProfile],
        *,
        team_id: str | None = None,
        min_score: float = MIN_THREAT_SCORE_FOR_REPORT,
        top_n: int = 3,
    ) -> list[PlayerThreatProfile]:
        """Return the top-N threat players, optionally filtered by team."""
        filtered = [p for p in profiles if p.threat_score >= min_score]
        if team_id:
            filtered = [p for p in filtered if p.team_id == team_id]
        filtered.sort(key=lambda p: p.threat_score, reverse=True)
        return filtered[:top_n]


def run_threat_attribution(
    index: EventIndex,
    *,
    output_dir: Path,
) -> tuple[list[PlayerThreatProfile], Path]:
    """
    Convenience function: compute profiles and write to disk.

    Returns (profiles, output_path).
    """
    attributor = ThreatAttributor(index)
    profiles = attributor.compute_profiles()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{index.job_id}_threat_profiles.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in profiles], f, indent=2, ensure_ascii=False)

    LOGGER.info(
        "ThreatAttributor: %d profiles written to %s",
        len(profiles), out_path,
    )
    return profiles, out_path


def load_threat_profiles(path: Path) -> list[PlayerThreatProfile]:
    """Load previously-saved threat profiles from disk."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [PlayerThreatProfile.model_validate(row) for row in data]
