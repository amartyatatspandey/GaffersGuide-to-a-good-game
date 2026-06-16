"""
Event Intelligence Layer — Report Enricher
==========================================

Connects the EventIndex and ThreatProfiles to the coaching cards in the LLM report.
Produces a new `{job_id}_report_enriched.json` artifact containing evidence clips,
threat context, and event frequency counts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from event_layer.models import EventIndex, ClipRecord
from event_layer.pipeline import load_event_index
from event_layer.retrieval import EvidenceRetriever
from event_layer.ontology import keywords_to_event_types
from models import EnrichedCoachingCard

LOGGER = logging.getLogger(__name__)

# Map tactical rule engine flaws to their corresponding event type codes
FLAW_TO_EVENT_TYPES = {
    "Stretched Defensive Shape": ["SHP_005"],
    "High Line Exposure": ["POS_005", "THR_006", "THR_001"],
    "High Defensive Line": ["POS_005", "THR_006", "THR_001"],
    "Suicidal High Line": ["POS_005", "THR_006", "THR_001"],
    "Poor Press Coordination": ["TRN_004", "SHP_001"],
    "Press Failed": ["TRN_004", "SHP_001"],
    "Lethargic Press": ["TRN_004", "SHP_001"],
    "Large Defensive Line Gap": ["SHP_005", "SHP_002"],
    "Midfield Disconnect": ["SHP_002", "SHP_005"],
    "Counter-Attack Vulnerability": ["TRN_005", "SHP_008"],
    "Low Compactness": ["SHP_004", "SHP_002"],
    "Parked Bus": ["SHP_004", "SHP_002"],
    "Half-Space Exploitation": ["POS_002", "THR_006"],
    "Poor Transition Recovery": ["MOV_003", "TRN_001"],
    "Over-Stretched Formation": ["SHP_005"],
}


def deduplicate_clips(clips: list[ClipRecord], merge_threshold: float = 0.70) -> list[ClipRecord]:
    """Remove clips that overlap with higher-relevance clips by >= merge_threshold."""
    kept: list[ClipRecord] = []
    sorted_clips = sorted(clips, key=lambda c: c.relevance_score, reverse=True)
    for clip in sorted_clips:
        a = (clip.start_frame, clip.end_frame)
        dominated = False
        for kept_clip in kept:
            b = (kept_clip.start_frame, kept_clip.end_frame)
            lo = max(a[0], b[0])
            hi = min(a[1], b[1])
            overlap = max(0, hi - lo)
            shorter = min(a[1] - a[0], b[1] - b[0])
            overlap_fraction = overlap / shorter if shorter > 0 else 0.0
            if overlap_fraction >= merge_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(clip)
    return kept


def enrich_report(report_path: Path, job_id: str, output_dir: Path) -> Path:
    """
    Enriches the standard tactical report with evidence clips and threat context.

    Saves `{job_id}_report_enriched.json` in the output directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    enriched_report_path = output_dir / f"{job_id}_report_enriched.json"

    # 1. Load the original report
    if not report_path.is_file():
        raise FileNotFoundError(f"Original report not found at {report_path}")

    with report_path.open("r", encoding="utf-8") as f:
        original_cards: list[dict[str, Any]] = json.load(f)

    # 2. Load EventIndex
    event_path = output_dir / f"{job_id}_events.json"
    if not event_path.is_file():
        raise FileNotFoundError(f"Event Index not found at {event_path}")
    index = load_event_index(event_path)
    retriever = EvidenceRetriever(index)

    # 3. Load Threat Profiles
    threat_path = output_dir / f"{job_id}_threat_profiles.json"
    threat_profiles: list[dict[str, Any]] = []
    if threat_path.is_file():
        try:
            with threat_path.open("r", encoding="utf-8") as f:
                threat_profiles = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load threat profiles: %s", exc)

    enriched_cards: list[dict[str, Any]] = []

    for card in original_cards:
        flaw = card.get("flaw", "")
        evidence_text = card.get("evidence", "")
        card_team = card.get("team", "global")

        # Skip enrichment for Match Summary card, but pass it through
        if flaw == "Match Summary":
            enriched_cards.append(card)
            continue

        # Map flaw to event types
        event_types = FLAW_TO_EVENT_TYPES.get(flaw, [])
        if not event_types:
            event_types = keywords_to_event_types(evidence_text)
            if not event_types:
                event_types = keywords_to_event_types(flaw)

        # Set up teams
        own_team = card_team
        opp_team = "team_1" if card_team == "team_0" else "team_0" if card_team == "team_1" else None

        own_team_events = []
        opp_team_events = []

        for et in event_types:
            # Threat, transition, and movement events are checked against the opponent team
            if et.startswith("THR_") or et.startswith("MOV_") or et in ("TRN_004", "TRN_005"):
                if opp_team:
                    opp_team_events.append(et)
                else:
                    own_team_events.append(et)
            else:
                own_team_events.append(et)

        # Retrieve candidate clips
        clips: list[ClipRecord] = []
        if own_team_events:
            q_own = retriever.build_query(
                observation=evidence_text,
                team_id=own_team,
                max_results=3,
            )
            q_own.event_types = own_team_events
            bundle_own = retriever.retrieve(q_own)
            clips.extend(bundle_own.clips)

        if opp_team_events:
            q_opp = retriever.build_query(
                observation=evidence_text,
                team_id=opp_team,
                max_results=3,
            )
            q_opp.event_types = opp_team_events
            bundle_opp = retriever.retrieve(q_opp)
            clips.extend(bundle_opp.clips)

        # Deduplicate and cap at top 3 clips
        final_clips = deduplicate_clips(clips)[:3]

        # Calculate event counts summary
        event_count_summary: dict[str, int] = {}
        for et in event_types:
            if et in own_team_events:
                cnt = len(index.filter(event_types=[et], team_id=own_team))
            elif et in opp_team_events:
                cnt = len(index.filter(event_types=[et], team_id=opp_team))
            else:
                cnt = len(index.filter(event_types=[et]))
            if cnt > 0:
                event_count_summary[et] = cnt

        # Fetch threat context for opponent team
        top_threats = []
        target_threat_team = opp_team or "team_1"
        sorted_threats = sorted(
            [p for p in threat_profiles if p.get("team_id") == target_threat_team],
            key=lambda p: p.get("threat_score", 0),
            reverse=True,
        )
        for profile in sorted_threats[:3]:
            top_threats.append({
                "player_id": profile["player_id"],
                "team_id": profile["team_id"],
                "threat_score": profile["threat_score"],
                "primary_threat_types": profile.get("primary_threat_types", []),
                "explanation": profile.get("explanation", ""),
            })

        threat_context = {"top_threats": top_threats}

        # Build enriched card dict
        enriched_card_dict = {
            **card,
            "job_id": job_id,
            "evidence_clips": [clip.model_dump() for clip in final_clips],
            "threat_context": threat_context,
            "event_count_summary": event_count_summary,
        }

        # Validate with Pydantic
        try:
            validated = EnrichedCoachingCard.model_validate(enriched_card_dict)
            enriched_cards.append(validated.model_dump())
        except Exception as exc:
            LOGGER.error("Enriched card validation failed for flaw %r: %s", flaw, exc)
            # Fall back to raw dictionary if validation fails (but log error)
            enriched_cards.append(enriched_card_dict)

    with enriched_report_path.open("w", encoding="utf-8") as f:
        json.dump(enriched_cards, f, indent=2, ensure_ascii=False)

    LOGGER.info("Enriched report saved to %s", enriched_report_path)
    return enriched_report_path
