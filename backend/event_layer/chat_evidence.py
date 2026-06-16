"""
Event Intelligence Layer — Evidence-Aware Chat Helper
======================================================

Parses natural language chat queries to extract entities (players, teams),
queries the EventIndex to retrieve supporting clips, and fetches player
threat profiles to attach visual evidence to chat responses.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from event_layer.pipeline import load_event_index
from event_layer.retrieval import EvidenceRetriever
from models import EvidenceAttachment

LOGGER = logging.getLogger(__name__)


def build_evidence_response(
    message: str,
    job_id: str,
    output_dir: Path,
) -> EvidenceAttachment | None:
    """
    Analyzes chat request, queries event index, retrieves matching clips and threat profiles,
    and returns a structured EvidenceAttachment.
    """
    event_path = output_dir / f"{job_id}_events.json"
    if not event_path.is_file():
        LOGGER.warning("Event index not found for job %s at %s", job_id, event_path)
        return None

    try:
        index = load_event_index(event_path)
    except Exception as exc:
        LOGGER.error("Failed to load event index for job %s: %s", job_id, exc)
        return None

    # 1. Parse player IDs (e.g. "player 7", "p7", "no. 10", "number 3")
    pids = [int(x) for x in re.findall(r'\b(?:player|player_id|number|no\.?|p)\s*(\d+)\b', message, re.IGNORECASE)]
    if not pids:
        # Fallback to standalone numbers that might indicate player IDs
        candidates = [int(x) for x in re.findall(r'\b([1-9][0-9]?)\b', message)]
        pids = [c for c in candidates if 1 <= c <= 99]

    # 2. Parse team ID
    team_id = None
    msg_lower = message.lower()
    if "team_0" in msg_lower or "team 0" in msg_lower or "red" in msg_lower:
        team_id = "team_0"
    elif "team_1" in msg_lower or "team 1" in msg_lower or "blue" in msg_lower:
        team_id = "team_1"

    # 3. Retrieve clips
    retriever = EvidenceRetriever(index)
    query = retriever.build_query(
        observation=message,
        player_ids=pids if pids else None,
        team_id=team_id,
        max_results=3,
    )
    bundle = retriever.retrieve(query)
    clips_dump = [clip.model_dump() for clip in bundle.clips]

    # 4. Load Threat Profiles
    threat_path = output_dir / f"{job_id}_threat_profiles.json"
    threat_profiles: list[dict[str, Any]] = []
    if threat_path.is_file():
        try:
            with threat_path.open("r", encoding="utf-8") as f:
                threat_profiles = json.load(f)
        except Exception as exc:
            LOGGER.warning("Failed to load threat profiles: %s", exc)

    # 5. Filter threat profiles based on query context
    if pids:
        matching_threats = [p for p in threat_profiles if p.get("player_id") in pids]
    elif team_id:
        matching_threats = [p for p in threat_profiles if p.get("team_id") == team_id]
    else:
        matching_threats = threat_profiles

    # Sort by threat score descending and take top 3
    matching_threats = sorted(
        matching_threats,
        key=lambda p: p.get("threat_score", 0.0),
        reverse=True,
    )[:3]

    top_threats_dump = []
    for profile in matching_threats:
        top_threats_dump.append({
            "player_id": profile["player_id"],
            "team_id": profile["team_id"],
            "threat_score": profile["threat_score"],
            "primary_threat_types": profile.get("primary_threat_types", []),
            "explanation": profile.get("explanation", ""),
        })

    return EvidenceAttachment(
        clips=clips_dump,
        top_threats=top_threats_dump,
        observation=message,
    )
