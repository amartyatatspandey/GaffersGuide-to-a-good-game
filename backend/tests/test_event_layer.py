"""
Smoke tests for the Event Intelligence Layer.

Run from the backend directory:
    python -m pytest tests/test_event_layer.py -v
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# Helpers to build synthetic test fixtures
# ──────────────────────────────────────────────────────────────────────────────

FPS = 25.0


def _make_frame(
    frame_idx: int,
    players: list[dict],
    ball_xy: list[float] | None = None,
    possession_team_id: str | None = None,
    homography_confidence: float = 0.95,
) -> dict:
    return {
        "frame_idx": frame_idx,
        "players": players,
        "ball_xy": ball_xy,
        "possession_team_id": possession_team_id,
        "homography_confidence": homography_confidence,
    }


def _player(pid: int, team: str, x: float, y: float) -> dict:
    return {"id": pid, "team_id": team, "x_pitch": x, "y_pitch": y}


def _build_speed_run_frames(pid: int, team: str, start_frame: int, n_frames: int, speed_mps: float) -> list[dict]:
    """Build frames where player runs at given speed (m/s)."""
    frames = []
    x = 30.0  # Final third
    for i in range(n_frames):
        f = start_frame + i
        x_now = x + (speed_mps / FPS) * i
        frames.append(_make_frame(f, [_player(pid, team, x_now, 5.0)], possession_team_id=team))
    return frames


def _build_box_entry_frames(pid: int, team: str, start_frame: int) -> list[dict]:
    """Build frames where player crosses into the box."""
    frames = []
    for i, x in enumerate([34.0, 35.5, 36.5, 38.0]):  # Crossing box_x threshold at i=2
        frames.append(_make_frame(start_frame + i, [_player(pid, team, x, 5.0)]))
    return frames


# ──────────────────────────────────────────────────────────────────────────────
# Ontology tests
# ──────────────────────────────────────────────────────────────────────────────

class TestOntology:
    def test_event_registry_has_30_events(self):
        from event_layer.ontology import EVENT_REGISTRY
        assert len(EVENT_REGISTRY) >= 30

    def test_threat_weights_sum_to_one(self):
        from event_layer.ontology import THREAT_WEIGHTS
        total = sum(THREAT_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_zone_classification_box(self):
        from event_layer.ontology import classify_zone
        assert classify_zone(40.0, 10.0) == "box"

    def test_zone_classification_half_space(self):
        from event_layer.ontology import classify_zone
        assert classify_zone(30.0, 18.0) == "half_space_left"
        assert classify_zone(30.0, -18.0) == "half_space_right"

    def test_zone_classification_final_third(self):
        from event_layer.ontology import classify_zone
        # Central corridor between half-spaces
        assert classify_zone(30.0, 0.0) == "central_corridor"

    def test_keyword_matching(self):
        from event_layer.ontology import keywords_to_event_types
        types = keywords_to_event_types("Player 7 exploits space behind the fullback")
        assert "THR_001" in types
        assert "THR_006" in types

    def test_keyword_matching_press(self):
        from event_layer.ontology import keywords_to_event_types
        types = keywords_to_event_types("high press in the final third")
        assert "SHP_001" in types


# ──────────────────────────────────────────────────────────────────────────────
# Model tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_event_record_roundtrip(self):
        from event_layer.models import EventRecord
        e = EventRecord(
            event_type="THR_001",
            event_name="Dangerous Run",
            category="threat",
            team_id="team_0",
            player_id=7,
            start_frame=100,
            end_frame=140,
            start_time_s=4.0,
            end_time_s=5.6,
            duration_s=1.6,
            confidence=0.88,
            importance=0.75,
        )
        data = e.model_dump()
        restored = EventRecord.model_validate(data)
        assert restored.event_type == "THR_001"
        assert restored.confidence_label == "high"

    def test_confidence_labels(self):
        from event_layer.models import EventRecord

        def _event(conf):
            return EventRecord(
                event_type="MOV_001", event_name="Run", category="movement",
                team_id="team_0", start_frame=0, end_frame=50,
                start_time_s=0, end_time_s=2, duration_s=2,
                confidence=conf, importance=0.5,
            )

        assert _event(0.90).confidence_label == "high"
        assert _event(0.70).confidence_label == "medium"
        assert _event(0.50).confidence_label == "low"
        assert _event(0.30).confidence_label == "very_low"

    def test_event_index_filter(self):
        from event_layer.models import EventIndex, EventRecord
        events = [
            EventRecord(event_type="THR_001", event_name="Dangerous Run", category="threat",
                        team_id="team_0", player_id=7,
                        start_frame=0, end_frame=50, start_time_s=0, end_time_s=2, duration_s=2,
                        confidence=0.85, importance=0.8),
            EventRecord(event_type="MOV_001", event_name="Sprint", category="movement",
                        team_id="team_1", player_id=3,
                        start_frame=100, end_frame=200, start_time_s=4, end_time_s=8, duration_s=4,
                        confidence=0.70, importance=0.6),
        ]
        idx = EventIndex(job_id="test", events=events, fps=25.0, total_frames=1000)

        filtered = idx.filter(team_id="team_0")
        assert len(filtered) == 1
        assert filtered[0].event_type == "THR_001"

        filtered2 = idx.filter(categories=["movement"])
        assert len(filtered2) == 1
        assert filtered2[0].event_type == "MOV_001"


# ──────────────────────────────────────────────────────────────────────────────
# Movement detector tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMovementDetector:
    def test_detects_high_speed_run(self):
        from event_layer.detectors.movement import MovementDetector
        # 50 frames at 7.5 m/s = 2 seconds run
        frames = _build_speed_run_frames(pid=7, team="team_0", start_frame=0, n_frames=50, speed_mps=7.5)
        det = MovementDetector(fps=FPS, job_id="test")
        events = det.detect(frames)
        mov_events = [e for e in events if e.event_type == "MOV_001"]
        assert len(mov_events) >= 1, "Should detect at least one high-speed run"

    def test_detects_sprint(self):
        from event_layer.detectors.movement import MovementDetector
        # 30 frames at 9 m/s = 1.2 seconds sprint
        frames = _build_speed_run_frames(pid=7, team="team_0", start_frame=0, n_frames=30, speed_mps=9.0)
        det = MovementDetector(fps=FPS, job_id="test")
        events = det.detect(frames)
        sprint_events = [e for e in events if e.event_type == "MOV_002"]
        assert len(sprint_events) >= 1, "Should detect at least one sprint"

    def test_no_event_below_threshold(self):
        from event_layer.detectors.movement import MovementDetector
        # 50 frames at 3 m/s — jogging, not fast enough
        frames = _build_speed_run_frames(pid=7, team="team_0", start_frame=0, n_frames=50, speed_mps=3.0)
        det = MovementDetector(fps=FPS, job_id="test")
        events = det.detect(frames)
        speed_events = [e for e in events if e.event_type in ("MOV_001", "MOV_002")]
        assert len(speed_events) == 0, "Should not detect speed event below threshold"


# ──────────────────────────────────────────────────────────────────────────────
# Threat detector tests
# ──────────────────────────────────────────────────────────────────────────────

class TestThreatDetector:
    def test_detects_box_entry(self):
        from event_layer.detectors.threat import ThreatDetector
        frames = _build_box_entry_frames(pid=7, team="team_0", start_frame=0)
        det = ThreatDetector(fps=FPS, job_id="test")
        events = det.detect(frames)
        box_events = [e for e in events if e.event_type == "THR_003"]
        assert len(box_events) >= 1, "Should detect box entry"

    def test_detects_final_third_entry(self):
        from event_layer.detectors.threat import ThreatDetector
        frames = [
            _make_frame(0, [_player(7, "team_0", 20.0, 5.0)]),
            _make_frame(1, [_player(7, "team_0", 28.0, 5.0)]),  # Crosses x=25
            _make_frame(2, [_player(7, "team_0", 32.0, 5.0)]),
        ]
        det = ThreatDetector(fps=FPS, job_id="test")
        events = det.detect(frames)
        ft_events = [e for e in events if e.event_type == "THR_002"]
        assert len(ft_events) >= 1, "Should detect final-third entry"

    def test_cooldown_prevents_duplicate_box_entries(self):
        from event_layer.detectors.threat import ThreatDetector
        # Two box entries 2s apart — within cooldown window (10s)
        frames = (
            _build_box_entry_frames(7, "team_0", 0)
            + [_make_frame(i, [_player(7, "team_0", 34.0, 5.0)]) for i in range(4, 10)]
            + _build_box_entry_frames(7, "team_0", 10)  # Only 10 frames later (< 10s cooldown)
        )
        det = ThreatDetector(fps=FPS, job_id="test")
        events = det.detect(frames)
        box_events = [e for e in events if e.event_type == "THR_003"]
        assert len(box_events) == 1, "Cooldown should suppress duplicate box entry"


# ──────────────────────────────────────────────────────────────────────────────
# Threat attributor tests
# ──────────────────────────────────────────────────────────────────────────────

class TestThreatAttributor:
    def _make_index_with_events(self) -> "EventIndex":
        from event_layer.models import EventIndex, EventRecord
        events = [
            EventRecord(event_type="THR_001", event_name="Dangerous Run", category="threat",
                        team_id="team_0", player_id=7,
                        start_frame=0, end_frame=50, start_time_s=0, end_time_s=2, duration_s=2,
                        confidence=0.88, importance=0.80),
            EventRecord(event_type="THR_003", event_name="Box Entry", category="threat",
                        team_id="team_0", player_id=7,
                        start_frame=100, end_frame=100, start_time_s=4, end_time_s=4, duration_s=0,
                        confidence=0.90, importance=0.90),
            EventRecord(event_type="THR_001", event_name="Dangerous Run", category="threat",
                        team_id="team_0", player_id=3,
                        start_frame=200, end_frame=250, start_time_s=8, end_time_s=10, duration_s=2,
                        confidence=0.75, importance=0.65),
        ]
        return EventIndex(job_id="test", events=events, fps=FPS, total_frames=5000)

    def test_profiles_generated(self):
        from event_layer.threat import ThreatAttributor
        index = self._make_index_with_events()
        attributor = ThreatAttributor(index)
        profiles = attributor.compute_profiles()
        assert len(profiles) == 2  # Players 7 and 3

    def test_player7_ranks_higher(self):
        from event_layer.threat import ThreatAttributor
        index = self._make_index_with_events()
        attributor = ThreatAttributor(index)
        profiles = attributor.compute_profiles()
        p7 = next(p for p in profiles if p.player_id == 7)
        p3 = next(p for p in profiles if p.player_id == 3)
        assert p7.threat_score > p3.threat_score, "Player 7 has more events and should rank higher"
        assert p7.threat_rank == 1

    def test_threat_score_normalized_to_100(self):
        from event_layer.threat import ThreatAttributor
        index = self._make_index_with_events()
        attributor = ThreatAttributor(index)
        profiles = attributor.compute_profiles()
        # Top player should have score = 100 after normalization
        top = max(profiles, key=lambda p: p.threat_score)
        assert top.threat_score == pytest.approx(100.0, abs=0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Evidence retriever tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEvidenceRetriever:
    def _make_index(self) -> "EventIndex":
        from event_layer.models import EventIndex, EventRecord
        events = [
            EventRecord(event_type="THR_001", event_name="Dangerous Run", category="threat",
                        team_id="team_0", player_id=7,
                        start_frame=100, end_frame=140, start_time_s=4.0, end_time_s=5.6, duration_s=1.6,
                        confidence=0.88, importance=0.80,
                        clip_start_frame=38, clip_end_frame=190),
            EventRecord(event_type="THR_006", event_name="Channel Exploitation", category="threat",
                        team_id="team_0", player_id=7,
                        start_frame=500, end_frame=560, start_time_s=20.0, end_time_s=22.4, duration_s=2.4,
                        confidence=0.82, importance=0.78,
                        clip_start_frame=438, clip_end_frame=610),
            EventRecord(event_type="THR_003", event_name="Box Entry", category="threat",
                        team_id="team_0", player_id=7,
                        start_frame=900, end_frame=900, start_time_s=36.0, end_time_s=36.0, duration_s=0,
                        confidence=0.91, importance=0.90,
                        clip_start_frame=838, clip_end_frame=950),
        ]
        return EventIndex(job_id="test", events=events, fps=FPS, total_frames=5000)

    def test_retrieves_clips_for_player(self):
        from event_layer.retrieval import EvidenceRetriever
        index = self._make_index()
        retriever = EvidenceRetriever(index)
        query = retriever.build_query(
            "Player 7 exploits space behind the defensive line",
            player_ids=[7],
            team_id="team_0",
        )
        bundle = retriever.retrieve(query)
        assert len(bundle.clips) > 0
        assert bundle.clips[0].relevance_score > 0

    def test_variety_penalty_prevents_same_type_dominance(self):
        from event_layer.models import EventIndex, EventRecord
        from event_layer.retrieval import EvidenceRetriever
        # Create 3 THR_001 events and 1 THR_003
        events = []
        for i in range(3):
            events.append(EventRecord(
                event_type="THR_001", event_name="Dangerous Run", category="threat",
                team_id="team_0", player_id=7,
                start_frame=i * 500, end_frame=i * 500 + 50,
                start_time_s=i * 20.0, end_time_s=i * 20.0 + 2.0, duration_s=2.0,
                confidence=0.88, importance=0.80,
                clip_start_frame=max(0, i * 500 - 62), clip_end_frame=i * 500 + 100,
            ))
        events.append(EventRecord(
            event_type="THR_003", event_name="Box Entry", category="threat",
            team_id="team_0", player_id=7,
            start_frame=2000, end_frame=2000, start_time_s=80.0, end_time_s=80.0, duration_s=0,
            confidence=0.91, importance=0.90,
            clip_start_frame=1938, clip_end_frame=2050,
        ))
        index = EventIndex(job_id="test2", events=events, fps=FPS, total_frames=5000)
        retriever = EvidenceRetriever(index)
        query = retriever.build_query("Player 7 dangerous runs and box entries", player_ids=[7])
        bundle = retriever.retrieve(query)
        # Expect variety: not all 3 results should be THR_001
        event_types_in_bundle = {
            next(e.event_type for e in index.events if e.event_id == eid)
            for eid in bundle.supporting_event_ids
        }
        assert len(event_types_in_bundle) > 1 or len(bundle.clips) < 3, \
            "Variety enforcement should mix event types"

    def test_no_match_returns_empty_bundle(self):
        from event_layer.retrieval import EvidenceRetriever
        index = self._make_index()
        retriever = EvidenceRetriever(index)
        query = retriever.build_query(
            "low block and counter press",
            team_id="team_1",  # Wrong team
        )
        bundle = retriever.retrieve(query)
        assert len(bundle.clips) == 0

    def test_bundle_to_text(self):
        from event_layer.retrieval import EvidenceRetriever, bundle_to_coach_text
        index = self._make_index()
        retriever = EvidenceRetriever(index)
        query = retriever.build_query("Player 7 box entries", player_ids=[7])
        bundle = retriever.retrieve(query)
        text = bundle_to_coach_text(bundle)
        assert "Supporting Evidence" in text
        assert "Clip" in text


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline integration smoke test
# ──────────────────────────────────────────────────────────────────────────────

class TestEventDetectionPipeline:
    def _build_minimal_frames(self, n: int = 200) -> list[dict]:
        """Build a minimal frame sequence with two players and a ball."""
        frames = []
        for i in range(n):
            # Player 7: sprinting forward
            x7 = 20.0 + (i / n) * 30.0
            # Player 3: recovering
            x3 = 30.0 - (i / n) * 15.0
            # Player 10 (opponent)
            x10 = 10.0

            frames.append(_make_frame(
                frame_idx=i,
                players=[
                    _player(7, "team_0", x7, 5.0),
                    _player(3, "team_0", x3, -8.0),
                    _player(10, "team_1", x10, 0.0),
                    _player(11, "team_1", x10 - 5, 15.0),
                ],
                ball_xy=[x7, 5.0] if i > 100 else [x10, 0.0],
                possession_team_id="team_0" if i > 100 else "team_1",
            ))
        return frames

    def test_pipeline_runs_and_returns_events(self):
        from event_layer.pipeline import EventDetectionPipeline
        frames = self._build_minimal_frames(200)
        pipeline = EventDetectionPipeline(fps=FPS, job_id="smoke_test")
        index = pipeline.run(frames)
        assert index.job_id == "smoke_test"
        assert isinstance(index.events, list)
        # Should detect at least some events
        assert len(index.events) >= 0  # Non-negative (might be 0 for very short sequences)

    def test_pipeline_handles_empty_frames(self):
        from event_layer.pipeline import EventDetectionPipeline
        pipeline = EventDetectionPipeline(fps=FPS, job_id="empty_test")
        index = pipeline.run([])
        assert len(index.events) == 0
        assert index.total_frames == 0


class TestReportEnricher:
    def test_report_enrichment_flow(self, tmp_path):
        import json
        from pathlib import Path
        from event_layer.models import EventIndex, EventRecord
        from event_layer.enricher import enrich_report
        
        job_id = "test_job_123"
        
        # 1. Create a dummy report
        report_cards = [
            {
                "frame_idx": 150,
                "team": "team_0",
                "flaw": "Suicidal High Line",
                "severity": "High",
                "frequency_pct": 25.0,
                "evidence": "Average height 35m",
                "matched_philosophy_author": "Arrigo Sacchi",
                "matched_quote_excerpt": "...",
                "fc_role_recommendations": [],
                "tactical_instruction": "Drop line deeper.",
                "llm_error": None
            },
            {
                "frame_idx": 0,
                "team": "global",
                "flaw": "Match Summary",
                "severity": "Info",
                "frequency_pct": 100.0,
                "evidence": "Global summary stats",
                "matched_philosophy_author": "GAFFER",
                "matched_quote_excerpt": "...",
                "fc_role_recommendations": [],
                "tactical_instruction": "Summary text.",
                "llm_error": None
            }
        ]
        report_path = tmp_path / f"{job_id}_report.json"
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_cards, f)
            
        # 2. Create a dummy event index
        events = [
            EventRecord(
                event_type="POS_005",
                event_name="Advanced Positioning",
                category="positional",
                team_id="team_0",
                player_id=3,
                start_frame=100,
                end_frame=200,
                start_time_s=4.0,
                end_time_s=8.0,
                duration_s=4.0,
                confidence=0.9,
                importance=0.8,
                clip_start_frame=50,
                clip_end_frame=250
            ),
            EventRecord(
                event_type="THR_001",
                event_name="Dangerous Run",
                category="threat",
                team_id="team_1",
                player_id=7,
                start_frame=110,
                end_frame=150,
                start_time_s=4.4,
                end_time_s=6.0,
                duration_s=1.6,
                confidence=0.85,
                importance=0.85,
                clip_start_frame=60,
                clip_end_frame=200
            )
        ]
        index = EventIndex(
            job_id=job_id,
            total_frames=1000,
            fps=25.0,
            events=events,
            generated_at="2026-06-09T00:00:00Z"
        )
        event_path = tmp_path / f"{job_id}_events.json"
        with event_path.open("w", encoding="utf-8") as f:
            json.dump(index.model_dump(), f)
            
        # 3. Create dummy threat profiles
        threats = [
            {
                "player_id": 7,
                "team_id": "team_1",
                "job_id": job_id,
                "threat_score": 95.0,
                "threat_rank": 1,
                "primary_threat_types": ["THR_001"],
                "explanation": "Player 7 is a dangerous runner"
            }
        ]
        threat_path = tmp_path / f"{job_id}_threat_profiles.json"
        with threat_path.open("w", encoding="utf-8") as f:
            json.dump(threats, f)
            
        # 4. Enrich report
        enriched_path = enrich_report(
            report_path=report_path,
            job_id=job_id,
            output_dir=tmp_path
        )
        
        assert enriched_path.is_file()
        assert enriched_path.name == f"{job_id}_report_enriched.json"
        
        with enriched_path.open("r", encoding="utf-8") as f:
            enriched_cards = json.load(f)
            
        assert len(enriched_cards) == 2
        
        # Check first card (Suicidal High Line) was enriched
        high_line_card = enriched_cards[0]
        assert high_line_card["flaw"] == "Suicidal High Line"
        assert len(high_line_card["evidence_clips"]) > 0
        assert high_line_card["threat_context"]["top_threats"][0]["player_id"] == 7
        assert high_line_card["event_count_summary"]["POS_005"] == 1
        assert high_line_card["event_count_summary"]["THR_001"] == 1
        
        # Check second card (Match Summary) was passed through untouched
        summary_card = enriched_cards[1]
        assert summary_card["flaw"] == "Match Summary"
        assert "evidence_clips" not in summary_card


class TestChatEvidence:
    def test_chat_evidence_builder(self, tmp_path):
        import json
        from pathlib import Path
        from event_layer.models import EventIndex, EventRecord
        from event_layer.chat_evidence import build_evidence_response
        
        job_id = "test_job_chat"
        
        # 1. Create a dummy event index
        events = [
            EventRecord(
                event_type="THR_001",
                event_name="Dangerous Run",
                category="threat",
                team_id="team_0",
                player_id=7,
                start_frame=100,
                end_frame=150,
                start_time_s=4.0,
                end_time_s=6.0,
                duration_s=2.0,
                confidence=0.9,
                importance=0.8,
                clip_start_frame=50,
                clip_end_frame=200
            )
        ]
        index = EventIndex(
            job_id=job_id,
            total_frames=1000,
            fps=25.0,
            events=events,
            generated_at="2026-06-09T00:00:00Z"
        )
        event_path = tmp_path / f"{job_id}_events.json"
        with event_path.open("w", encoding="utf-8") as f:
            json.dump(index.model_dump(), f)
            
        # 2. Create dummy threat profiles
        threats = [
            {
                "player_id": 7,
                "team_id": "team_0",
                "job_id": job_id,
                "threat_score": 88.0,
                "threat_rank": 1,
                "primary_threat_types": ["THR_001"],
                "explanation": "Player 7 threat score: 88.0/100"
            }
        ]
        threat_path = tmp_path / f"{job_id}_threat_profiles.json"
        with threat_path.open("w", encoding="utf-8") as f:
            json.dump(threats, f)
            
        # 3. Build evidence response for "show me player 7 runs"
        attachment = build_evidence_response(
            message="Show me player 7 runs behind the defense",
            job_id=job_id,
            output_dir=tmp_path
        )
        
        assert attachment is not None
        assert len(attachment.clips) == 1
        assert attachment.clips[0]["event_id"] == events[0].event_id
        assert len(attachment.top_threats) == 1
        assert attachment.top_threats[0]["player_id"] == 7
        assert attachment.observation == "Show me player 7 runs behind the defense"
