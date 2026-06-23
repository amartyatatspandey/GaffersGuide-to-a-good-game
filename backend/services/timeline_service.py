import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from scripts.tactical_rule_engine import RuleEngine

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BACKEND_ROOT / "output"
TACTICS_JSON_PATH = BACKEND_ROOT / "data" / "zsl_tactics.json"

class TimelineService:
    @staticmethod
    def load_philosophies() -> List[Dict[str, Any]]:
        """Loads philosophical guidelines from zsl_tactics.json."""
        if not TACTICS_JSON_PATH.is_file():
            return []
        try:
            with open(TACTICS_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("philosophies", [])
        except Exception as e:
            LOGGER.error("Failed to load philosophies: %s", e)
            return []

    @classmethod
    def get_philosophy_for_phase(cls, phase: str, philosophies: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Finds a matching philosophy quote based on the classified tactical phase."""
        phase_lower = phase.lower()
        
        # Map phase labels to potential tags in zsl_tactics.json
        tag_mappings = {
            "high press": ["high press", "gegenpress"],
            "low block": ["low block", "parked bus", "compact defense"],
            "mid block": ["mid block", "press traps"],
            "possession dominance": ["positional play", "build-up", "tiki-taka", "total football"],
            "counter attack": ["counter-attack", "transition speed", "direct verticality"]
        }
        
        target_tags = tag_mappings.get(phase_lower, [phase_lower])
        
        for phil in philosophies:
            tags = [t.lower() for t in phil.get("tags", [])]
            for t in target_tags:
                if t in tags or any(t in existing_tag for existing_tag in tags):
                    return phil
        return None

    @classmethod
    def generate_timeline(cls, job_id: str) -> List[Dict[str, Any]]:
        """
        Slices the match frame-by-frame metrics and events into adaptive segments,
        calculates average KPIs, classifies tactical phases, and maps clips.
        """
        metrics_path = OUTPUT_DIR / f"{job_id}_tactical_metrics.json"
        events_path = OUTPUT_DIR / f"{job_id}_events.json"

        if not metrics_path.is_file():
            LOGGER.warning("Metrics timeline file not found for job: %s", job_id)
            return []

        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                frames: List[Dict[str, Any]] = json.load(f)
        except Exception as e:
            LOGGER.error("Failed to read metrics file for job %s: %s", job_id, e)
            return []

        if not frames:
            return []

        # Load events if available
        events = []
        if events_path.is_file():
            try:
                with open(events_path, "r", encoding="utf-8") as f:
                    event_data = json.load(f)
                    events = event_data.get("events", [])
            except Exception as e:
                LOGGER.error("Failed to read events file for job %s: %s", job_id, e)

        # 1. Determine timeline bounds & adaptive segment size
        max_frame = max(f.get("frame_idx", 0) for f in frames)
        duration_s = max_frame / 25.0
        
        # Determine segment size in seconds
        if duration_s < 180:        # < 3 min (Short test clip)
            segment_duration = 30   # 30-second segments
        elif duration_s < 600:      # < 10 min
            segment_duration = 60   # 1-minute segments
        elif duration_s < 1800:     # < 30 min
            segment_duration = 300  # 5-minute segments
        else:
            segment_duration = 900  # 15-minute segments

        philosophies = cls.load_philosophies()
        engine = RuleEngine()
        segments = []
        
        num_segments = int(duration_s / segment_duration) + (1 if duration_s % segment_duration > 0 else 0)
        
        for i in range(num_segments):
            start_s = i * segment_duration
            end_s = min((i + 1) * segment_duration, duration_s)
            
            start_frame = int(start_s * 25)
            end_frame = int(end_s * 25)
            
            # Slice frames in this segment
            seg_frames = [f for f in frames if start_frame <= f.get("frame_idx", 0) < end_frame]
            if not seg_frames:
                continue

            # Calculate Segment metrics via RuleEngine consistency
            summary = engine.calculate_tactical_scores(seg_frames)
            
            # Helper to calculate average metrics for classification
            def get_avg_team_metrics(tid: str) -> Dict[str, float]:
                team_frames = [f.get(tid, {}) for f in seg_frames if tid in f]
                if not team_frames:
                    return {}
                return {
                    "deepest_x": sum(float(f.get("deepest_x", 0.0) or 0.0) for f in team_frames) / len(team_frames),
                    "pitch_control_pct": sum(float(f.get("pitch_control_pct", 50.0) or 50.0) for f in team_frames) / len(team_frames),
                    "avg_speed_kmh": sum(float(f.get("avg_speed_kmh", 0.0) or 0.0) for f in team_frames) / len(team_frames),
                    "centroid_x": sum(f.get("centroid", [0.0])[0] for f in team_frames) / len(team_frames)
                }

            t0_avg = get_avg_team_metrics("team_0")
            t1_avg = get_avg_team_metrics("team_1")
            
            # Heuristic Tactical Phase Classifier
            def classify_phase(avg: Dict[str, float], engine_score: Dict[str, Any]) -> str:
                if not avg:
                    return "Mid Block"
                
                # Counter Attack: High player speed + high transition velocity
                if avg.get("avg_speed_kmh", 0.0) > 18.0 or engine_score.get("transition_speed", 0.0) > 75.0:
                    return "Counter Attack"
                
                # Possession Dominance: high control and pushed forward
                if avg.get("pitch_control_pct", 0.0) > 55.0 and avg.get("centroid_x", 0.0) > 0.0:
                    return "Possession Dominance"
                
                # Defensive lines
                deep_x = avg.get("deepest_x", 0.0)
                if deep_x < -15.0:
                    return "Low Block"
                elif deep_x > 10.0:
                    return "High Press"
                else:
                    return "Mid Block"

            t0_phase = classify_phase(t0_avg, summary.get("team_0", {}))
            t1_phase = classify_phase(t1_avg, summary.get("team_1", {}))
            
            t0_phil = cls.get_philosophy_for_phase(t0_phase, philosophies)
            t1_phil = cls.get_philosophy_for_phase(t1_phase, philosophies)
            
            # Label formatter
            start_m = int(start_s / 60)
            end_m = int(end_s / 60)
            label = f"{start_m}-{end_m} min" if start_m != end_m else f"{start_m} min"
            
            # Format segment verbal explanation
            def make_explanation(team_name: str, phase: str, avg: Dict[str, float], score: Dict[str, Any]) -> str:
                if not avg:
                    return f"{team_name} played in a structured shape."
                return (
                    f"Aligned in a {phase} style (Tactical Power: {score.get('tactical_power', 50.0):.1f}). "
                    f"Maintained an average defensive line at {avg.get('deepest_x', 0.0):.1f}m "
                    f"and controlled {avg.get('pitch_control_pct', 50.0):.0f}% of the territory."
                )

            # Filter events belonging to this segment
            segment_events = []
            for ev in events:
                evt_start = ev.get("start_time_s", 0.0)
                if start_s <= evt_start < end_s:
                    segment_events.append({
                        "event_id": ev.get("event_id"),
                        "event_name": ev.get("event_name"),
                        "player_id": ev.get("player_id"),
                        "team_id": ev.get("team_id"),
                        "start_time_s": ev.get("start_time_s"),
                        "end_time_s": ev.get("end_time_s"),
                        "description": ev.get("description"),
                        "confidence_pct": int(ev.get("confidence", 0.8) * 100),
                        "importance": ev.get("importance", 0.5)
                    })
            
            # Sort events by importance descending and take top 5
            segment_events.sort(key=lambda e: e["importance"], reverse=True)
            key_events = segment_events[:5]

            segments.append({
                "segment_idx": i,
                "label": label,
                "start_time_s": start_s,
                "end_time_s": end_s,
                "team_0": {
                  "phase": t0_phase,
                  "philosophy_quote": t0_phil.get("quote") if t0_phil else "No quote available.",
                  "philosophy_author": t0_phil.get("author") if t0_phil else "Tactical Engine",
                  "explanation": make_explanation("Red Team", t0_phase, t0_avg, summary.get("team_0", {})),
                  "metrics": {
                      "tactical_power": summary.get("team_0", {}).get("tactical_power", 50.0),
                      "compactness": summary.get("team_0", {}).get("compactness", 50.0),
                      "transition_speed": summary.get("team_0", {}).get("transition_speed", 50.0),
                      "defensive_shape": summary.get("team_0", {}).get("defensive_shape", 50.0)
                  }
                },
                "team_1": {
                  "phase": t1_phase,
                  "philosophy_quote": t1_phil.get("quote") if t1_phil else "No quote available.",
                  "philosophy_author": t1_phil.get("author") if t1_phil else "Tactical Engine",
                  "explanation": make_explanation("Blue Team", t1_phase, t1_avg, summary.get("team_1", {})),
                  "metrics": {
                      "tactical_power": summary.get("team_1", {}).get("tactical_power", 50.0),
                      "compactness": summary.get("team_1", {}).get("compactness", 50.0),
                      "transition_speed": summary.get("team_1", {}).get("transition_speed", 50.0),
                      "defensive_shape": summary.get("team_1", {}).get("defensive_shape", 50.0)
                  }
                },
                "key_events": key_events
            })
            
        return segments
