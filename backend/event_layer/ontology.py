"""
Event Intelligence Layer — Ontology
=====================================

Defines all 30 V1 event types, zone boundaries, threat signal weights,
and keyword→event_type mappings for evidence retrieval.

Coordinate system (matches parallel_pipeline.py / TacticalRadar):
  - Origin: pitch centre circle
  - X: positive toward attacking goal
  - Y: positive toward right touchline
  - Pitch: 105m × 68m  →  X in [-52.5, 52.5], Y in [-34, 34]
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Event type registry
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EventTypeSpec:
    code: str          # Ontology code, e.g. "THR_001"
    name: str          # Human name
    category: str      # movement | positional | threat | shape | transition
    description: str   # One-line football meaning


EVENT_REGISTRY: dict[str, EventTypeSpec] = {
    # ── Movement ──────────────────────────────────────────────────────────────
    "MOV_001": EventTypeSpec("MOV_001", "High-Speed Run",   "movement",   "Sustained burst ≥ 6.5 m/s for ≥ 1.5 seconds"),
    "MOV_002": EventTypeSpec("MOV_002", "Sprint",           "movement",   "Peak velocity ≥ 8.0 m/s for ≥ 0.5 seconds"),
    "MOV_003": EventTypeSpec("MOV_003", "Recovery Run",     "movement",   "High-speed movement toward own goal after losing possession"),
    "MOV_004": EventTypeSpec("MOV_004", "Overlap Run",      "movement",   "Wide player moves beyond attacker along touchline corridor"),
    "MOV_005": EventTypeSpec("MOV_005", "Underlap Run",     "movement",   "Inverted player cuts inside into half-space"),
    "MOV_006": EventTypeSpec("MOV_006", "Third-Man Run",    "movement",   "Off-ball run timed to arrive after a combination"),
    "MOV_007": EventTypeSpec("MOV_007", "Diagonal Run",     "movement",   "Run at ≥ 30° diagonal angle across defensive lines"),

    # ── Positional ────────────────────────────────────────────────────────────
    "POS_001": EventTypeSpec("POS_001", "Wide Positioning",         "positional", "Player sustained in outer 15% of pitch width"),
    "POS_002": EventTypeSpec("POS_002", "Half-Space Occupation",    "positional", "Player holds position in half-space in final third"),
    "POS_003": EventTypeSpec("POS_003", "Between-Lines",            "positional", "Player positioned between opponent defensive and midfield lines"),
    "POS_004": EventTypeSpec("POS_004", "Deep Positioning",         "positional", "Attacker retreats to own half to receive"),
    "POS_005": EventTypeSpec("POS_005", "Advanced Positioning",     "positional", "Defender pushes beyond centre — high line indicator"),
    "POS_006": EventTypeSpec("POS_006", "Pressing Trap Position",   "positional", "Player holds forced-corridor position in coordinated press"),

    # ── Threat ────────────────────────────────────────────────────────────────
    "THR_001": EventTypeSpec("THR_001", "Dangerous Run",            "threat", "Forward run into final third uncontested by a defender"),
    "THR_002": EventTypeSpec("THR_002", "Final-Third Entry",        "threat", "Player crosses x = +35 m moving forward"),
    "THR_003": EventTypeSpec("THR_003", "Box Entry",                "threat", "Player enters penalty box area"),
    "THR_004": EventTypeSpec("THR_004", "Transition Involvement",   "threat", "Player active in possession chain within 5s of transition"),
    "THR_005": EventTypeSpec("THR_005", "Dangerous Reception",      "threat", "Receives ball in final third with ≥ 2 m space"),
    "THR_006": EventTypeSpec("THR_006", "Channel Exploitation",     "threat", "Runs through corridor between CB and FB uncontested"),
    "THR_007": EventTypeSpec("THR_007", "Isolated Defender Exploit","threat", "1v1 in wide area with ≥ 3 m separation from defensive support"),

    # ── Shape ─────────────────────────────────────────────────────────────────
    "SHP_001": EventTypeSpec("SHP_001", "High Press Moment",        "shape", "Whole team defensive line x ≥ +20 m; pressure index ≤ 4.5 m"),
    "SHP_002": EventTypeSpec("SHP_002", "Mid Block",                "shape", "Defensive line x: −5 to +15 m; compact area ≤ 900 sq m"),
    "SHP_003": EventTypeSpec("SHP_003", "Low Block",                "shape", "Deepest defender x ≤ −25 m; team area ≤ 600 sq m"),
    "SHP_004": EventTypeSpec("SHP_004", "Compact Shape",            "shape", "Team width ≤ 35 m AND team length ≤ 40 m"),
    "SHP_005": EventTypeSpec("SHP_005", "Stretched Shape",          "shape", "Team length ≥ 65 m OR team width ≥ 55 m"),
    "SHP_006": EventTypeSpec("SHP_006", "Overload Zone",            "shape", "≥ 3 players from same team in a 15 m × 15 m zone"),
    "SHP_007": EventTypeSpec("SHP_007", "Pressing Trap Triggered",  "shape", "Coordinated press initiated and opponent forced to target zone"),
    "SHP_008": EventTypeSpec("SHP_008", "Counter-Attack Launch",    "shape", "≥ 3 players advance ≥ 5 m within 3 seconds of gaining possession"),

    # ── Transition ────────────────────────────────────────────────────────────
    "TRN_001": EventTypeSpec("TRN_001", "Defensive Transition",     "transition", "Team loses possession; all players tracked moving back"),
    "TRN_002": EventTypeSpec("TRN_002", "Offensive Transition",     "transition", "Team gains possession; forward players make advancing movements"),
    "TRN_003": EventTypeSpec("TRN_003", "Press Success",            "transition", "Ball won within 5 seconds of initiating press"),
    "TRN_004": EventTypeSpec("TRN_004", "Press Failure",            "transition", "Opposition plays through press with ≥ 2 passes"),
    "TRN_005": EventTypeSpec("TRN_005", "Counter-Attack Sequence",  "transition", "Offensive transition reaching final third within 8 seconds"),
}


def get_event_spec(event_type: str) -> EventTypeSpec:
    """Return the EventTypeSpec for a given ontology code."""
    if event_type not in EVENT_REGISTRY:
        raise KeyError(f"Unknown event type: {event_type!r}")
    return EVENT_REGISTRY[event_type]


# ──────────────────────────────────────────────────────────────────────────────
# Pitch zone boundaries (metres, origin at centre)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


ZONE_BOUNDS: dict[str, ZoneBounds] = {
    "defensive_third":    ZoneBounds(-52.5, -17.5, -34.0, 34.0),
    "middle_third":       ZoneBounds(-17.5,  17.5, -34.0, 34.0),
    "final_third":        ZoneBounds( 17.5,  52.5, -34.0, 34.0),

    # Channels (final third corridor between FB and CB)
    "left_channel":       ZoneBounds( 20.0,  52.5,  20.0, 34.0),
    "right_channel":      ZoneBounds( 20.0,  52.5, -34.0,-20.0),

    # Half-spaces (most dangerous receiving zone in modern football)
    "half_space_left":    ZoneBounds( 25.0,  52.5,  12.0, 25.0),
    "half_space_right":   ZoneBounds( 25.0,  52.5, -25.0,-12.0),

    # Central corridor (x > 25, between half-spaces)
    "central_corridor":   ZoneBounds( 25.0,  52.5, -12.0, 12.0),

    # Penalty box
    "box":                ZoneBounds( 36.5,  52.5, -20.2, 20.2),

    # Wide areas (outside half-spaces, full pitch)
    "wide_left":          ZoneBounds(-52.5,  52.5,  25.0, 34.0),
    "wide_right":         ZoneBounds(-52.5,  52.5, -34.0,-25.0),
}

# Ordered from most specific to most general for zone classification
ZONE_PRIORITY: list[str] = [
    "box",
    "left_channel",
    "right_channel",
    "half_space_left",
    "half_space_right",
    "central_corridor",
    "wide_left",
    "wide_right",
    "final_third",
    "middle_third",
    "defensive_third",
]


def classify_zone(x: float, y: float) -> str:
    """Return the most specific pitch zone for a given (x, y) position."""
    for zone_name in ZONE_PRIORITY:
        if ZONE_BOUNDS[zone_name].contains(x, y):
            return zone_name
    return "unknown"


FINAL_THIRD_ZONES = {"final_third", "box", "left_channel", "right_channel",
                     "half_space_left", "half_space_right", "central_corridor"}


# ──────────────────────────────────────────────────────────────────────────────
# Threat signal weights
# ──────────────────────────────────────────────────────────────────────────────

# Weights encode football domain knowledge. Must sum to 1.0.
THREAT_WEIGHTS: dict[str, float] = {
    "THR_003": 0.25,  # Box Entry — highest danger zone
    "THR_001": 0.22,  # Dangerous Run — requires defensive error
    "THR_006": 0.18,  # Channel Exploitation — structural vulnerability
    "THR_007": 0.15,  # Isolated Defender Exploit — 1v1 chance creation
    "THR_002": 0.10,  # Final-Third Entry — necessary precursor
    "THR_005": 0.07,  # Dangerous Reception — context-dependent
    "THR_004": 0.03,  # Transition Involvement — participation metric
}

assert abs(sum(THREAT_WEIGHTS.values()) - 1.0) < 1e-9, "Threat weights must sum to 1.0"

# Penalty multiplier for players whose only contribution is low-value events
TRANSITION_ONLY_PENALTY = 0.7

# Minimum threat score to appear in the coach report
MIN_THREAT_SCORE_FOR_REPORT = 15.0


# ──────────────────────────────────────────────────────────────────────────────
# Keyword → event type mapping for evidence query construction
# ──────────────────────────────────────────────────────────────────────────────

KEYWORD_EVENT_MAP: list[tuple[list[str], list[str]]] = [
    # (keyword_patterns, event_types)
    (["dangerous run", "exploits space", "runs behind", "behind the line"],
     ["THR_001", "THR_006"]),

    (["box", "penalty area", "penalty box", "inside the box"],
     ["THR_003", "THR_001"]),

    (["high press", "pressing", "press", "pressing trap"],
     ["SHP_001", "TRN_003"]),

    (["counter-attack", "counter attack", "transition", "break"],
     ["TRN_005", "TRN_002"]),

    (["overlap", "overlapping run", "width", "fullback"],
     ["MOV_004", "THR_007"]),

    (["half-space", "half space", "between lines", "between the lines"],
     ["POS_002", "POS_003"]),

    (["low block", "deep", "parked bus", "sitting deep"],
     ["SHP_003", "POS_004"]),

    (["sprint", "pace", "speed", "fast"],
     ["MOV_001", "MOV_002"]),

    (["recovery", "tracking back", "defensive run"],
     ["MOV_003", "TRN_001"]),

    (["diagonal", "cut inside", "inverted"],
     ["MOV_007", "MOV_005"]),

    (["channel", "in behind", "between cb and fb"],
     ["THR_006", "THR_001"]),

    (["isolated", "one on one", "1v1", "duel"],
     ["THR_007"]),

    (["compact", "shape", "narrow", "tight"],
     ["SHP_004", "SHP_002"]),

    (["stretched", "open", "gap", "space between"],
     ["SHP_005"]),

    (["overload", "numerical advantage", "outnumber"],
     ["SHP_006"]),
]


def keywords_to_event_types(text: str) -> list[str]:
    """Extract relevant event types from a natural language observation."""
    text_lower = text.lower()
    matched: list[str] = []
    seen: set[str] = set()
    for keywords, event_types in KEYWORD_EVENT_MAP:
        if any(kw in text_lower for kw in keywords):
            for et in event_types:
                if et not in seen:
                    matched.append(et)
                    seen.add(et)
    return matched


# ──────────────────────────────────────────────────────────────────────────────
# Detection thresholds (centralised here for single-source-of-truth)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectionThresholds:
    # Movement
    HIGH_SPEED_RUN_MPS: float = 6.5
    SPRINT_MPS: float = 8.0
    HIGH_SPEED_MIN_DURATION_S: float = 1.5
    SPRINT_MIN_DURATION_S: float = 0.5
    RECOVERY_SPEED_MPS: float = 5.0
    RECOVERY_MIN_DURATION_S: float = 0.8
    SPEED_SMOOTHING_WINDOW: int = 3
    RUN_MERGE_GAP_FRAMES: int = 10

    # Positional
    WIDE_Y_THRESHOLD_M: float = 25.0        # |y| >= this → wide
    HALF_SPACE_X_MIN_M: float = 25.0
    HALF_SPACE_Y_INNER_M: float = 12.0
    HALF_SPACE_Y_OUTER_M: float = 25.0
    BETWEEN_LINES_MIN_X_M: float = 10.0    # attacker x relative to opponent lines
    MIN_POSITIONAL_DURATION_S: float = 3.0
    POSITIONAL_EXIT_GAP_FRAMES: int = 10

    # Threat
    FINAL_THIRD_X_M: float = 25.0
    BOX_X_M: float = 36.5
    BOX_Y_M: float = 20.2
    UNCONTESTED_RADIUS_M: float = 3.0
    DANGEROUS_RUN_MIN_DURATION_S: float = 0.6
    UNCONTESTED_FRACTION: float = 0.75
    CHANNEL_X_MIN_M: float = 20.0
    CHANNEL_Y_INNER_M: float = 20.0
    CHANNEL_Y_OUTER_M: float = 34.0
    CHANNEL_UNCONTESTED_M: float = 4.5
    ISOLATED_DEFENDER_SUPPORT_M: float = 3.0
    RECEPTION_SPACE_M: float = 2.0

    # Shape
    HIGH_PRESS_DEF_LINE_X_M: float = 20.0
    HIGH_PRESS_PRESSURE_INDEX_M: float = 4.5
    HIGH_PRESS_MIN_DURATION_S: float = 1.5
    MID_BLOCK_X_MIN_M: float = -5.0
    MID_BLOCK_X_MAX_M: float = 15.0
    MID_BLOCK_AREA_M2: float = 900.0
    LOW_BLOCK_X_M: float = -25.0
    LOW_BLOCK_AREA_M2: float = 600.0
    COMPACT_WIDTH_M: float = 35.0
    COMPACT_LENGTH_M: float = 40.0
    STRETCHED_LENGTH_M: float = 65.0
    STRETCHED_WIDTH_M: float = 55.0
    OVERLOAD_ZONE_SIZE_M: float = 15.0
    OVERLOAD_MIN_PLAYERS: int = 3
    COUNTER_ADVANCE_M: float = 5.0
    COUNTER_ADVANCE_TIME_S: float = 3.0

    # Transition
    TRANSITION_WINDOW_S: float = 3.0
    PRESS_SUCCESS_WINDOW_S: float = 5.0
    COUNTER_REACH_FINAL_THIRD_S: float = 8.0

    # Clip extraction
    CLIP_PRE_PADDING_S: float = 2.5
    CLIP_POST_PADDING_S: float = 2.0
    CLIP_OVERLAP_MERGE_FRACTION: float = 0.70  # Merge clips overlapping this much

    # General
    HOMOGRAPHY_CONFIDENCE_DISCOUNT_THRESHOLD: float = 0.60


# Singleton instance used across all detectors
THRESHOLDS = DetectionThresholds()
