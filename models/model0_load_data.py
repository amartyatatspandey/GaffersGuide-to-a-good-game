from pathlib import Path
import json
import pandas as pd

# Base directory for SkillCorner open-data inside your cleaned repo
BASE = Path(__file__).resolve().parents[1] / "opendata" / "data" / "matches"


def _load_jsonl(path):
    frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    return frames


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path):
    return pd.read_csv(path, low_memory=False)


def load_match(match_id):
    """
    Loads tracking frames, metadata, events and phases for a given SkillCorner match ID.

    Expected structure:
        opendata/data/matches/<match_id>/
            *_tracking_extrapolated.jsonl
            *_match.json
            *_dynamic_events.csv
            *_phases_of_play.csv
    """

    match_dir = BASE / str(match_id)

    if not match_dir.exists():
        raise FileNotFoundError(f"Match folder not found: {match_dir}")

    # TRACKING FILE
    tracking = None
    candidates = list(match_dir.glob("*tracking_extrapolated.jsonl")) \
                 + list(match_dir.glob("*tracking.jsonl")) \
                 + list(match_dir.glob("structured_data.json"))

    if not candidates:
        raise FileNotFoundError(f"No tracking file in {match_dir}")

    tf = candidates[0]

    if tf.suffix == ".jsonl":
        tracking = _load_jsonl(tf)
    else:
        raw = _load_json(tf)
        tracking = raw.get("frames", raw)

    # METADATA
    meta_files = list(match_dir.glob("*match.json"))
    if not meta_files:
        raise FileNotFoundError(f"No match.json in {match_dir}")
    metadata = _load_json(meta_files[0])

    # EVENTS
    event_files = list(match_dir.glob("*dynamic_events.csv"))
    events = _load_csv(event_files[0]) if event_files else pd.DataFrame()

    # PHASES
    phase_files = list(match_dir.glob("*phases_of_play.csv"))
    phases = _load_csv(phase_files[0]) if phase_files else pd.DataFrame()

    return tracking, metadata, events, phases