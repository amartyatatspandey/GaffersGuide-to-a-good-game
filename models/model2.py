"""
models/model2.py

Model 2 — Physical performance summary (paired output).
Returns a dict with "pairs": [
    {"title": "Total distance (km)", "A": "/match_output/...", "B": "/match_output/..."},
    ...
]
This file is designed to be robust to missing tracking / partial frames.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import warnings

# ---- Config ----
FPS = 25                          # frames per second for tracking files (adjust if needed)
SPRINT_SPEED_MPS = 7.0            # threshold for "sprint" (m/s) — tuneable
IMG_DPI = 200
N_TOP_PLAYERS = 6                 # when plotting player bars, show top N for readability

# ---- Helpers ----
def find_tracking_file(match_path: Path):
    """Return Path to tracking json/jsonl or None."""
    candidates = list(match_path.glob("*tracking*.json")) + list(match_path.glob("*structured_data*.json"))
    if candidates:
        return candidates[0]
    # fallback to jsonl
    jl = list(match_path.glob("*tracking*.jsonl"))
    if jl:
        return jl[0]
    return None

def load_tracking_frames(frames_file: Path):
    """Load frames from JSON or JSONL. Return a list of frames (may be empty)."""
    if frames_file is None:
        return []
    suf = frames_file.suffix.lower()
    frames = []
    try:
        if suf == ".jsonl":
            with frames_file.open("r", encoding="utf8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        frames.append(json.loads(line))
                    except Exception:
                        # tolerate bad lines
                        continue
        else:
            with frames_file.open("r", encoding="utf8") as fh:
                payload = json.load(fh)
                # Some JSON tracking files themselves are lists, others have keys
                if isinstance(payload, list):
                    frames = payload
                elif isinstance(payload, dict):
                    # try common keys
                    for k in ("frames", "data", "tracking"):
                        if k in payload and isinstance(payload[k], list):
                            frames = payload[k]
                            break
                    # fallback: if dict contains many numeric keys, treat as list values
                    if not frames:
                        # attempt to extract by presence of 'player_data' pattern
                        if "player_data" in payload:
                            frames = [payload]
    except Exception as e:
        warnings.warn(f"Failed to load tracking frames: {e}")
        return []
    return frames

def extract_positions_from_frame(frame):
    """
    Expect frame to contain 'player_data' or similar list of player dicts with
    'player_id','x','y' (x/y in meters). Returns dict player_id -> (x,y)
    """
    out = {}
    if not frame:
        return out
    # The frame format may vary; check common places
    candidates = []
    if isinstance(frame, dict):
        if "player_data" in frame and isinstance(frame["player_data"], list):
            candidates = frame["player_data"]
        elif "players" in frame and isinstance(frame["players"], list):
            candidates = frame["players"]
        else:
            # fallback: find first list-valued item that looks like players
            for v in frame.values():
                if isinstance(v, list):
                    if len(v) == 0:
                        continue
                    first = v[0]
                    if isinstance(first, dict) and ("player_id" in first or "id" in first):
                        candidates = v
                        break

    if not candidates:
        return out

    for p in candidates:
        pid = p.get("player_id") or p.get("id") or p.get("playerId") or p.get("sid")
        if pid is None:
            continue
        try:
            x = p.get("x")
            y = p.get("y")
            if x is None or y is None:
                # some datasets use 'position' or nested {'x','y'}
                pos = p.get("position") or p.get("pos")
                if isinstance(pos, dict):
                    x = pos.get("x", x)
                    y = pos.get("y", y)
            x = float(x) if x is not None else None
            y = float(y) if y is not None else None
        except Exception:
            x = None; y = None
        if x is None or y is None:
            continue
        out[int(pid)] = (x, y)
    return out

def compute_player_metrics(frames, fps=FPS):
    """
    Given frames list, compute per-player:
       - total distance (meters)
       - sprint seconds (seconds above SPRINT_SPEED_MPS)
       - frames counted
    Returns dict player_id -> metrics dict
    """
    # We will keep last-known positions per player to compute delta distances
    last_pos = {}
    totals = {}   # pid -> total_meters
    sprint_frames = {}  # pid -> count of sprint frames
    valid_frames = 0

    for i, frame in enumerate(frames):
        pos = extract_positions_from_frame(frame)
        if not pos:
            continue  # skip empty frames
        valid_frames += 1
        dt = 1.0 / fps

        # compute instantaneous speeds where possible
        for pid, (x, y) in pos.items():
            if pid not in totals:
                totals[pid] = 0.0
                sprint_frames[pid] = 0
            if pid in last_pos:
                lx, ly = last_pos[pid]
                dx = x - lx
                dy = y - ly
                dist = math.hypot(dx, dy)
                totals[pid] += dist
                speed = dist / dt
                if speed >= SPRINT_SPEED_MPS:
                    sprint_frames[pid] += 1
            # update last pos
            last_pos[pid] = (x, y)

    # convert sprint frames -> seconds; handle players with zero frames gracefully
    metrics = {}
    for pid in totals.keys():
        frames_for_pid = sprint_frames.get(pid, 0)
        metrics[pid] = {
            "total_m": totals[pid],
            "sprint_seconds": frames_for_pid / fps,
        }
    return metrics

def split_players_by_team_from_events(match_path: Path):
    """
    Attempt to find team membership using events CSV in match_path.
    Returns (teamA_id, teamB_id, mapping player_id->team)
    If events absent or not parsable, fallback to (None,None,{}).
    """
    teamA = None; teamB = None
    player_team = {}
    csvs = list(match_path.glob("*dynamic_events*.csv"))
    if not csvs:
        # fallback: try match.json for team ids
        js = list(match_path.glob("*match.json"))
        if js:
            try:
                meta = json.loads(js[0].read_text(encoding="utf8"))
                home = meta.get("home_team", {}).get("id")
                away = meta.get("away_team", {}).get("id")
                return home, away, {}
            except Exception:
                return None, None, {}
        return None, None, {}

    try:
        df = pd.read_csv(csvs[0], dtype={"team_id": object, "player_id": object})
        # infer two team ids
        team_ids = df["team_id"].dropna().unique().tolist()
        if len(team_ids) >= 2:
            teamA = team_ids[0]
            # choose a different id for teamB
            for t in team_ids:
                if t != teamA:
                    teamB = t
                    break
        # mapping players
        for _, row in df.iterrows():
            pid = row.get("player_id")
            tid = row.get("team_id")
            if pd.isna(pid) or pd.isna(tid):
                continue
            player_team[int(pid)] = tid
    except Exception:
        return None, None, {}
    return teamA, teamB, player_team

# ---- Plotting helpers ----
def plot_player_bar(metric_dict, title, outfile: Path, top_n=N_TOP_PLAYERS, unit="", color="#2bd97a"):
    """
    metric_dict: {pid: value}
    Produces a horizontal bar plot of top players by value.
    """
    if not metric_dict:
        # create placeholder image
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white", fontsize=14)
        ax.axis("off")
        fig.savefig(outfile, dpi=IMG_DPI, bbox_inches="tight")
        plt.close()
        return

    # sort players by value desc
    items = sorted(metric_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    pids = [str(k) for k, v in items]
    vals = [v for k, v in items]

    fig, ax = plt.subplots(figsize=(8, 4))
    y_pos = np.arange(len(pids))
    ax.barh(y_pos, vals, align="center", color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pids)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel(unit)
    ax.set_title(title, color="white")
    ax.grid(axis="x", alpha=0.25)
    fig.patch.set_facecolor("#0a120a")
    ax.set_facecolor("#0a120a")
    # set tick colors to white/muted
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="white")
    fig.savefig(outfile, dpi=IMG_DPI, bbox_inches="tight")
    plt.close()

# ---- Main entrypoint ----
def model_main(match_path):
    """
    Entry point called by Flask.
    Returns:
      {
        "pairs": [
           {"title": "Total distance (km)", "A": "/match_output/<path to img>", "B": "/match_output/<...>"},
           ...
        ],
        "output_folder": str(out_dir)
      }
    """

    match_path = Path(match_path)
    match_id = match_path.name
    out_dir = match_path / "model2_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load tracking frames (prefer JSON/JSONL) ----
    frames_file = find_tracking_file(match_path)
    frames = load_tracking_frames(frames_file)

    # If frames empty, we can still attempt to use events (limited)
    teamA_id, teamB_id, player_team_map = split_players_by_team_from_events(match_path)

    if not frames:
        # no frames; fallback: attempt to compute per-player aggregates from events if possible
        # but dynamic events rarely carry distances. We'll return placeholders.
        # Create placeholder images to keep UI consistent.
        placeholderA = out_dir / f"player_total_distance_A.png"
        placeholderB = out_dir / f"player_total_distance_B.png"
        plot_player_bar({}, f"Total distance — Team {teamA_id or 'A'}", placeholderA, unit="km")
        plot_player_bar({}, f"Total distance — Team {teamB_id or 'B'}", placeholderB, unit="km")

        sprintA = out_dir / f"player_sprint_seconds_A.png"
        sprintB = out_dir / f"player_sprint_seconds_B.png"
        plot_player_bar({}, f"Sprint seconds — Team {teamA_id or 'A'}", sprintA, unit="s")
        plot_player_bar({}, f"Sprint seconds — Team {teamB_id or 'B'}", sprintB, unit="s")

        fatA = out_dir / f"fatigue_score_A.png"
        fatB = out_dir / f"fatigue_score_B.png"
        plot_player_bar({}, f"Fatigue score — Team {teamA_id or 'A'}", fatA, unit="%")
        plot_player_bar({}, f"Fatigue score — Team {teamB_id or 'B'}", fatB, unit="%")

        pairs = [
            {"title": "Total Distance (km)", "A": f"/opendata/data/matches/{match_id}/model2_output/{placeholderA.name}", "B": f"/opendata/data/matches/{match_id}/model2_output/{placeholderB.name}"},
            {"title": "Sprint Seconds (s)", "A": f"/opendata/data/matches/{match_id}/model2_output/{sprintA.name}", "B": f"/opendata/data/matches/{match_id}/model2_output/{sprintB.name}"},
            {"title": "Fatigue Score (%)", "A": f"/opendata/data/matches/{match_id}/model2_output/{fatA.name}", "B": f"/opendata/data/matches/{match_id}/model2_output/{fatB.name}"},
        ]
        return {"pairs": pairs, "output_folder": str(out_dir)}

    # ---- compute player-level metrics from frames ----
    metrics = compute_player_metrics(frames, fps=FPS)
    if not metrics:
        # same placeholder behavior if metrics empty
        return model_main_no_frames(match_path, match_id, out_dir, teamA_id, teamB_id)

    # ---- split metrics by team using player_team_map if available
    teamA_metrics = {}
    teamB_metrics = {}

    if player_team_map:
        for pid, m in metrics.items():
            team = player_team_map.get(pid)
            if team == teamA_id:
                teamA_metrics[pid] = m
            elif team == teamB_id:
                teamB_metrics[pid] = m
            else:
                # unknown mapping; heuristically split by odd/even id to at least separate players
                if int(pid) % 2 == 0:
                    teamB_metrics[pid] = m
                else:
                    teamA_metrics[pid] = m
    else:
        # attempt heuristic: split player ids into two buckets by odd/even or by median id
        pids = sorted(metrics.keys())
        if not pids:
            return model_main_no_frames(match_path, match_id, out_dir, teamA_id, teamB_id)
        mid = len(pids)//2
        left = pids[:mid]
        right = pids[mid:]
        for pid in left:
            teamA_metrics[pid] = metrics[pid]
        for pid in right:
            teamB_metrics[pid] = metrics[pid]

    # Convert team metrics into simple numeric dicts for plotting
    totalA = {pid: round(m["total_m"] / 1000.0, 3) for pid, m in teamA_metrics.items()}  # km
    totalB = {pid: round(m["total_m"] / 1000.0, 3) for pid, m in teamB_metrics.items()}

    sprintA = {pid: round(m["sprint_seconds"], 2) for pid, m in teamA_metrics.items()}
    sprintB = {pid: round(m["sprint_seconds"], 2) for pid, m in teamB_metrics.items()}

    # Fatigue score: fraction of sprint seconds relative to total playing seconds (very rough)
    # First compute approximate playing seconds per player as total_frames * dt - we don't have per-player frames count, so approximate by total frames / FPS
    approx_total_seconds = max(1.0, len(frames) / FPS)
    fatA = {pid: round(100.0 * (m["sprint_seconds"] / approx_total_seconds), 2) for pid, m in teamA_metrics.items()}
    fatB = {pid: round(100.0 * (m["sprint_seconds"] / approx_total_seconds), 2) for pid, m in teamB_metrics.items()}

    # ---- make plots ----
    f_total_A = out_dir / f"player_total_distance_A.png"
    f_total_B = out_dir / f"player_total_distance_B.png"
    plot_player_bar(totalA, f"Total distance (km) — Team {teamA_id or 'A'}", f_total_A, unit="km")
    plot_player_bar(totalB, f"Total distance (km) — Team {teamB_id or 'B'}", f_total_B, unit="km")

    f_sprint_A = out_dir / f"player_sprint_seconds_A.png"
    f_sprint_B = out_dir / f"player_sprint_seconds_B.png"
    plot_player_bar(sprintA, f"Sprint seconds — Team {teamA_id or 'A'}", f_sprint_A, unit="s")
    plot_player_bar(sprintB, f"Sprint seconds — Team {teamB_id or 'B'}", f_sprint_B, unit="s")

    f_fat_A = out_dir / f"fatigue_score_A.png"
    f_fat_B = out_dir / f"fatigue_score_B.png"
    plot_player_bar(fatA, f"Fatigue score (%) — Team {teamA_id or 'A'}", f_fat_A, unit="%")
    plot_player_bar(fatB, f"Fatigue score (%) — Team {teamB_id or 'B'}", f_fat_B, unit="%")

    # ---- build pairs result with paths that the front-end can serve directly.
    # We return images under the /opendata/data/matches/<match_id>/... path so your cleaned_app pairing works.
    def make_url(p: Path):
        return f"/opendata/data/matches/{match_id}/model2_output/{p.name}"

    pairs = [
        {"title": "Total Distance (km)", "A": make_url(f_total_A), "B": make_url(f_total_B)},
        {"title": "Sprint Seconds (s)", "A": make_url(f_sprint_A), "B": make_url(f_sprint_B)},
        {"title": "Fatigue Score (%)", "A": make_url(f_fat_A), "B": make_url(f_fat_B)},
    ]

    return {"pairs": pairs, "output_folder": str(out_dir)}


# Fallback helper when no frames — keep UI stable
def model_main_no_frames(match_path, match_id, out_dir, teamA_id, teamB_id):
    out_dir.mkdir(parents=True, exist_ok=True)
    placeholderA = out_dir / f"player_total_distance_A.png"
    placeholderB = out_dir / f"player_total_distance_B.png"
    plot_player_bar({}, f"Total distance — Team {teamA_id or 'A'}", placeholderA, unit="km")
    plot_player_bar({}, f"Total distance — Team {teamB_id or 'B'}", placeholderB, unit="km")

    sprintA = out_dir / f"player_sprint_seconds_A.png"
    sprintB = out_dir / f"player_sprint_seconds_B.png"
    plot_player_bar({}, f"Sprint seconds — Team {teamA_id or 'A'}", sprintA, unit="s")
    plot_player_bar({}, f"Sprint seconds — Team {teamB_id or 'B'}", sprintB, unit="s")

    fatA = out_dir / f"fatigue_score_A.png"
    fatB = out_dir / f"fatigue_score_B.png"
    plot_player_bar({}, f"Fatigue score — Team {teamA_id or 'A'}", fatA, unit="%")
    plot_player_bar({}, f"Fatigue score — Team {teamB_id or 'B'}", fatB, unit="%")

    pairs = [
        {"title": "Total Distance (km)", "A": f"/opendata/data/matches/{match_id}/model2_output/{placeholderA.name}", "B": f"/opendata/data/matches/{match_id}/model2_output/{placeholderB.name}"},
        {"title": "Sprint Seconds (s)", "A": f"/opendata/data/matches/{match_id}/model2_output/{sprintA.name}", "B": f"/opendata/data/matches/{match_id}/model2_output/{sprintB.name}"},
        {"title": "Fatigue Score (%)", "A": f"/opendata/data/matches/{match_id}/model2_output/{fatA.name}", "B": f"/opendata/data/matches/{match_id}/model2_output/{fatB.name}"},
    ]
    return {"pairs": pairs, "output_folder": str(out_dir)}