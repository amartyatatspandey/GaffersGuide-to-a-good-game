# models/model2.py
"""
Model 2 — Player-level Physical Performance (Option A)
Signature: model_main(match_id, match_folder) -> dict

- Uses 'player_data' frames (each frame has player entries with player_id, x, y)
- Infers speeds from position differences (ASSUMED_DT if timestamps missing)
- Produces per-player bar charts (all players on x-axis, colored by team)
- Saves PNGs into <match_folder>/model2_output/
- Returns team totals + served_plots list for cleaned_app.py
"""

import json
from pathlib import Path
from math import hypot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ---- Tunables ----
SPRINT_SPEED_THRESHOLD = 7.0   # m/s used to classify sprint (tune as needed)
ASSUMED_DT = 0.1               # seconds between frames if timestamp missing
COORD_SCALE = 1.0              # multiply coordinate diffs by this to convert to meters (1.0 if coords already meters)

# ---------------- Helpers ----------------
def parse_timestamp(ts):
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        if isinstance(ts, str) and ts.count(":") == 2:
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        return float(ts)
    except Exception:
        return None

def find_tracking_file(match_folder: Path):
    # prefer extrapolated jsonl, then any jsonl/json containing 'tracking' or 'player'
    p = Path(match_folder)
    if not p.exists():
        return None
    for candidate in p.iterdir():
        name = candidate.name.lower()
        if candidate.is_file() and "extrapolated" in name and candidate.suffix in (".jsonl", ".ndjson", ".json"):
            return candidate
    for candidate in p.iterdir():
        name = candidate.name.lower()
        if candidate.is_file() and ("tracking" in name or "player" in name) and candidate.suffix in (".jsonl", ".ndjson", ".json"):
            return candidate
    for candidate in p.rglob("*.jsonl"):
        if "tracking" in candidate.name.lower() or "player" in candidate.name.lower():
            return candidate
    return None

def load_match_meta(match_folder: Path):
    files = list(match_folder.glob("*_match.json")) + list(match_folder.glob("match.json"))
    if not files:
        return None
    try:
        with files[0].open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None

def safe_iter_jsonl(path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if not first:
            return
        first_strip = first.strip()
        if first_strip.startswith("{"):
            # jsonl: yield first then rest
            try:
                yield json.loads(first_strip)
            except Exception:
                pass
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
        else:
            # maybe whole-array JSON
            try:
                rest = fh.read()
                whole = first + rest
                data = json.loads(whole)
                if isinstance(data, list):
                    for item in data:
                        yield item
            except Exception:
                return

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def player_colors_by_team(player_ids, player_to_team, home_id, away_id):
    # return color list for players; A = blue, B = orange (matplotlib defaults)
    colors = []
    for pid in player_ids:
        tid = player_to_team.get(pid)
        if tid == home_id:
            colors.append("#2f7bbf")  # blue-ish
        elif tid == away_id:
            colors.append("#ee8b3b")  # orange-ish
        else:
            colors.append("#7f7f7f")  # grey fallback
    return colors

# ---------------- Main ----------------
def model_main(match_id, match_folder):
    """
    match_folder: path to opendata/data/matches/<match_id>
    Returns:
      - total_distance_A, total_distance_B (meters)
      - sprint_A, sprint_B (meters)
      - fatigue_A, fatigue_B (0-100)
      - served_plots: list of image paths (accessible by cleaned_app via /match_output/)
    """
    try:
        match_folder = Path(match_folder)
        if not match_folder.exists():
            raise RuntimeError(f"Match folder not found: {match_folder}")

        meta = load_match_meta(match_folder)
        # build player -> team mapping from match metadata if available
        player_to_team = {}
        if meta:
            for p in meta.get("players", []):
                pid = p.get("id")
                tid = p.get("team_id")
                if pid is not None and tid is not None:
                    player_to_team[pid] = tid
        home_id = meta.get("home_team", {}).get("id") if meta else None
        away_id = meta.get("away_team", {}).get("id") if meta else None

        tracking_file = find_tracking_file(match_folder)
        if not tracking_file:
            raise RuntimeError(f"Tracking file not found under {match_folder}")

        # per-player trackers
        last_pos = {}        # pid -> (x,y)
        last_time = {}       # pid -> seconds
        total_dist = {}      # pid -> meters
        sprint_dist = {}     # pid -> meters counted as sprint
        sprint_seconds = {}  # pid -> seconds spent sprinting
        # optional per-frame detection counts
        detection_count = {}

        # iterate frames
        for frame in safe_iter_jsonl(tracking_file):
            if not isinstance(frame, dict):
                continue
            ts = parse_timestamp(frame.get("timestamp"))
            # support either 'player_data' or 'players'
            players = frame.get("player_data") or frame.get("players") or []
            if not isinstance(players, list):
                continue

            for p in players:
                # skillcorner-style: player_id, x, y, is_detected
                pid = p.get("player_id") or p.get("id")
                x = p.get("x")
                y = p.get("y")
                is_detected = p.get("is_detected", True)

                if pid is None or x is None or y is None:
                    continue

                # init containers
                total_dist.setdefault(pid, 0.0)
                sprint_dist.setdefault(pid, 0.0)
                sprint_seconds.setdefault(pid, 0.0)
                detection_count.setdefault(pid, 0)

                if not is_detected:
                    # still update last positions? better to skip movement when not detected
                    # don't update last_pos to avoid false large jumps when reacquired
                    detection_count[pid] += 0
                else:
                    detection_count[pid] += 1

                prev = last_pos.get(pid)
                prev_t = last_time.get(pid)

                # determine dt
                dt = ASSUMED_DT
                if ts is not None and prev_t is not None:
                    dt = max(1e-6, ts - prev_t)

                if prev is not None and is_detected:
                    dx = (x - prev[0]) * COORD_SCALE
                    dy = (y - prev[1]) * COORD_SCALE
                    dist = hypot(dx, dy)
                    total_dist[pid] += dist
                    speed = dist / dt if dt > 0 else 0.0
                    if speed >= SPRINT_SPEED_THRESHOLD:
                        sprint_dist[pid] += dist
                        sprint_seconds[pid] += dt

                # update last only when detected to avoid teleports
                if is_detected:
                    last_pos[pid] = (x, y)
                    if ts is not None:
                        last_time[pid] = ts
                    else:
                        last_time[pid] = last_time.get(pid, 0.0) + ASSUMED_DT

        # ---- Aggregate per-team and per-player results ----
        # Determine players list (all seen)
        player_ids = sorted(total_dist.keys())

        # Build lists for plotting
        distances = [round(total_dist.get(pid, 0.0), 2) for pid in player_ids]
        sprints = [round(sprint_dist.get(pid, 0.0), 2) for pid in player_ids]
        sprint_secs = [round(sprint_seconds.get(pid, 0.0), 2) for pid in player_ids]

        # compute player-level fatigue score: a simple composite
        # fatigue = normalized sprint_seconds (0..100 by 600s -> 10min) + small normalized distance factor
        fatigue_scores = []
        for pid, ss, d in zip(player_ids, sprint_secs, distances):
            mins = ss / 60.0
            base = min(100.0, (mins / 10.0) * 100.0)   # 10 minutes sprinting -> 100
            extra = min(20.0, (d / 10000.0) * 20.0)    # 10km -> +20
            fatigue_scores.append(round(base + extra, 2))

        # Team aggregation
        team_totals = {"A": 0.0, "B": 0.0}
        team_sprints = {"A": 0.0, "B": 0.0}
        team_sprint_seconds = {"A": 0.0, "B": 0.0}

        # assign each player to team label A/B using meta mapping; fallback: infer from x sign if no mapping
        fallback_label = {}
        if not player_to_team:
            # infer using last_pos x (if present)
            for pid, pos in last_pos.items():
                x = pos[0] if pos is not None else 0.0
                fallback_label[pid] = "A" if x < 0 else "B"

        for pid in player_ids:
            tid = player_to_team.get(pid)
            if tid is not None:
                if home_id is not None and tid == home_id:
                    lab = "A"
                elif away_id is not None and tid == away_id:
                    lab = "B"
                else:
                    lab = "A"  # unknown team id -> default A
            else:
                lab = fallback_label.get(pid, "A")
            team_totals[lab] += total_dist.get(pid, 0.0)
            team_sprints[lab] += sprint_dist.get(pid, 0.0)
            team_sprint_seconds[lab] += sprint_seconds.get(pid, 0.0)

        # compute team fatigue summaries
        def compute_fatigue(sprint_seconds_val, total_distance_val):
            mins = sprint_seconds_val / 60.0
            score = min(100.0, (mins / 10.0) * 100.0)
            extra = min(20.0, (total_distance_val / 10000.0) * 20.0)
            return round(score + extra, 2)

        total_distance_A = round(team_totals["A"], 2)
        total_distance_B = round(team_totals["B"], 2)
        sprint_A = round(team_sprints["A"], 2)
        sprint_B = round(team_sprints["B"], 2)
        fatigue_A = compute_fatigue(team_sprint_seconds["A"], team_totals["A"])
        fatigue_B = compute_fatigue(team_sprint_seconds["B"], team_totals["B"])

        # -------------------- Plotting --------------------
        out_dir = match_folder / "model2_output"
        ensure_dir(out_dir)

        # labels for x-axis: prefer player short_name if present in meta, else id
        id_to_label = {}
        if meta:
            for p in meta.get("players", []):
                pid = p.get("id")
                short = p.get("short_name") or f"{p.get('first_name','') or ''} {p.get('last_name','') or ''}".strip()
                if pid is not None:
                    id_to_label[pid] = short if short else str(pid)
        # fallback use id
        xlabels = [id_to_label.get(pid, str(pid)) for pid in player_ids]

        # colors: by team
        colors = player_colors_by_team(player_ids, player_to_team, home_id, away_id)

        # Plot 1: Total distance per player
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        bars = ax1.bar(range(len(player_ids)), distances, color=colors)
        ax1.set_xticks(range(len(player_ids)))
        ax1.set_xticklabels(xlabels, rotation=45, ha="right")
        ax1.set_ylabel("Total Distance (meters)")
        ax1.set_title("Total Distance — per player")
        for idx, v in enumerate(distances):
            ax1.text(idx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig1.tight_layout()
        td_file = out_dir / "players_total_distance.png"
        fig1.savefig(td_file)
        plt.close(fig1)

        # Plot 2: Sprint distance per player
        fig2, ax2 = plt.subplots(figsize=(12, 5))
        bars2 = ax2.bar(range(len(player_ids)), sprints, color=colors)
        ax2.set_xticks(range(len(player_ids)))
        ax2.set_xticklabels(xlabels, rotation=45, ha="right")
        ax2.set_ylabel("Sprint Distance (meters)")
        ax2.set_title(f"Sprint Distance (speed ≥ {SPRINT_SPEED_THRESHOLD} m/s) — per player")
        for idx, v in enumerate(sprints):
            ax2.text(idx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig2.tight_layout()
        sd_file = out_dir / "players_sprint_distance.png"
        fig2.savefig(sd_file)
        plt.close(fig2)

        # Plot 3: Fatigue per player
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        bars3 = ax3.bar(range(len(player_ids)), fatigue_scores, color=colors)
        ax3.set_xticks(range(len(player_ids)))
        ax3.set_xticklabels(xlabels, rotation=45, ha="right")
        ax3.set_ylabel("Fatigue Score (0-100)")
        ax3.set_title("Fatigue Score — per player")
        for idx, v in enumerate(fatigue_scores):
            ax3.text(idx, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        fig3.tight_layout()
        fat_file = out_dir / "players_fatigue.png"
        fig3.savefig(fat_file)
        plt.close(fig3)

        # Served plot paths that cleaned_app can use:
        served = [
            f"/match_output/{match_id}/model2_output/{td_file.name}",
            f"/match_output/{match_id}/model2_output/{sd_file.name}",
            f"/match_output/{match_id}/model2_output/{fat_file.name}",
        ]

        # return numbers plus served plots
        return {
            "total_distance_A": total_distance_A,
            "total_distance_B": total_distance_B,
            "sprint_A": sprint_A,
            "sprint_B": sprint_B,
            "fatigue_A": fatigue_A,
            "fatigue_B": fatigue_B,
            "served_plots": served,
            # extras for debugging/UI if needed
            "per_player": {
                "player_ids": player_ids,
                "labels": xlabels,
                "distances": distances,
                "sprint_distances": sprints,
                "sprint_seconds": sprint_secs,
                "fatigue_scores": fatigue_scores,
            }
        }

    except Exception as e:
        raise RuntimeError(f"Model 2 processing failed: {e}")

# quick local invocation for debugging
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 3:
        print("Usage: python models/model2.py <match_id> <match_folder>")
    else:
        mid = sys.argv[1]
        mfolder = sys.argv[2]
        print(json.dumps(model_main(mid, mfolder), indent=2))