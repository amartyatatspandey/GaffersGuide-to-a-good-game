# model1_multi_matches.py
# Extended model1 to handle multiple matches (default: 6)
# Outputs:
#  - per-match: def_line_A.png, def_line_B.png, summary_{matchid}.csv
#  - aggregated: agg_percent_time_teamA.png, agg_percent_time_teamB.png, agg_summary.csv

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial import ConvexHull
import pandas as pd
import glob
import os

# Try importing your existing data loader if available
try:
    from model0_load_data import load_tracking, load_match_metadata
except Exception:
    load_tracking = None
    load_match_metadata = None

FPS = 25  # default frames-per-second used for minute conversion

# -------------------------
# Helpers (same logic as in model1)
# -------------------------
def frame_index_to_minute(frame_idx):
    return frame_idx / (FPS * 60.0)

def minute_to_bucket(minute):
    if minute < 0:
        return "unknown"
    if minute < 30:
        return "deep"
    if minute < 60:
        return "normal"
    if minute < 90:
        return "high"
    return "aggressive"

def map_players_to_teams(frames, meta):
    home_team_id = meta.get("home_team", {}).get("id") if isinstance(meta, dict) else None
    away_team_id = meta.get("away_team", {}).get("id") if isinstance(meta, dict) else None

    all_pids = []
    for f in frames:
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            if pid is not None:
                all_pids.append(pid)
    all_pids = list(dict.fromkeys(all_pids))

    home_players = set()
    away_players = set()
    found_teamfield = False
    for f in frames:
        for p in f.get("player_data", []):
            pid = p.get("player_id") or p.get("id")
            if pid is None:
                continue
            if p.get("team_id") is not None:
                found_teamfield = True
                if p.get("team_id") == home_team_id:
                    home_players.add(pid)
                elif p.get("team_id") == away_team_id:
                    away_players.add(pid)
            elif p.get("team") is not None:
                found_teamfield = True
                if p.get("team") == meta.get("home_team", {}).get("name") or p.get("team") == home_team_id:
                    home_players.add(pid)
                else:
                    away_players.add(pid)

    if not found_teamfield or (not home_players and not away_players):
        mid = len(all_pids) // 2
        home_players = set(all_pids[:mid])
        away_players = set(all_pids[mid:])

    return home_players, away_players, home_team_id, away_team_id

def compute_defensive_line_height(frames, team_players=None):
    heights = []
    for f in frames:
        players = []
        for p in f["player_data"]:
            pid = p.get("player_id") or p.get("id")
            if p.get("x") is None or p.get("y") is None:
                continue
            if team_players is not None:
                if pid not in team_players:
                    continue
            players.append({"x": p["x"], "y": p["y"], "id": pid})
        if len(players) == 0:
            heights.append(np.nan)
            continue
        # defensive players = 4 deepest => sorting by x ascending (assumption)
        players_sorted = sorted(players, key=lambda p: p["x"])
        def_line = players_sorted[:4]
        avg_y = np.mean([p["y"] for p in def_line]) if def_line else np.nan
        heights.append(avg_y)
    return heights

# plotting function (scatter + line) - color coded by bucket
def plot_defensive_line_time_series(heights, title, save_path):
    minutes = [frame_index_to_minute(i) for i in range(len(heights))]
    buckets = [minute_to_bucket(m) for m in minutes]
    color_map = {"deep": "tab:blue", "normal": "tab:green", "high": "tab:orange", "aggressive": "tab:red", "unknown": "gray"}
    colors = [color_map.get(b, "gray") for b in buckets]

    plt.figure(figsize=(12, 4))
    plt.scatter(minutes, heights, c=colors, s=8)
    plt.plot(minutes, heights, linewidth=0.7, alpha=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=color_map[k], markersize=6) 
               for k in ["deep", "normal", "high", "aggressive"]]
    plt.legend(handles=handles, title="Minute bucket")
    plt.title(title)
    plt.xlabel("Match minute")
    plt.ylabel("Defensive line mean Y")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# compute percent time in each bucket for a heights series
def percent_time_buckets(heights):
    mins = [frame_index_to_minute(i) for i in range(len(heights))]
    buckets = [minute_to_bucket(m) for m in mins]
    s = pd.Series(buckets)
    counts = s.value_counts().reindex(["deep","normal","high","aggressive"], fill_value=0)
    pct = 100.0 * counts / counts.sum() if counts.sum() > 0 else counts
    return pct.to_dict()

# -------------------------
# Load match data (robust)
# -------------------------
def load_match_from_path(path):
    """
    Accepts either:
     - path pointing to a JSON that is a list of frames (tracking),
     - or a directory containing 'tracking.json' and 'metadata.json',
     - or fallback to using model0_load_data.load_tracking() if available (only for first match).
    Returns frames (list) and meta (dict).
    """
    # If path is None and loader exists, use it
    if path is None and load_tracking is not None:
        frames = load_tracking()
        meta = load_match_metadata() if load_match_metadata is not None else {}
        return frames, meta

    # If path is a directory:
    p =  Path(path)
    if p.is_dir():
        # try common filenames
        candidates = ["tracking.json", "structured_data.json", "frames.json"]
        frames = None
        meta = {}
        for c in candidates:
            fp = p / c
            if fp.exists():
                with open(fp, "r", encoding="utf-8") as fh:
                    frames = json.load(fh)
                break
        # metadata file
        for mf in ["metadata.json", "match_metadata.json", "meta.json"]:
            mp = p / mf
            if mp.exists():
                try:
                    with open(mp, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                except:
                    meta = {}
                break
        if frames is None:
            raise FileNotFoundError(f"No tracking file found in folder {p}")
        return frames, meta

    # If path is a file
    if p.is_file():
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # If the file contains both frames and meta as dict keys:
        if isinstance(data, dict) and ("frames" in data or "player_data" in (data.keys())):
            # try common patterns
            if "frames" in data:
                frames = data["frames"]
                meta = data.get("meta", {}) or data.get("match", {})
            else:
                # if single object representing a frame sequence
                frames = data if isinstance(data, list) else data.get("player_data", [])
                meta = {}
        elif isinstance(data, list):
            frames = data
            meta = {}
        else:
            raise ValueError(f"Unrecognized JSON structure in {p}")
        return frames, meta

    raise FileNotFoundError(f"Path {path} not found")

# -------------------------
# Run for multiple matches
# -------------------------
def run_on_matches(match_paths, output_dir="model1_output_multi"):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    aggregated_rows = []
    agg_teamA_pcts = []
    agg_teamB_pcts = []
    match_labels = []

    for idx, mpath in enumerate(match_paths):
        try:
            print(f"Processing match {idx+1}/{len(match_paths)} -> {mpath}")
            frames, meta = load_match_from_path(mpath)
        except Exception as e:
            print(f"Failed to load match {mpath}: {e}")
            continue

        match_id = meta.get("match_id") or meta.get("id") or f"match_{idx+1}"
        out_dir = out_root / f"{match_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Map players to teams
        home_players, away_players, home_team_id, away_team_id = map_players_to_teams(frames, meta)

        # Defensive line heights
        dlA = compute_defensive_line_height(frames, team_players=home_players)
        dlB = compute_defensive_line_height(frames, team_players=away_players)

        # per-match plots
        plot_defensive_line_time_series(dlA, f"Team {home_team_id or 'A'} Defensive Line - {match_id}", out_dir / f"def_line_A.png")
        plot_defensive_line_time_series(dlB, f"Team {away_team_id or 'B'} Defensive Line - {match_id}", out_dir / f"def_line_B.png")

        # percent time in each bucket
        pctA = percent_time_buckets(dlA)
        pctB = percent_time_buckets(dlB)

        # save per-match CSV
        df_match = pd.DataFrame({
            "bucket":["deep","normal","high","aggressive"],
            "teamA_pct":[pctA.get(k,0) for k in ["deep","normal","high","aggressive"]],
            "teamB_pct":[pctB.get(k,0) for k in ["deep","normal","high","aggressive"]],
        })
        csv_path = out_dir / f"summary_{match_id}.csv"
        df_match.to_csv(csv_path, index=False)
        print(f"Saved per-match summary to {csv_path}")

        # accumulate for aggregate
        agg_teamA_pcts.append([pctA.get(k,0) for k in ["deep","normal","high","aggressive"]])
        agg_teamB_pcts.append([pctB.get(k,0) for k in ["deep","normal","high","aggressive"]])
        match_labels.append(str(match_id))
        aggregated_rows.append((match_id, pctA, pctB))

    # Create aggregated plots if we processed at least one match
    if agg_teamA_pcts:
        aggA = np.array(agg_teamA_pcts)  # shape: (n_matches, 4)
        aggB = np.array(agg_teamB_pcts)
        labels = ["deep","normal","high","aggressive"]

        # create a grouped bar chart across matches for Team A
        n = aggA.shape[0]
        x = np.arange(n)
        width = 0.18
        plt.figure(figsize=(12,5))
        for i in range(4):
            plt.bar(x + i*width, aggA[:,i], width=width, label=labels[i])
        plt.xticks(x + 1.5*width, match_labels, rotation=45)
        plt.ylabel("Percent time (%)")
        plt.title("Team A: Percent time in defensive-line buckets (per match)")
        plt.legend()
        plt.tight_layout()
        aggA_path = out_root / "agg_percent_time_teamA.png"
        plt.savefig(aggA_path)
        plt.close()
        print(f"Saved aggregated Team A plot to {aggA_path}")

        # Team B
        plt.figure(figsize=(12,5))
        for i in range(4):
            plt.bar(x + i*width, aggB[:,i], width=width, label=labels[i])
        plt.xticks(x + 1.5*width, match_labels, rotation=45)
        plt.ylabel("Percent time (%)")
        plt.title("Team B: Percent time in defensive-line buckets (per match)")
        plt.legend()
        plt.tight_layout()
        aggB_path = out_root / "agg_percent_time_teamB.png"
        plt.savefig(aggB_path)
        plt.close()
        print(f"Saved aggregated Team B plot to {aggB_path}")

        # Save aggregated CSV
        agg_rows = []
        for i, mid in enumerate(match_labels):
            row = {"match_id": mid}
            for j, k in enumerate(labels):
                row[f"teamA_{k}_pct"] = float(aggA[i,j])
                row[f"teamB_{k}_pct"] = float(aggB[i,j])
            agg_rows.append(row)
        df_agg = pd.DataFrame(agg_rows)
        df_agg.to_csv(out_root / "agg_summary.csv", index=False)
        print(f"Saved aggregated CSV to {out_root / 'agg_summary.csv'}")

    print("All done. Outputs are in:", out_root)

# -------------------------
# CLI / ENTRY POINT
# -------------------------
if __name__ == "__main__":
    """
    Usage:
      - Put your 6 match JSONs (or folders) in a directory named 'matches/' and name them any way you like.
      - Or pass explicit paths in the list below.
    Example:
      python model1_multi_matches.py
    """
    # Attempt to find up to 6 matches automatically under ./matches/
    candidate_dir = Path("matches")
    if candidate_dir.exists() and candidate_dir.is_dir():
        # pick up to 6 match files/folders inside matches/
        entries = sorted([str(p) for p in candidate_dir.iterdir() if not p.name.startswith(".")])[:6]
        if len(entries) == 0 and load_tracking is not None:
            # fallback to using built-in loader 6 times (if it supports different matchs) - unlikely
            entries = [None] * 6
    else:
        # default fallback: if load_tracking exists, run single match from it
        if load_tracking is not None:
            entries = [None]  # process the loaded match
        else:
            # nothing found; inform user and exit
            raise FileNotFoundError("No 'matches/' directory found and no data-loader available. Place up to 6 match JSONs or directories under ./matches/")

    # If user wants exactly 6, but fewer files exist, script will process whatever it finds.
    run_on_matches(entries)
