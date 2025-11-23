import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
FPS = 25
N_DEFENDERS = 4
IMG_DPI = 200


# =========================================================
# HELPERS
# =========================================================
def compute_def_line_x(frame, team_players):
    xs = []
    for p in frame.get("player_data", []):
        pid = p.get("player_id") or p.get("id")
        if pid in team_players:
            val = p.get("x")
            if val is not None:
                xs.append(val)

    if len(xs) == 0:
        return np.nan

    xs_sorted = sorted(xs)
    return xs_sorted[:N_DEFENDERS]


def infer_orientation(A_means, B_means):
    """Based on average X, infer who defends left or right."""
    A_vals = [v for v in A_means if not np.isnan(v)]
    B_vals = [v for v in B_means if not np.isnan(v)]

    if len(A_vals) == 0 or len(B_vals) == 0:
        return +1, +1

    if np.mean(A_vals) < np.mean(B_vals):
        return +1, -1
    else:
        return -1, +1


def normalize_to_100(values):
    arr = np.array(values, dtype=float)
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    if np.isclose(vmin, vmax):
        return np.full_like(arr, 50.0)

    return ((arr - vmin) / (vmax - vmin)) * 100.0


# =========================================================
# MAIN PLOT FUNCTION (with legends + quarter lines)
# =========================================================
def plot_def_line(series, outfile, label, total_frames):
    """Colour-coded defensive line plot with quarter markers + halftime."""

    # Bucket classification
    def bucket(x):
        if np.isnan(x): return "unknown"
        if x < 30: return "deep"
        if x < 60: return "balanced"
        if x < 90: return "high"
        return "aggressive"

    buckets = [bucket(v) for v in series]

    colors = {
        "deep":      "blue",
        "balanced":  "green",
        "high":      "orange",
        "aggressive":"red",
        "unknown":   "gray",
    }

    # minutes axis
    minutes = np.arange(len(series)) / (FPS * 60)

    plt.figure(figsize=(14, 4))

    # scatter colour-coded
    plt.scatter(
        minutes,
        series,
        s=5,
        c=[colors[b] for b in buckets],
        linewidth=0,
        alpha=0.9
    )

    # faint line behind scatter
    plt.plot(minutes, series, alpha=0.25, color="white")

    # quarter + half markers
    total_minutes = total_frames / (FPS * 60)

    quarter1 = total_minutes * 0.25
    half     = total_minutes * 0.50
    quarter3 = total_minutes * 0.75

    for x in [quarter1, half, quarter3]:
        plt.axvline(x, color="white", alpha=0.15, linestyle="--")

    plt.text(quarter1, max(series)*1.02, "Q1", color="white", fontsize=8)
    plt.text(half,     max(series)*1.02, "HT", color="white", fontsize=8)
    plt.text(quarter3, max(series)*1.02, "Q3", color="white", fontsize=8)

    # title + labels
    plt.title(f"Defensive Line (0–100 metric) — {label}")
    plt.xlabel("Match minutes")
    plt.ylabel("Defensive Line (0–100)")
    plt.grid(True, alpha=0.2)

    # legend
    patches = [
        plt.Line2D([0], [0], marker='o', linestyle='', 
                   markerfacecolor=colors[b], markersize=6, label=b)
        for b in colors.keys()
    ]
    plt.legend(handles=patches, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=IMG_DPI)
    plt.close()


# =========================================================
# MODEL 1 ENTRY POINT
# =========================================================
def model_main(match_path):
    match_path = Path(match_path)
    match_id = match_path.name

    out_dir = match_path / "model1_output"
    out_dir.mkdir(exist_ok=True)

    # -----------------------------
    # Load events (to get team IDs)
    # -----------------------------
    event_files = list(match_path.glob("*dynamic_events*.csv"))
    if len(event_files) > 0:
        events = pd.read_csv(event_files[0])
        try:
            teamA = events["team_id"].mode().iloc[0]
            all_teams = events["team_id"].unique().tolist()
            all_teams.remove(teamA)
            teamB = all_teams[0]
        except:
            teamA, teamB = 1, 2
    else:
        events = None
        teamA, teamB = 1, 2

    # -----------------------------
    # Load tracking (JSON or JSONL)
    # -----------------------------
    json_files  = list(match_path.glob("*tracking*.json"))
    jsonl_files = list(match_path.glob("*tracking*.jsonl"))

    frames_file = None
    if json_files:
        frames_file = json_files[0]
    elif jsonl_files:
        frames_file = jsonl_files[0]
    else:
        raise RuntimeError(f"No tracking JSON or JSONL found in {match_path}")

    # Load frames
    frames = []
    if frames_file.suffix == ".jsonl":
        with open(frames_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frames.append(json.loads(line))
                except:
                    pass
    else:
        with open(frames_file, "r") as f:
            frames = json.load(f)

    if len(frames) == 0:
        raise RuntimeError(f"Tracking file {frames_file} contains no valid frames")

    total_frames = len(frames)

    # -----------------------------
    # Determine players
    # -----------------------------
    if events is None:
        return {
            "plots": [],
            "output_folder": str(out_dir)
        }

    teamA_players = set(events[events["team_id"] == teamA]["player_id"].unique())
    teamB_players = set(events[events["team_id"] == teamB]["player_id"].unique())

    # -----------------------------
    # Raw defensive line per frame
    # -----------------------------
    rawA, rawB = [], []
    for f in frames:
        rawA.append(compute_def_line_x(f, teamA_players))
        rawB.append(compute_def_line_x(f, teamB_players))

    # mean of deepest N players
    A_means = [np.nanmean(x) if isinstance(x, list) else np.nan for x in rawA]
    B_means = [np.nanmean(x) if isinstance(x, list) else np.nan for x in rawB]

    # -----------------------------
    # Orientation correction
    # -----------------------------
    oriA, oriB = infer_orientation(A_means, B_means)
    A_adj = np.array(A_means) * oriA
    B_adj = np.array(B_means) * oriB

    # -----------------------------
    # Normalize to 0–100
    # -----------------------------
    A_norm = normalize_to_100(A_adj)
    B_norm = normalize_to_100(B_adj)

    # -----------------------------
    # Plotting
    # -----------------------------
    fA = out_dir / "def_line_A.png"
    fB = out_dir / "def_line_B.png"

    plot_def_line(A_norm, fA, f"Team {teamA}", total_frames)
    plot_def_line(B_norm, fB, f"Team {teamB}", total_frames)

    # -----------------------------
    # Return to Flask
    # -----------------------------
    return {
        "plots": [str(fA), str(fB)],
        "output_folder": str(out_dir)
    }