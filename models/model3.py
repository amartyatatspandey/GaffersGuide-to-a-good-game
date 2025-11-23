# models/model3.py
import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pitch extents (same as other models)
X_MIN, X_MAX = -52.5, 52.5
Y_MIN, Y_MAX = -34.0, 34.0

IMG_DPI = 200

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def infer_orientation_from_events(events, teamA_id, teamB_id):
    """
    Returns (oriA, oriB) where ori is +1 if lower x => defending left (small x = deep),
    or -1 if large x = deep. Uses median x_start per team ignoring NaNs.
    """
    a_x = events.loc[events["team_id"] == teamA_id, "x_start"].dropna().astype(float)
    b_x = events.loc[events["team_id"] == teamB_id, "x_start"].dropna().astype(float)

    if len(a_x) == 0 or len(b_x) == 0:
        return +1, -1

    a_med = a_x.median()
    b_med = b_x.median()
    if a_med < b_med:
        return +1, -1
    else:
        return -1, +1

def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

def plot_full_heatmap(counts, xedges, yedges, outfile, title):
    fig, ax = plt.subplots(figsize=(10, 7))
    # flip y so that plotting coordinate equals pitch orientation
    pcm = ax.pcolormesh(xedges, yedges, counts.T, shading="auto")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0a120a")
    ax.set_title(title, color="white")
    cbar = plt.colorbar(pcm, ax=ax, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.tight_layout()
    fig.savefig(outfile, dpi=IMG_DPI)
    plt.close()

def draw_27cell_annotation(team_counts_3x3, outpath, team_name, orientation):
    """
    Draw the 3x3 per third grid annotated with class labels and counts.
    team_counts_3x3: dict{third_index: 3x3 numpy array of counts}
    orientation: +1 means defending left → attack right; we will label thirds accordingly.
    """
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#071007")

    # third boundaries (left->right)
    thirds = [X_MIN, (X_MIN+X_MAX)/3 + X_MIN, (X_MIN+X_MAX)/3*2 + X_MIN]  # not used precisely
    # For drawing a neat grid we'll compute per-third x ranges
    third_width = (X_MAX - X_MIN) / 3.0

    # y rows top-to-bottom
    row_height = (Y_MAX - Y_MIN) / 3.0

    # Colors for tile fill (map counts to colormap)
    max_count = max([arr.max() if arr.size else 0 for arr in team_counts_3x3.values()] + [1])
    cmap = plt.cm.get_cmap("Greens")

    # labels rule: a simple mapping based on third index and cell location
    def classify_cell(third, col, row):
        # third: 0=left,1=middle,2=right (in absolute coords). Orientation shifts meaning.
        # Map orientation so attacking third is where ori indicates (attack_right if +1)
        # We'll then return human-readable labels per (third, col, row)
        # col: 0..2 (left->right within the third), row: 0..2 (bottom->top)
        # We'll produce short classification strings.
        third_name = {0: "defensive", 1: "middle", 2: "attacking"}[third]
        # row 2 is top (y high), row 1 center, row0 bottom (y low)
        y_loc = {0: "low", 1: "mid", 2: "high"}[row]
        x_loc = {0: "outer", 1: "half-space", 2: "inner"}[col]
        if third_name == "attacking" and col == 2 and row == 1:
            return "Penetrative central"
        if third_name == "attacking" and col == 0:
            return "Right wing (if attack right) / Left wing"
        if third_name == "attacking" and col == 2:
            return "Inside-left/center final third"
        if third_name == "middle" and y_loc == "mid":
            return "Midfield heavy"
        if third_name == "defensive" and col == 2:
            return "High defensive line"
        return f"{third_name.title()} — {x_loc} / {y_loc}"

    for third in range(3):
        x0 = X_MIN + third * third_width
        for col in range(3):
            # column internal fraction within third
            col_x0 = x0 + (col) * (third_width / 3.0)
            for row in range(3):
                y0 = Y_MIN + row * row_height
                count = int(team_counts_3x3[third][col, row])
                color = cmap(count / (max_count + 1e-6))
                rect = plt.Rectangle((col_x0, y0), third_width / 3.0, row_height,
                                     facecolor=color, edgecolor="#0a2a18", linewidth=0.8, alpha=0.95)
                ax.add_patch(rect)
                # label: count + short classification
                lbl = classify_cell(third, col, row)
                ax.text(col_x0 + (third_width/6.0), y0 + row_height/2.0,
                        f"{count}\n{lbl}", va="center", ha="center", fontsize=8, color="white")

    ax.set_title(f"{team_name} — 3×3 per third classification", color="white")
    plt.tight_layout()
    fig.savefig(outpath, dpi=IMG_DPI)
    plt.close()

def model_main(match_path):
    """
    Entrypoint called by Flask. match_path should be a string path to match dir.
    Returns:
        {
            "served_plots": [list of file paths for serving],
            "full_heatmap": path or None,
            "output_folder": str(out_dir)
        }
    """
    match_path = Path(match_path)
    out_dir = match_path / "model3_output"
    ensure_out_dir(out_dir)

    # find dynamic events csv
    csv_candidates = list(match_path.glob("*dynamic_events*.csv")) + list(match_path.glob("*events*.csv"))
    if not csv_candidates:
        raise RuntimeError(f"No dynamic events CSV found in {match_path}")

    # load events (use first candidate), keep columns we need
    try:
        events = pd.read_csv(csv_candidates[0])
    except Exception as e:
        raise RuntimeError(f"Failed to load events CSV: {e}")

    # Ensure x_start/y_start exist
    if "x_start" not in events.columns or "y_start" not in events.columns:
        raise RuntimeError("Events CSV does not contain x_start/y_start columns")

    # Clean numeric columns
    events["x_start"] = pd.to_numeric(events["x_start"], errors="coerce")
    events["y_start"] = pd.to_numeric(events["y_start"], errors="coerce")

    # determine two team ids
    team_ids = pd.unique(events["team_id"].dropna().astype(int))
    if len(team_ids) < 2:
        # fallback: try team_in_possession_id or team_id from other columns
        # but for now raise since we need two teams
        raise RuntimeError("Could not derive two team ids from events CSV")

    teamA_id, teamB_id = int(team_ids[0]), int(team_ids[1])

    # team names (best-effort)
    teamA_name = str(events.loc[events["team_id"] == teamA_id, "team_shortname"].dropna().unique()[:1] or teamA_id)
    teamB_name = str(events.loc[events["team_id"] == teamB_id, "team_shortname"].dropna().unique()[:1] or teamB_id)

    # skip rows without coordinates
    events = events[events["x_start"].notna() & events["y_start"].notna()]

    # orientation
    oriA, oriB = infer_orientation_from_events(events, teamA_id, teamB_id)

    # Build full-match heatmaps using 50x34 bins (10px-ish)
    nx, ny = 50, 34
    xedges = np.linspace(X_MIN, X_MAX, nx + 1)
    yedges = np.linspace(Y_MIN, Y_MAX, ny + 1)

    served = []

    def process_team(tid, name, suffix, ori):
        team_ev = events[events["team_id"] == tid]
        xs = team_ev["x_start"].astype(float).to_numpy()
        ys = team_ev["y_start"].astype(float).to_numpy()
        if len(xs) == 0:
            # create blank heatmap array
            H = np.zeros((nx, ny))
        else:
            H, xe, ye = np.histogram2d(xs, ys, bins=[xedges, yedges])
        # save full heatmap image
        full_path = out_dir / f"ball_heatmap_full_{suffix}.png"
        plot_full_heatmap(H, xedges, yedges, full_path, f"Ball heatmap (team {name})")
        served.append(str(full_path))

        # compute 3x3 per third counts => dict third index -> 3x3 numpy
        third_counts = {}
        third_width = (X_MAX - X_MIN) / 3.0
        row_height = (Y_MAX - Y_MIN) / 3.0

        for third in range(3):
            third_x0 = X_MIN + third * third_width
            # create 3x3 zero
            grid = np.zeros((3, 3), dtype=int)
            # filter events in this third
            mask_th = (xs >= third_x0) & (xs < (third_x0 + third_width))
            xs_th = xs[mask_th]
            ys_th = ys[mask_th]
            for xval, yval in zip(xs_th, ys_th):
                # compute column 0..2 within third
                col = int(min(2, math.floor(((xval - third_x0) / third_width) * 3.0)))
                row = int(min(2, math.floor(((yval - Y_MIN) / (Y_MAX - Y_MIN)) * 3.0)))
                # flip row so visualization bottom->top matches intended mapping: keep as 0..2 bottom->top
                row = max(0, min(2, row))
                grid[col, row] += 1
            third_counts[third] = grid

        # draw annotated 27-cell figure
        grid_path = out_dir / f"three_by_three_{suffix}.png"
        draw_27cell_annotation(third_counts, grid_path, name, ori)
        served.append(str(grid_path))

        return H, third_counts, full_path, grid_path

    H_A, counts_A, fullA, gridA = process_team(teamA_id, teamA_name, "A", oriA)
    H_B, counts_B, fullB, gridB = process_team(teamB_id, teamB_name, "B", oriB)

    return {
        "served_plots": served,
        "full_heatmap": str(fullA) if served else None,
        "output_folder": str(out_dir)
    }