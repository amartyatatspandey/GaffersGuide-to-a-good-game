import json
import numpy as np
import cv2
import math
from pathlib import Path

# Paths
DATA_PATH = Path("backend/output/clean_tactical_data.json")
OUTPUT_PATH = Path("backend/output/tactical_metrics.json")

FPS = 30.0
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


class TacticalAnalyzer:
    def __init__(self):
        self.player_history = {}  # Tracks {id: {'frame': idx, 'pt': [x,y]}} for speed math

        # Pre-compute grid for Pitch Control (10x7 grid)
        self.grid_x = np.linspace(-PITCH_LENGTH / 2, PITCH_LENGTH / 2, 10)
        self.grid_y = np.linspace(-PITCH_WIDTH / 2, PITCH_WIDTH / 2, 7)
        self.grid_points = np.array([[x, y] for x in self.grid_x for y in self.grid_y])

    def calc_speed(self, player_id, current_frame, current_pt):
        speed_kmh = 0.0
        if player_id in self.player_history:
            last_frame = self.player_history[player_id]["frame"]
            last_pt = self.player_history[player_id]["pt"]

            frames_elapsed = current_frame - last_frame
            if 0 < frames_elapsed < FPS * 2:  # Only calc if seen in the last 2 seconds
                dist_meters = math.hypot(
                    current_pt[0] - last_pt[0], current_pt[1] - last_pt[1]
                )
                time_seconds = frames_elapsed / FPS
                speed_ms = dist_meters / time_seconds
                speed_kmh = speed_ms * 3.6

        # Update history
        self.player_history[player_id] = {"frame": current_frame, "pt": current_pt}
        return speed_kmh

    def calculate_pressure_index(self, team_pts, opp_pts):
        if not team_pts or not opp_pts:
            return 0.0
        team_arr = np.array(team_pts)
        opp_arr = np.array(opp_pts)

        min_distances = []
        for pt in team_arr:
            # Euclidean distance to all opponents
            dists = np.linalg.norm(opp_arr - pt, axis=1)
            min_distances.append(np.min(dists))

        return float(np.mean(min_distances))  # Lower = tighter marking

    def calculate_line_gaps(self, team_pts):
        if len(team_pts) < 6:
            return 0.0, 0.0

        # Sort by X coordinate (depth)
        sorted_x = sorted([pt[0] for pt in team_pts])

        # Split into approximate thirds (Defense, Midfield, Attack)
        third = len(sorted_x) // 3
        def_x = np.mean(sorted_x[:third])
        mid_x = np.mean(sorted_x[third : 2 * third])
        att_x = np.mean(sorted_x[2 * third :])

        return float(abs(mid_x - def_x)), float(abs(att_x - mid_x))

    def calculate_pitch_control(self, team_0_pts, team_1_pts):
        if not team_0_pts or not team_1_pts:
            return 50.0, 50.0

        t0_arr = np.array(team_0_pts)
        t1_arr = np.array(team_1_pts)

        t0_control = 0
        for g_pt in self.grid_points:
            dist_to_t0 = np.min(np.linalg.norm(t0_arr - g_pt, axis=1))
            dist_to_t1 = np.min(np.linalg.norm(t1_arr - g_pt, axis=1))
            if dist_to_t0 < dist_to_t1:
                t0_control += 1

        t0_pct = (t0_control / len(self.grid_points)) * 100.0
        return float(t0_pct), float(100.0 - t0_pct)

    def analyze_team_spatial(self, team_pts):
        if len(team_pts) < 3:
            return None
        pts_array = np.array(team_pts, dtype=np.float32)

        centroid = [float(np.mean(pts_array[:, 0])), float(np.mean(pts_array[:, 1]))]
        min_x, max_x = float(np.min(pts_array[:, 0])), float(np.max(pts_array[:, 0]))
        width = float(np.max(pts_array[:, 1]) - np.min(pts_array[:, 1]))
        length = float(max_x - min_x)

        try:
            hull = cv2.convexHull(pts_array.reshape(-1, 1, 2))
            area = float(cv2.contourArea(hull))
        except:
            area = 0.0

        return {
            "centroid": centroid,
            "area_sq_meters": area,
            "team_width_m": width,
            "team_length_m": length,
            "deepest_x": min_x,
            "highest_x": max_x,
        }


def run_analytics():
    if not DATA_PATH.exists():
        print(f"Error: {DATA_PATH} not found.")
        return

    with open(DATA_PATH, "r") as f:
        frames = json.load(f)

    analyzer = TacticalAnalyzer()
    analytics_timeline = []

    for frame_data in frames:
        frame_idx = frame_data.get("frame_idx", 0)
        players = frame_data.get("players", [])

        t0_pts, t1_pts = [], []
        t0_speeds, t1_speeds = [], []

        for p in players:
            pt = p.get("radar_pt")
            pid = p.get("id")
            team = p.get("team")

            if pt is None or pid is None:
                continue

            speed = analyzer.calc_speed(pid, frame_idx, pt)

            if team == "team_0":
                t0_pts.append(pt)
                t0_speeds.append(speed)
            elif team == "team_1":
                t1_pts.append(pt)
                t1_speeds.append(speed)

        # 1. Base Spatial Metrics
        metrics_0 = analyzer.analyze_team_spatial(t0_pts)
        metrics_1 = analyzer.analyze_team_spatial(t1_pts)

        if metrics_0 and metrics_1:
            # 2. Advanced Metrics
            t0_pct, t1_pct = analyzer.calculate_pitch_control(t0_pts, t1_pts)

            t0_def_mid, t0_mid_att = analyzer.calculate_line_gaps(t0_pts)
            t1_def_mid, t1_mid_att = analyzer.calculate_line_gaps(t1_pts)

            metrics_0.update(
                {
                    "pitch_control_pct": t0_pct,
                    "pressure_index_m": analyzer.calculate_pressure_index(t0_pts, t1_pts),
                    "line_gap_def_mid_m": t0_def_mid,
                    "line_gap_mid_att_m": t0_mid_att,
                    "avg_speed_kmh": float(np.mean(t0_speeds)) if t0_speeds else 0.0,
                    "max_speed_kmh": float(np.max(t0_speeds)) if t0_speeds else 0.0,
                }
            )

            metrics_1.update(
                {
                    "pitch_control_pct": t1_pct,
                    "pressure_index_m": analyzer.calculate_pressure_index(t1_pts, t0_pts),
                    "line_gap_def_mid_m": t1_def_mid,
                    "line_gap_mid_att_m": t1_mid_att,
                    "avg_speed_kmh": float(np.mean(t1_speeds)) if t1_speeds else 0.0,
                    "max_speed_kmh": float(np.max(t1_speeds)) if t1_speeds else 0.0,
                }
            )

            analytics_timeline.append(
                {"frame_idx": frame_idx, "team_0": metrics_0, "team_1": metrics_1}
            )

    with open(OUTPUT_PATH, "w") as f:
        json.dump(analytics_timeline, f, indent=4)

    print(f"Generated advanced tactical analytics for {len(analytics_timeline)} frames!")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_analytics()
