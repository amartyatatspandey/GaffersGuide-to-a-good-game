import json
import asyncio
from pathlib import Path
from backend.services.parallel_pipeline import GlobalRefiner, build_metrics_timeline
from backend.scripts.e2e_shared import TacticalFrame, TacticalPlayer

def test():
    with open("backend/output/85b22806c17d49ceb0e65c15ec016f6e_tracking_data.json") as f:
        data = json.load(f)
    
    raw_frames = []
    for f in data["frames"]:
        players = []
        for p in f["players"]:
            # Ensure p["radar_pt"] is valid
            rp = None
            if "x_pitch" in p and p["x_pitch"] is not None and "y_pitch" in p and p["y_pitch"] is not None:
                rp = [p["x_pitch"], p["y_pitch"]]
            players.append(TacticalPlayer(id=p.get("id"), team=p.get("team_id"), radar_pt=rp))
        
        raw_frames.append(TacticalFrame(
            frame_idx=f["frame_idx"],
            players=players,
            ball_xy=f.get("ball_xy"),
            possession_team_id=f.get("possession_team_id")
        ))
    
    print(f"Loaded {len(raw_frames)} frames.")
    
    refiner = GlobalRefiner()
    refined_frames = refiner.refine(
        raw_frames,
        frame_factory=lambda frame_idx, players: TacticalFrame(
            frame_idx=frame_idx,
            players=players,
            ball_xy=None,
            possession_team_id=None,
        ),
        player_factory=lambda pid, team, radar_pt: TacticalPlayer(
            id=pid, team=team, radar_pt=radar_pt
        ),
    )
    
    print("Refined frames:", len(refined_frames))

test()
