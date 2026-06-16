import numpy as np
from backend.scripts.global_refiner import GlobalRefiner
from backend.scripts.e2e_shared import TacticalFrame, TacticalPlayer

def test():
    refiner = GlobalRefiner()
    
    # 3 frames, 1 player
    frames = [
        TacticalFrame(frame_idx=0, players=[TacticalPlayer(id=1, team="team_0", radar_pt=[0,0])], ball_xy=None, possession_team_id=None),
        TacticalFrame(frame_idx=1, players=[], ball_xy=None, possession_team_id=None),
        TacticalFrame(frame_idx=2, players=[TacticalPlayer(id=1, team="team_0", radar_pt=[10,10])], ball_xy=None, possession_team_id=None),
    ]
    
    def frame_factory(idx, players):
        return TacticalFrame(frame_idx=idx, players=players, ball_xy=None, possession_team_id=None)
    
    def player_factory(pid, team, pt):
        return TacticalPlayer(id=pid, team=team, radar_pt=pt)
        
    res = refiner.refine(frames, frame_factory=frame_factory, player_factory=player_factory)
    print("Success! refined frames:", len(res))

test()
