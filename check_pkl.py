import pickle

with open("backend/output/psg_inter_coords.pkl", "rb") as f:
    data = pickle.load(f)

frames = data["frames"]
print(f"Total frames: {len(frames)}")

frames_with_ball = sum(1 for f in frames if f["ball_xy"] is not None)
frames_with_players = sum(1 for f in frames if len(f["players_tactical"]) > 0)

print(f"Frames with players: {frames_with_players}/{len(frames)}")
print(f"Frames with ball: {frames_with_ball}/{len(frames)}")

# Show first 5 frames with ball
count = 0
for i, frame in enumerate(frames):
    if frame["ball_xy"] is not None:
        print(f"Frame {frame['frame_idx']}: ball={frame['ball_xy']} players={len(frame['players_tactical'])}")
        count += 1
    if count >= 5:
        break