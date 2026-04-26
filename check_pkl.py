# Save as check_pkl.py in your project root and run it
import pickle

with open("backend/output/psg_inter_coords.pkl", "rb") as f:
    data = pickle.load(f)

frames = data["frames"]
print(f"Total frames: {len(frames)}")

# Check first 5 frames with detections
count = 0
for frame in frames:
    if frame["tracks"]:
        print(f"Frame {frame['frame_idx']}: {len(frame['tracks'])} tracks, ball={frame['ball_xy']}")
        count += 1
    if count >= 5:
        break

# Summary
total_with_tracks = sum(1 for f in frames if f["tracks"])
total_with_ball = sum(1 for f in frames if f["ball_xy"])
print(f"\nFrames with detections: {total_with_tracks}/{len(frames)}")
print(f"Frames with ball: {total_with_ball}/{len(frames)}")