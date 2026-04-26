import pickle

with open("detections.pkl", "rb") as f:
    frames = pickle.load(f)

print("Frames:", len(frames))
print("Sample:", frames[0]) 
import pickle

with open("detections.pkl", "rb") as f:
    frames = pickle.load(f)

print(f"Total frames: {len(frames)}")

# Check ball detections
total_with_ball = 0
for frame in frames:
    for det in frame:
        if det['class'] == 1:  # ball
            total_with_ball += 1
            break

print(f"Frames with ball: {total_with_ball}/{len(frames)}")

# Show first 5 frames with ball
count = 0
for i, frame in enumerate(frames):
    ball = [d for d in frame if d['class'] == 1]
    if ball:
        print(f"Frame {i}: ball conf={ball[0]['confidence']:.3f} bbox={ball[0]['bbox']}")
        count += 1
    if count >= 5:
        break