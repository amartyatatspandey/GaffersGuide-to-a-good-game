# visualize.py
import pickle
import cv2
from pathlib import Path

# Load detections
with open("detections.pkl", "rb") as f:
    frames = pickle.load(f)

# Load video
video_path = "/Users/nishoosingh/Documents/football_ai/psg_inter.mp4"
cap = cv2.VideoCapture(video_path)

# Output video
output_path = "backend/output/tracked_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

for i, frame_data in enumerate(frames):
    ret, frame = cap.read()
    if not ret:
        break

    # Draw players
    for bbox in frame_data["players"]:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "player", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Draw ball
    if frame_data["ball"] is not None:
        x1, y1, x2, y2 = map(int, frame_data["ball"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, "ball", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Frame counter
    cv2.putText(frame, f"Frame: {i}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    writer.write(frame)

cap.release()
writer.release()
print(f"Done! Saved to {output_path}")