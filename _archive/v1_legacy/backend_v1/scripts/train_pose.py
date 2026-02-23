"""
Train Pitch Calibration Brain: YOLOv8 Nano Pose on Roboflow pitch keypoints.

Uses backend/data/roboflow_pitch (data.yaml). Saves to backend/models/pitch_pose_v2.
"""
from pathlib import Path

from ultralytics import YOLO

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    backend_root = script_dir.parent
    data_yaml = backend_root / "data" / "roboflow_pitch" / "data.yaml"
    project_dir = backend_root / "models"

    model = YOLO("yolov8n-pose.pt")
    model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        device="mps",
        batch=16,
        project=str(project_dir),
        name="pitch_pose_v2",
    )

    print("✅ Training Complete. Model saved at backend/models/pitch_pose_v2/weights/best.pt")
