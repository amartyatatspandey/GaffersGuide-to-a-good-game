import cv2
import json
import numpy as np
from pathlib import Path
import logging

LOGGER = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent

def generate_video_overlay(job_id: str) -> bool:
    """Generate the annotated tactical radar video from tracking_data.json."""
    video_path = BACKEND_ROOT / "data" / "uploads" / f"{job_id}.mp4"
    tracking_data = BACKEND_ROOT / "output" / f"{job_id}_tracking_data.json"
    homography = BACKEND_ROOT / "output" / f"{job_id}_homographies.json"
    out_path = BACKEND_ROOT / "output" / f"{job_id}_tracking_overlay.mp4"
    
    if out_path.exists():
        return True
        
    if not video_path.exists() or not tracking_data.exists() or not homography.exists():
        LOGGER.error("Missing required files for video generation.")
        return False
        
    try:
        # Import inside to avoid circular deps
        from scripts.track_teams import TacticalRadar
        
        with open(tracking_data, "r") as f:
            data = json.load(f)
            
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        radar = TacticalRadar(json_path=str(homography), video_res=(width, height))
        
        out_h = max(height, radar.radar_h)
        out_w = width + radar.radar_w
        
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
        
        frames = data.get("frames", [])
        
        for frame_data in frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            radar_img = radar.draw_blank_pitch()
            
            for p in frame_data.get("players", []):
                team = p.get("team_id")
                
                color = (128, 128, 128)
                if team == "team_0":
                    color = (0, 0, 255)
                elif team == "team_1":
                    color = (255, 0, 0)
                    
                rp_x = p.get("x_pitch")
                rp_y = p.get("y_pitch")
                if rp_x is not None and rp_y is not None:
                    cv2.circle(radar_img, (int(rp_x * 10.0), int(rp_y * 10.0)), 5, color, -1)
                    
                cx = p.get("x_canvas")
                cy = p.get("y_canvas")
                if cx is not None and cy is not None:
                    # Draw a small bounding box or circle on the main frame
                    cv2.circle(frame, (int(cx), int(cy)), 6, color, 2)
                    
            composed = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            composed[0:height, 0:width] = frame
            composed[0:radar.radar_h, width:width+radar.radar_w] = radar_img
            
            writer.write(composed)
            
        writer.release()
        cap.release()
        return True
    except Exception as e:
        LOGGER.error(f"Error generating video overlay: {e}")
        return False
