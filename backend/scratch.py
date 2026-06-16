import sys
from pathlib import Path
import os

# Add backend to path
sys.path.insert(0, str(Path().resolve()))

from scripts.run_calibrator_on_video import _calibrate_chunk

if __name__ == '__main__':
    video_path = "/Users/nishoosingh/Documents/GaffersGuide-to-a-good-game/backend/data/match_test.mp4"
    # Get the actual video the user uploaded
    videos = list(Path("output").glob("*.mp4"))
    if videos:
        video_path = str(videos[0].absolute())
        
    print(f"Testing on {video_path}")
    weights_dir = Path("references/sn-calibration/resources").absolute()
    
    # We will test only the first 60 frames to be fast
    try:
        res = _calibrate_chunk(
            video_path=str(video_path),
            weights_dir=str(weights_dir),
            use_advanced_calibration=True,
            sample_every=30,
            start_frame=0,
            end_frame=60,
            chunk_idx=0,
            progress_dict={}
        )
        print("Done testing! Extracted:", len(res))
    except Exception as e:
        import traceback
        traceback.print_exc()
