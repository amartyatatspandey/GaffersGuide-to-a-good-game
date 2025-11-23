# import json

# # Path to your real tracking JSONL file
# file_path = "opendata/data/matches/1925299/1925299_tracking_extrapolated.jsonl"

# with open(file_path, "r") as f:
#     for i, line in enumerate(f):
#         frame = json.loads(line)
#         # Count detected players in this frame
#         detected_players = [p for p in frame.get("player_data", []) if p.get("is_detected")]
#         print(f"Frame {i}: {len(detected_players)} detected players")
#         if i >= 20:  # just check first 20 frames
#             break
from model0_load_data import load_tracking, load_match_metadata

frames = load_tracking()
meta = load_match_metadata()

print("---- Checking first frame ----")
for p in frames[0]["player_data"]:
    print(p.keys())
    break

print("\n---- Checking metadata ----")
print(meta.keys())
