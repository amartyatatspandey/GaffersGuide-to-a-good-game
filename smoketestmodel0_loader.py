
from models.model0_load_data import load_match
m = list(__import__('pathlib').pathlib.Path("opendata/data/matches").iterdir())[0]  # pick first folder
match_id = m.name
print("Testing match_id:", match_id)
frames, meta, events, phases = load_match(match_id)
print("frames type:", type(frames), "len:", len(frames) if hasattr(frames,'__len__') else "unknown")
print("meta keys:", list(meta.keys())[:10])
print("events shape:", getattr(events, "shape", None))
print("phases shape:", getattr(phases, "shape", None))

