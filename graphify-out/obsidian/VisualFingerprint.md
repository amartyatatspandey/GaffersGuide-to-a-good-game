---
source_file: "backend/scripts/pipeline_core/reid_healer.py"
type: "code"
community: "Community 5"
location: "L354"
tags:
  - graphify/code
  - graphify/EXTRACTED
  - community/Community_5
---

# VisualFingerprint

## Connections
- [[.__init__()_227]] - `method` [EXTRACTED]
- [[.__init__()_218]] - `calls` [INFERRED]
- [[._ensure_model_downloaded()]] - `method` [EXTRACTED]
- [[._load_model()]] - `method` [EXTRACTED]
- [[.extract_features()_3]] - `method` [EXTRACTED]
- [[Drop stale ReID  radar state for IDs not seen for 300 frames (~12 s @ 25 fps).]] - `uses` [INFERRED]
- [[Firewall between ByteTrack IDs and reid_healer optional ReID + radar distance c]] - `uses` [INFERRED]
- [[HybridIDHealer]] - `uses` [INFERRED]
- [[Loads OSNet x0.25 (MSMT17) and extracts L2-normalized 512-D embeddings from play]] - `rationale_for` [EXTRACTED]
- [[Optional ReID + radar ID healing (extracted from track_teams for Rule 4).]] - `uses` [INFERRED]
- [[Optionally rewrites tracker IDs on ``detections`` using ReID + radar proximity.]] - `uses` [INFERRED]
- [[reid_healer.py]] - `contains` [EXTRACTED]

#graphify/code #graphify/EXTRACTED #community/Community_5