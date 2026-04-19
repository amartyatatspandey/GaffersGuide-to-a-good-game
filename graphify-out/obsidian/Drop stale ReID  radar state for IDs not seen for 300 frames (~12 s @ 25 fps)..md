---
source_file: "backend/scripts/pipeline_core/track_teams_reid_hybrid.py"
type: "rationale"
community: "Community 5"
location: "L68"
tags:
  - graphify/rationale
  - graphify/EXTRACTED
  - community/Community_5
---

# Drop stale ReID / radar state for IDs not seen for >300 frames (~12 s @ 25 fps).

## Connections
- [[.cleanup_ghost_ids()]] - `rationale_for` [EXTRACTED]
- [[VisualFingerprint]] - `uses` [INFERRED]

#graphify/rationale #graphify/EXTRACTED #community/Community_5