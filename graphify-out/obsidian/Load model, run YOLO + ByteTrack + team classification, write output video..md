---
source_file: "backend/scripts/pipeline_core/track_teams.py"
type: "rationale"
community: "Community 5"
location: "L159"
tags:
  - graphify/rationale
  - graphify/INFERRED
  - community/Community_5
---

# Load model, run YOLO + ByteTrack + team classification, write output video.

## Connections
- [[HybridIDHealer]] - `uses` [INFERRED]
- [[TacticalRadar]] - `uses` [INFERRED]
- [[TeamClassifier]] - `uses` [INFERRED]
- [[main()_10]] - `rationale_for` [EXTRACTED]

#graphify/rationale #graphify/INFERRED #community/Community_5