---
type: community
cohesion: 0.10
members: 25
---

# Community 29

**Cohesion:** 0.10 - loosely connected
**Members:** 25 nodes

## Members
- [[.estimate_from_extremities()]] - code - backend/calibration/ports.py
- [[.run()_1]] - code - backend/services/cv_router.py
- [[BallFrameLike]] - code - backend/calculators/ball_visibility.py
- [[CVRunner]] - code - backend/services/cv_router.py
- [[Callable port for line-based homography from extremities (pitch plane - image).]] - rationale - backend/calibration/ports.py
- [[Check whether ball is in a team's defensive third on 1050x680 radar.]] - rationale - backend/calculators/advanced_ball_metrics.py
- [[Compute advanced ball-dependent tactical metrics from refined frame stream.]] - rationale - backend/calculators/advanced_ball_metrics.py
- [[Euclidean distance from ball to nearest player on a team for one frame.]] - rationale - backend/calculators/advanced_ball_metrics.py
- [[FrameLike]] - code - backend/calculators/advanced_ball_metrics.py
- [[FrameLike_1]] - code - backend/calculators/possession.py
- [[HomographyEstimatorPort]] - code - backend/calibration/ports.py
- [[Narrow protocols for pitch  homography (Rule 4.1).  Vendor ``SoccerPitch`` from]] - rationale - backend/calibration/ports.py
- [[PlayerLike]] - code - backend/calculators/advanced_ball_metrics.py
- [[PlayerLike_1]] - code - backend/calculators/possession.py
- [[Protocol]] - code
- [[Return positive forward progression for a team along radar X.]] - rationale - backend/calculators/advanced_ball_metrics.py
- [[_nearest_player_distance()]] - code - backend/calculators/advanced_ball_metrics.py
- [[advanced_ball_metrics.py]] - code - backend/calculators/advanced_ball_metrics.py
- [[ball_visibility.py]] - code - backend/calculators/ball_visibility.py
- [[compute_advanced_ball_metrics()]] - code - backend/calculators/advanced_ball_metrics.py
- [[is_defensive_third()]] - code - backend/calculators/advanced_ball_metrics.py
- [[ports.py]] - code - backend/calibration/ports.py
- [[possession.py]] - code - backend/calculators/possession.py
- [[team_forward_progress()]] - code - backend/calculators/advanced_ball_metrics.py
- [[team_to_id()]] - code - backend/calculators/possession.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/Community_29
SORT file.name ASC
```

## Connections to other communities
- 8 edges to [[_COMMUNITY_Community 1]]
- 4 edges to [[_COMMUNITY_Community 5]]
- 2 edges to [[_COMMUNITY_Community 8]]
- 1 edge to [[_COMMUNITY_Community 11]]
- 1 edge to [[_COMMUNITY_Community 3]]

## Top bridge nodes
- [[Protocol]] - degree 9, connects to 2 communities
- [[CVRunner]] - degree 4, connects to 2 communities
- [[compute_advanced_ball_metrics()]] - degree 9, connects to 1 community
- [[advanced_ball_metrics.py]] - degree 9, connects to 1 community
- [[possession.py]] - degree 5, connects to 1 community