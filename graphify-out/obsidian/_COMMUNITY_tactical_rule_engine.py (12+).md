---
type: community
cohesion: 0.15
members: 20
---

# tactical_rule_engine.py (12+)

**Cohesion:** 0.15 - loosely connected
**Members:** 20 nodes

## Members
- [[.__init__()_234]] - code - backend/services/tactical_rule_engine.py
- [[.evaluate_team()]] - code - backend/services/tactical_rule_engine.py
- [[Backwards-compatible frame-by-frame evaluator.          The production pipeline]] - rationale - backend/services/tactical_rule_engine.py
- [[Evaluate tactical metrics and produce chunk-level aggregated insights.      Retu]] - rationale - backend/services/tactical_rule_engine.py
- [[Evaluate tactical rules cumulatively across the full metrics timeline.      Prod]] - rationale - backend/services/tactical_rule_engine.py
- [[Extract the numeric value from     Team spent {x}% of the time with this flaw,]] - rationale - backend/tests/test_pipeline.py
- [[RuleEngine]] - code - backend/services/tactical_rule_engine.py
- [[_evidence_text()]] - code - backend/services/tactical_rule_engine.py
- [[_load_json()]] - code - backend/tests/test_pipeline.py
- [[_parse_massive_gap_meters()]] - code - backend/tests/test_pipeline.py
- [[_severity_rank()]] - code - backend/services/tactical_rule_engine.py
- [[evaluate_timeline()]] - code - backend/services/tactical_rule_engine.py
- [[main()_25]] - code - backend/services/tactical_rule_engine.py
- [[run_engine()]] - code - backend/services/tactical_rule_engine.py
- [[tactical_rule_engine.py]] - code - backend/services/tactical_rule_engine.py
- [[test_analytics_math()]] - code - backend/tests/test_pipeline.py
- [[test_fastapi_dry_run()]] - code - backend/tests/test_pipeline.py
- [[test_pipeline.py]] - code - backend/tests/test_pipeline.py
- [[test_rag_synthesis()]] - code - backend/tests/test_pipeline.py
- [[test_rule_engine_logic()]] - code - backend/tests/test_pipeline.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/tactical_rule_engine.py_(12+)
SORT file.name ASC
```

## Connections to other communities
- 6 edges to [[_COMMUNITY_main.py (50+)]]
- 3 edges to [[_COMMUNITY_run_e2e_legacy.py (66+)]]

## Top bridge nodes
- [[evaluate_timeline()]] - degree 8, connects to 2 communities
- [[.evaluate_team()]] - degree 4, connects to 2 communities
- [[run_engine()]] - degree 7, connects to 1 community
- [[test_rule_engine_logic()]] - degree 5, connects to 1 community
- [[_severity_rank()]] - degree 3, connects to 1 community