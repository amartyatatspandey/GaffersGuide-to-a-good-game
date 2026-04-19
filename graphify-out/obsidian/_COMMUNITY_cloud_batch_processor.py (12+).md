---
type: community
cohesion: 0.04
members: 70
---

# cloud_batch_processor.py (12+)

**Cohesion:** 0.04 - loosely connected
**Members:** 70 nodes

## Members
- [[.run()_3]] - code - backend/services/cv_router.py
- [[.run()_2]] - code - backend/services/cv_router.py
- [[Asyncio entry point used by run_e2e_cloud.]] - rationale - backend/tests/test_e2e_llm_local.py
- [[Clone Soccer_Analysis into referencesexternal, or pull if already present.]] - rationale - backend/scripts/auxiliary_tools/setup_references.py
- [[CloudCVRunner]] - code - backend/services/cv_router.py
- [[Create referencesexternal and modelspretrained if they do not exist.]] - rationale - backend/scripts/auxiliary_tools/setup_references.py
- [[Expected ~3 min @ 25 fps → 4501 packets (per ffprobe -count_packets).]] - rationale - backend/tests/test_match_test_asset.py
- [[Local Ollama completions for the CV→…→RAG E2E pipeline (job-time LLM).]] - rationale - backend/scripts/auxiliary_tools/e2e_llm_local.py
- [[LocalCVRunner]] - code - backend/services/cv_router.py
- [[Per-match file {video_stem}_homographies.json under GAFFERS_HOMOGRAPHY_DIR or o]] - rationale - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[Re-encode to H.264 (video) + AAC (audio), the most reliable QuickTime combo.]] - rationale - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[Reference setup for transfer learning clone Soccer_Analysis repo and extract .p]] - rationale - backend/scripts/auxiliary_tools/setup_references.py
- [[Remove `list` query parameter to avoid yt-dlp playlist expansion.]] - rationale - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[Return (ok, reason). Requires readable JSON with non-empty homographies list]] - rationale - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[Return all .pt files under root (YOLO weight files).]] - rationale - backend/scripts/auxiliary_tools/setup_references.py
- [[Run E2E for one video with GAFFERS_HOMOGRAPHY_JSON set to that match's calibrati]] - rationale - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[Run Ollama completions for each synthesized prompt (mirrors cloud ``run_llm`` sh]] - rationale - backend/scripts/auxiliary_tools/e2e_llm_local.py
- [[Run directory setup, clonepull repo, and extract .pt weights.]] - rationale - backend/scripts/auxiliary_tools/setup_references.py
- [[Sanity checks for ``backenddatamatch_test.mp4`` (workspace  pipeline dev clip]] - rationale - backend/tests/test_match_test_asset.py
- [[Scan cloned repo for .pt files and copy them to backendmodelspretrained.     R]] - rationale - backend/scripts/auxiliary_tools/setup_references.py
- [[Tests for Ollama preflight, auto-start policy, and error codes.]] - rationale - backend/tests/test_ollama_client.py
- [[Tests for coachadvice job mode with local LLM refresh.]] - rationale - backend/tests/test_coach_advice_job_local.py
- [[Tests for job-time local LLM completion helper (Ollama path).]] - rationale - backend/tests/test_e2e_llm_local.py
- [[Unset OLLAMA_AUTO_START on a non-Cloud host should still attempt spawn on connec]] - rationale - backend/tests/test_ollama_client.py
- [[_copy_artifact()]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[_homography_path_for_video()]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[_log_skip_homography()]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[_run_single_video()]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[_safe_stem()]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[_sample_record()]] - code - backend/tests/test_e2e_llm_local.py
- [[_unique_target_path()]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[_validate_homography_json()_1]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[clean_youtube_url_remove_playlist_params()]] - code - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[clone_or_pull_repo()]] - code - backend/scripts/auxiliary_tools/setup_references.py
- [[cloud_batch_processor.py]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[copy_pt_to_pretrained()]] - code - backend/scripts/auxiliary_tools/setup_references.py
- [[cv_router.py]] - code - backend/services/cv_router.py
- [[download_eda_matches.py]] - code - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[download_match_segment()]] - code - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[e2e_llm_local.py]] - code - backend/scripts/auxiliary_tools/e2e_llm_local.py
- [[ensure_directories()]] - code - backend/scripts/auxiliary_tools/setup_references.py
- [[ensure_quicktime_compatible()]] - code - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[find_pt_files()]] - code - backend/scripts/auxiliary_tools/setup_references.py
- [[get()]] - code - backend/services/cv_router.py
- [[main()_16]] - code - backend/scripts/auxiliary_tools/cloud_batch_processor.py
- [[main()_23]] - code - backend/scripts/auxiliary_tools/download_eda_matches.py
- [[main()_17]] - code - backend/scripts/auxiliary_tools/setup_references.py
- [[object]] - code
- [[run_llm_local()]] - code - backend/scripts/auxiliary_tools/e2e_llm_local.py
- [[setup_references.py]] - code - backend/scripts/auxiliary_tools/setup_references.py
- [[test_coach_advice_job_local.py]] - code - backend/tests/test_coach_advice_job_local.py
- [[test_default_auto_start_off_on_cloud_when_env_unset()]] - code - backend/tests/test_ollama_client.py
- [[test_default_auto_start_when_env_unset_and_not_cloud()]] - code - backend/tests/test_ollama_client.py
- [[test_e2e_llm_local.py]] - code - backend/tests/test_e2e_llm_local.py
- [[test_ensure_ollama_auto_start_spawns_and_retries()]] - code - backend/tests/test_ollama_client.py
- [[test_ensure_ollama_cloud_run_no_spawn_without_in_cloud_flag()]] - code - backend/tests/test_ollama_client.py
- [[test_ensure_ollama_cloud_run_spawns_when_in_cloud_flag()]] - code - backend/tests/test_ollama_client.py
- [[test_ensure_ollama_connect_no_binary_raises_not_installed()]] - code - backend/tests/test_ollama_client.py
- [[test_ensure_ollama_tags_http_error_offline()]] - code - backend/tests/test_ollama_client.py
- [[test_job_advice_refreshes_with_ollama_when_local_engine()]] - code - backend/tests/test_coach_advice_job_local.py
- [[test_job_advice_skips_refresh_when_cloud_engine()]] - code - backend/tests/test_coach_advice_job_local.py
- [[test_lifecycle_start_noop_when_tags_already_ok()]] - code - backend/tests/test_ollama_client.py
- [[test_match_test_asset.py]] - code - backend/tests/test_match_test_asset.py
- [[test_match_test_mp4_non_trivial_size()]] - code - backend/tests/test_match_test_asset.py
- [[test_match_test_mp4_packet_count_matches_expected()]] - code - backend/tests/test_match_test_asset.py
- [[test_ollama_client.py]] - code - backend/tests/test_ollama_client.py
- [[test_run_llm_local_empty_prompt_skips_llm()]] - code - backend/tests/test_e2e_llm_local.py
- [[test_run_llm_local_engine_routing_error_on_card()]] - code - backend/tests/test_e2e_llm_local.py
- [[test_run_llm_local_maps_completions()]] - code - backend/tests/test_e2e_llm_local.py
- [[test_run_llm_local_sync_wrapper()]] - code - backend/tests/test_e2e_llm_local.py

## Live Query (requires Dataview plugin)

```dataview
TABLE source_file, type FROM #community/cloud_batch_processor.py_(12+)
SORT file.name ASC
```

## Connections to other communities
- 21 edges to [[_COMMUNITY_main.py (50+)]]
- 15 edges to [[_COMMUNITY_run_e2e_legacy.py (66+)]]
- 4 edges to [[_COMMUNITY_model_complexity.py (25+)]]
- 4 edges to [[_COMMUNITY_transforms.py (47+)]]
- 2 edges to [[_COMMUNITY_engine.py (23+)]]
- 1 edge to [[_COMMUNITY_extract_tactical_library_from_pdfs.py (10+)]]
- 1 edge to [[_COMMUNITY_sampler.py (19+)]]
- 1 edge to [[_COMMUNITY_dataset.py (36+)]]
- 1 edge to [[_COMMUNITY_densenet.py (23+)]]

## Top bridge nodes
- [[object]] - degree 18, connects to 7 communities
- [[.run()_3]] - degree 26, connects to 3 communities
- [[cv_router.py]] - degree 5, connects to 2 communities
- [[test_job_advice_refreshes_with_ollama_when_local_engine()]] - degree 4, connects to 2 communities
- [[test_job_advice_skips_refresh_when_cloud_engine()]] - degree 4, connects to 2 communities