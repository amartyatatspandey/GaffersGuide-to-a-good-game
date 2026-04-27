# Library Explainer Shot Manifest

## Capture Settings
- Theme: dark terminal style
- Frame size: 1920x1080
- Font: monospace (18px body, 22px title)
- Naming: `<section>_<order>_<slug>.png`

## Shot List
| Filename | Section | Content Source | Expected Visible Output | Narration Mapping |
|---|---|---|---|---|
| `01_install_01_base_install.png` | Installation | `pip install gaffers-guide` | Command plus install log excerpt | "Start with lightweight base install." |
| `01_install_02_vision_install.png` | Installation | `pip install "gaffers-guide[vision]"` | Command plus dependency resolution excerpt | "Upgrade to full vision stack when needed." |
| `02_imports_01_spatial_import.png` | Imports | `from gaffers_guide.spatial import HomographyEngine` | Successful import confirmation | "Spatial module imports cleanly." |
| `02_imports_02_io_import.png` | Imports | `from gaffers_guide.io import parse_tracking_json` | Successful import confirmation | "IO parser module is available." |
| `02_imports_03_pipeline_import.png` | Imports | `from gaffers_guide.pipeline import MatchAnalysisPipeline` | Successful import confirmation | "Pipeline module is import-ready." |
| `02_imports_04_pipeline_config_import.png` | Imports | `from gaffers_guide.pipeline.config import PipelineConfig` | Successful import confirmation | "Pipeline config API is exposed." |
| `03_examples_output_01_spatial_code.png` | Examples | README spatial snippet | Code block | "Spatial mapping quickstart code." |
| `03_examples_output_02_spatial_output.png` | Examples | README spatial snippet execution | `pitch_point` dict output | "Spatial mapping returns pitch coordinates." |
| `03_examples_output_03_io_code.png` | Examples | README tactical IO snippet | Code block | "Tracking parser quickstart code." |
| `03_examples_output_04_io_output.png` | Examples | Tactical IO snippet execution | Parsed keys/type preview | "Parser returns structured payload." |
| `03_examples_output_05_engine_code.png` | Examples | README full engine snippet (adapted path) | Code block | "Full pipeline invocation entrypoint." |
| `03_examples_output_06_engine_output.png` | Examples | Full engine run output (artifact path) | Final report path | "Pipeline writes tactical report artifact." |
| `04_cli_01_help.png` | CLI | `gaffers-guide --help` | Command usage text | "CLI entrypoint and commands." |
| `04_cli_02_profiles.png` | CLI | `gaffers-guide profiles list` | Profile list (`fast`, `balanced`, `high_res`, `sahi`) | "Built-in quality profiles." |
| `04_cli_03_run_help.png` | CLI | `gaffers-guide run --help` | Run command arguments | "Run contract and flags." |
| `04_cli_04_invalid_profile.png` | CLI | invalid profile example | Parser error message | "Clear validation and error handling." |
| `05_e2e_artifacts_01_qa_summary.png` | E2E Evidence | `QA_SUMMARY.md` | PASS verdict and profile matrix | "All profiles validated on match test clip." |
| `05_e2e_artifacts_02_artifact_listing.png` | E2E Evidence | `backend/output/*` listing | Tracking, metrics, report JSON files | "Artifacts are generated per profile." |
| `05_e2e_artifacts_03_tracking_json_preview.png` | E2E Evidence | `fast_tracking_data.json` | top-level keys preview | "Tracking timeline structure." |
| `05_e2e_artifacts_04_metrics_json_preview.png` | E2E Evidence | `fast_tactical_metrics.json` | list length + frame sample | "Metrics timeline structure." |
| `05_e2e_artifacts_05_report_json_preview.png` | E2E Evidence | `fast_report.json` | cards list preview | "Final tactical report structure." |

## Notes
- The engine code shot uses `backend/data/match_test.mp4` and a scratch output path for reproducibility.
- The install screenshots are command-focused excerpts to keep frames readable.
