# Gaffers Guide Video Assets Handoff

This document explains the full `docs/video_assets` package so your video team can edit the explainer without needing engineering context.

## What This Package Contains

- `SHOT_MANIFEST.md`: master shot plan with filename, source command/code, expected output, and narration mapping.
- `CAPTURE_PROFILE.md`: visual standards used across all screenshots (resolution, colors, typography, spacing).
- `SCREENSHOT_INDEX.md`: ordered list of all screenshots (1 to 21) with relative path and file size.
- `SCREENSHOT_INDEX.json`: machine-readable version of the same index.
- `screenshots/`: final PNG assets grouped by story section.
- `screenshots_bundle.zip`: zipped handoff of the `screenshots/` folder.
- `generate_screenshots.py`: script used to generate the screenshot set reproducibly.

## Story Structure For The Video

Use this sequence in the edit timeline:

1. **Installation** (`screenshots/01_install/`)
2. **Import Proof** (`screenshots/02_imports/`)
3. **Examples + Output** (`screenshots/03_examples_output/`)
4. **CLI Experience** (`screenshots/04_cli/`)
5. **E2E Credibility Evidence** (`screenshots/05_e2e_artifacts/`)

This order matches the manifest and is already aligned to an explainer arc (setup -> usage -> proof).

## Exact Asset Inventory

- Total screenshots: **21**
- All assets are **1920x1080 PNG** and styled consistently.
- Full ordered list is in `SCREENSHOT_INDEX.md`.

## Folder-By-Folder Breakdown

### `screenshots/01_install`

Purpose: establish package onboarding.

- Base install command (`pip install gaffers-guide`)
- Vision install command (`pip install "gaffers-guide[vision]"`)

### `screenshots/02_imports`

Purpose: prove clean imports across the modular SDK.

- `gaffers_guide.spatial`
- `gaffers_guide.io`
- `gaffers_guide.pipeline`
- `gaffers_guide.pipeline.config`

### `screenshots/03_examples_output`

Purpose: show real usage and resulting outputs.

- Spatial mapping code + output
- Tactical IO code + output
- Full engine code + run output

### `screenshots/04_cli`

Purpose: demonstrate CLI contract and usability.

- `gaffers-guide --help`
- `gaffers-guide profiles list`
- `gaffers-guide run --help`
- Invalid profile error handling

### `screenshots/05_e2e_artifacts`

Purpose: provide credibility from real QA evidence.

- QA summary
- Artifact listing
- Tracking JSON preview
- Metrics JSON preview
- Report JSON preview

## How To Use In Edit

- Prefer one shot every 3-6 seconds depending on voiceover density.
- Use gentle zoom-in on long terminal outputs to guide viewer attention.
- Keep transitions simple (cross dissolve or hard cut) to preserve technical clarity.
- For code/output pairs, show code first, then output immediately after.
- For CLI section, pair each shot with short narration of command intent and result.

## Recommended Narration Anchors

Use `SHOT_MANIFEST.md` for line-by-line mapping, but a concise narration flow is:

- Install quickly with base or vision extras.
- Import modules by domain with zero friction.
- Run examples and inspect outputs.
- Use CLI profiles for quality/speed control.
- Validate production credibility through E2E artifacts.

## Quality Notes

- Visual profile is defined in `CAPTURE_PROFILE.md`.
- If you need retakes, regenerate with `generate_screenshots.py` so style remains consistent.
- Keep screenshots uncropped where possible; crop only if necessary for framing in your video template.

## Engineering Contact Context

If the team asks what these images prove:

- They demonstrate import integrity, runnable examples, CLI contract, and artifact outputs.
- The evidence set is based on the same repository and QA cycle used for match-test validation.
