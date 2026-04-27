from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SCREENSHOT_ROOT = ROOT / "docs" / "video_assets" / "screenshots"
PYTHON_BIN = ROOT / ".venv" / "bin" / "python"
CLI_BIN = ROOT / ".venv" / "bin" / "gaffers-guide"

BG = "#0d1117"
FG = "#c9d1d9"
ACCENT = "#58a6ff"
WIDTH = 1920
HEIGHT = 1080
PADDING = 48


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ["Menlo.ttc", "SFMono-Regular.otf", "DejaVuSansMono.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


TITLE_FONT = _font(22)
BODY_FONT = _font(18)


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = proc.stdout.strip()
    return f"$ {' '.join(cmd)}\n{output}\n(exit_code={proc.returncode})"


def render_text_shot(title: str, lines: Iterable[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    y = PADDING
    draw.text((PADDING, y), title, fill=ACCENT, font=TITLE_FONT)
    y += 52
    line_height = 28
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            y += line_height
            continue
        wrapped = []
        current = ""
        for token in line.split(" "):
            trial = f"{current} {token}".strip()
            if draw.textlength(trial, font=BODY_FONT) <= WIDTH - (2 * PADDING):
                current = trial
            else:
                if current:
                    wrapped.append(current)
                current = token
        if current:
            wrapped.append(current)
        for part in wrapped:
            if y > HEIGHT - PADDING:
                break
            draw.text((PADDING, y), part, fill=FG, font=BODY_FONT)
            y += line_height
        if y > HEIGHT - PADDING:
            break
    img.save(output_path)


def latest_qa_dir() -> Path:
    candidates = sorted((ROOT / "backend" / "output").glob("qa_matchtest_*"), reverse=True)
    if not candidates:
        raise FileNotFoundError("No qa_matchtest_* output directory found.")
    return candidates[0]


def capture_install_imports() -> None:
    install_base = run_cmd([str(PYTHON_BIN), "-m", "pip", "install", "gaffers-guide"])
    install_vision = run_cmd([str(PYTHON_BIN), "-m", "pip", "install", "gaffers-guide[vision]"])
    render_text_shot("Base Install", install_base.splitlines(), SCREENSHOT_ROOT / "01_install" / "01_install_01_base_install.png")
    render_text_shot("Vision Install", install_vision.splitlines(), SCREENSHOT_ROOT / "01_install" / "01_install_02_vision_install.png")

    snippets = [
        ("Spatial Import", "from gaffers_guide.spatial import HomographyEngine"),
        ("IO Import", "from gaffers_guide.io import parse_tracking_json"),
        ("Pipeline Import", "from gaffers_guide.pipeline import MatchAnalysisPipeline"),
        ("PipelineConfig Import", "from gaffers_guide.pipeline.config import PipelineConfig"),
    ]
    for idx, (title, snippet) in enumerate(snippets, start=1):
        code = (
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            f"{snippet}\n"
            "print('import_ok')\n"
        )
        out = run_cmd([str(PYTHON_BIN), "-c", code])
        render_text_shot(
            title,
            out.splitlines(),
            SCREENSHOT_ROOT / "02_imports" / f"02_imports_{idx:02d}_{title.lower().replace(' ', '_')}.png",
        )


def capture_examples() -> None:
    spatial_code = [
        "import numpy as np",
        "from gaffers_guide.spatial import HomographyEngine",
        "corners_px = np.array([[120.0, 50.0], [1800.0, 45.0], [1900.0, 1030.0], [80.0, 1035.0]], dtype=np.float64)",
        "engine = HomographyEngine()",
        "mapping = engine.fit(corners_px, frame_shape=(1080, 1920))",
        "pitch_point = mapping.pixel_to_pitch((960.0, 540.0))",
        "print(pitch_point.to_dict())",
    ]
    render_text_shot("Spatial Mapping Code", spatial_code, SCREENSHOT_ROOT / "03_examples_output" / "03_examples_output_01_spatial_code.png")
    spatial_out = run_cmd([str(PYTHON_BIN), "-c", "\n".join(spatial_code)])
    render_text_shot("Spatial Mapping Output", spatial_out.splitlines(), SCREENSHOT_ROOT / "03_examples_output" / "03_examples_output_02_spatial_output.png")

    io_code = [
        "from pathlib import Path",
        "from gaffers_guide.io import parse_tracking_json",
        "tracking = parse_tracking_json(Path('backend/output/fast_tracking_data.json'))",
        "print(type(tracking).__name__)",
        "print(list(tracking.keys())[:4])",
    ]
    render_text_shot("Tactical IO Code", io_code, SCREENSHOT_ROOT / "03_examples_output" / "03_examples_output_03_io_code.png")
    io_out = run_cmd([str(PYTHON_BIN), "-c", "\n".join(io_code)])
    render_text_shot("Tactical IO Output", io_out.splitlines(), SCREENSHOT_ROOT / "03_examples_output" / "03_examples_output_04_io_output.png")

    engine_code = [
        "from pathlib import Path",
        "from gaffers_guide.pipeline import MatchAnalysisPipeline",
        "from gaffers_guide.pipeline.config import PipelineConfig",
        "pipeline = MatchAnalysisPipeline.from_profile('balanced')",
        "report_path = pipeline.process_video(PipelineConfig(video=Path('backend/data/match_test.mp4'), output_dir=Path('backend/output/demo_video_assets'), quality_profile='balanced'))",
        "print(report_path)",
    ]
    render_text_shot("Full Engine Code", engine_code, SCREENSHOT_ROOT / "03_examples_output" / "03_examples_output_05_engine_code.png")
    engine_out = run_cmd([str(CLI_BIN), "run", "--video", str(ROOT / "backend" / "data" / "match_test.mp4"), "--output", str(ROOT / "backend" / "output" / "demo_video_assets"), "--quality-profile", "fast"])
    render_text_shot("Full Engine Output", engine_out.splitlines(), SCREENSHOT_ROOT / "03_examples_output" / "03_examples_output_06_engine_output.png")


def capture_cli_e2e() -> None:
    cli_shots = [
        ("CLI Help", [str(CLI_BIN), "--help"], "04_cli_01_help.png"),
        ("Profiles List", [str(CLI_BIN), "profiles", "list"], "04_cli_02_profiles.png"),
        ("Run Help", [str(CLI_BIN), "run", "--help"], "04_cli_03_run_help.png"),
        (
            "Invalid Profile Error",
            [str(CLI_BIN), "run", "--video", str(ROOT / "backend" / "data" / "match_test.mp4"), "--output", str(ROOT / "backend" / "output" / "invalid_profile_demo"), "--quality-profile", "invalid"],
            "04_cli_04_invalid_profile.png",
        ),
    ]
    for title, cmd, filename in cli_shots:
        out = run_cmd(cmd)
        render_text_shot(title, out.splitlines(), SCREENSHOT_ROOT / "04_cli" / filename)

    qa_dir = latest_qa_dir()
    qa_summary = (qa_dir / "QA_SUMMARY.md").read_text(encoding="utf-8").splitlines()
    render_text_shot("QA Summary", qa_summary, SCREENSHOT_ROOT / "05_e2e_artifacts" / "05_e2e_artifacts_01_qa_summary.png")

    ls_lines = run_cmd(["/bin/ls", "-lh", str(ROOT / "backend" / "output")]).splitlines()
    render_text_shot("Artifact Listing", ls_lines, SCREENSHOT_ROOT / "05_e2e_artifacts" / "05_e2e_artifacts_02_artifact_listing.png")

    with (ROOT / "backend" / "output" / "fast_tracking_data.json").open("r", encoding="utf-8") as f:
        tracking = json.load(f)
    tracking_preview = [
        "fast_tracking_data.json",
        f"top_keys={list(tracking.keys())}",
        f"frames_count={len(tracking.get('frames', []))}",
        f"telemetry_keys={list(tracking.get('telemetry', {}).keys())[:8]}",
    ]
    render_text_shot("Tracking JSON Preview", tracking_preview, SCREENSHOT_ROOT / "05_e2e_artifacts" / "05_e2e_artifacts_03_tracking_json_preview.png")

    with (ROOT / "backend" / "output" / "fast_tactical_metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics_preview = [
        "fast_tactical_metrics.json",
        f"type={type(metrics).__name__}",
        f"entries={len(metrics)}",
        f"sample_keys={list(metrics[0].keys())[:10] if metrics else []}",
    ]
    render_text_shot("Metrics JSON Preview", metrics_preview, SCREENSHOT_ROOT / "05_e2e_artifacts" / "05_e2e_artifacts_04_metrics_json_preview.png")

    with (ROOT / "backend" / "output" / "fast_report.json").open("r", encoding="utf-8") as f:
        report = json.load(f)
    report_preview = [
        "fast_report.json",
        f"type={type(report).__name__}",
        f"cards={len(report)}",
        f"first_card_keys={list(report[0].keys()) if report and isinstance(report[0], dict) else []}",
    ]
    render_text_shot("Report JSON Preview", report_preview, SCREENSHOT_ROOT / "05_e2e_artifacts" / "05_e2e_artifacts_05_report_json_preview.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["install-imports", "examples", "cli-e2e", "all"],
        default="all",
    )
    args = parser.parse_args()
    if args.stage in {"install-imports", "all"}:
        capture_install_imports()
    if args.stage in {"examples", "all"}:
        capture_examples()
    if args.stage in {"cli-e2e", "all"}:
        capture_cli_e2e()


if __name__ == "__main__":
    main()
