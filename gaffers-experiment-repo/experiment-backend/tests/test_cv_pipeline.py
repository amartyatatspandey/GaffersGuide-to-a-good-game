from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.cv_pipeline import process_video  # noqa: E402


def test_process_video_writes_experiment_artifacts(tmp_path: Path) -> None:
    source_video = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "fixtures"
        / "short"
        / "short_czech_england_120s.mp4"
    )
    if not source_video.is_file():
        raise AssertionError("Expected experiment fixture clip to exist.")

    artifacts = process_video(
        source_video,
        output_dir=tmp_path,
        output_prefix="exp_test",
        decoder_mode="opencv",
    )
    assert artifacts.report_path.is_file()
    assert artifacts.tracking_path.is_file()
    assert str(artifacts.report_path).startswith(str(tmp_path))
    assert str(artifacts.tracking_path).startswith(str(tmp_path))
