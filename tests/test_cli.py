from __future__ import annotations
import pytest
from gaffers_guide.cli import main


def test_cli_run_balanced(tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()
    out = tmp_path / "output"
    result = main(["run", "--video", str(video), "--output", str(out), "--precision", "balanced"])
    assert result == 0


def test_cli_run_fast(tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()
    out = tmp_path / "output"
    result = main(["run", "--video", str(video), "--output", str(out), "--precision", "fast"])
    assert result == 0


def test_cli_run_sahi(tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()
    out = tmp_path / "output"
    result = main(["run", "--video", str(video), "--output", str(out), "--precision", "sahi"])
    assert result == 0


def test_cli_invalid_precision_rejected(tmp_path):
    video = tmp_path / "test.mp4"
    video.touch()
    out = tmp_path / "output"
    with pytest.raises(SystemExit):
        main(["run", "--video", str(video), "--output", str(out), "--precision", "garbage"])
