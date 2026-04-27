from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from gaffers_guide import cli


def test_run_command_parses_and_invokes_pipeline() -> None:
    with patch("gaffers_guide.cli.TacticalPipeline") as pipeline_cls:
        exit_code = cli.main(
            [
                "run",
                "--video",
                "test.mp4",
                "--output",
                "out",
                "--quality-profile",
                "fast",
            ]
        )

    assert exit_code == 0
    pipeline_cls.assert_called_once()
    run_call = pipeline_cls.return_value.run.call_args
    assert run_call is not None
    assert run_call.kwargs["video"] == Path("test.mp4")
    assert run_call.kwargs["output"] == Path("out")


def test_invalid_profile_is_rejected_by_parser() -> None:
    parser = cli._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--video",
                "test.mp4",
                "--output",
                "out",
                "--quality-profile",
                "ultra",
            ]
        )
