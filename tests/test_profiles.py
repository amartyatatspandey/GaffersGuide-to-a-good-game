from __future__ import annotations

import pytest

from gaffers_guide.profiles import resolve_profile


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "fast",
            {
                "name": "fast",
                "imgsz": 480,
                "conf_threshold": 0.35,
                "sahi_enabled": False,
                "sahi_slice_size": 320,
                "sahi_overlap_ratio": 0.1,
                "frame_skip": 3,
                "batch_size": 16,
            },
        ),
        (
            "balanced",
            {
                "name": "balanced",
                "imgsz": 640,
                "conf_threshold": 0.25,
                "sahi_enabled": False,
                "sahi_slice_size": 320,
                "sahi_overlap_ratio": 0.2,
                "frame_skip": 1,
                "batch_size": 8,
            },
        ),
        (
            "high_res",
            {
                "name": "high_res",
                "imgsz": 1280,
                "conf_threshold": 0.20,
                "sahi_enabled": False,
                "sahi_slice_size": 512,
                "sahi_overlap_ratio": 0.2,
                "frame_skip": 1,
                "batch_size": 4,
            },
        ),
        (
            "sahi",
            {
                "name": "sahi",
                "imgsz": 1280,
                "conf_threshold": 0.20,
                "sahi_enabled": True,
                "sahi_slice_size": 512,
                "sahi_overlap_ratio": 0.25,
                "frame_skip": 1,
                "batch_size": 2,
            },
        ),
    ],
)
def test_profile_config_exact_values(name: str, expected: dict[str, object]) -> None:
    profile = resolve_profile(name)
    assert profile.name == expected["name"]
    assert profile.imgsz == expected["imgsz"]
    assert profile.conf_threshold == expected["conf_threshold"]
    assert profile.sahi_enabled == expected["sahi_enabled"]
    assert profile.sahi_slice_size == expected["sahi_slice_size"]
    assert profile.sahi_overlap_ratio == expected["sahi_overlap_ratio"]
    assert profile.frame_skip == expected["frame_skip"]
    assert profile.batch_size == expected["batch_size"]


def test_unknown_profile_raises_value_error() -> None:
    with pytest.raises(ValueError):
        resolve_profile("ultra")
