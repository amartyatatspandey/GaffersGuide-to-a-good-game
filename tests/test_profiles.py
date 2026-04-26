from __future__ import annotations
import pytest
from gaffers_guide.profiles import resolve_profile, VALID_PROFILES


def test_all_valid_profiles_resolve():
    for name in VALID_PROFILES:
        profile = resolve_profile(name)
        assert profile.name == name


def test_invalid_profile_raises():
    with pytest.raises(ValueError, match="Unknown quality profile"):
        resolve_profile("nonexistent")


def test_sahi_only_enabled_for_sahi_profile():
    assert resolve_profile("fast").sahi_enabled is False
    assert resolve_profile("balanced").sahi_enabled is False
    assert resolve_profile("high_res").sahi_enabled is False
    assert resolve_profile("sahi").sahi_enabled is True


def test_fast_skips_frames():
    assert resolve_profile("fast").frame_skip > 1


def test_balanced_no_frame_skip():
    assert resolve_profile("balanced").frame_skip == 1


def test_fast_lower_imgsz_than_high_res():
    assert resolve_profile("fast").imgsz < resolve_profile("high_res").imgsz


def test_pipeline_runner_maps_profile_to_sahi_config():
    pytest.importorskip("numpy")
    pytest.importorskip("supervision")
    from backend.services.cv.pipeline_runner import build_sahi_config_from_profile
    profile = resolve_profile("sahi")
    config = build_sahi_config_from_profile(profile)
    assert config.enabled is True
    assert config.conf == profile.conf_threshold
    assert config.slice_w == profile.sahi_slice_size

    profile_fast = resolve_profile("fast")
    config_fast = build_sahi_config_from_profile(profile_fast)
    assert config_fast.enabled is False
