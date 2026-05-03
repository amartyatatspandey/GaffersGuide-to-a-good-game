from __future__ import annotations

import logging
from pathlib import Path

from gaffers_guide.profiles import ProfileConfig
from .optimized_sahi_wrapper import ContextAwareSAHIConfig, OptimizedSAHIWrapper

log = logging.getLogger(__name__)


def build_sahi_config_from_profile(profile: ProfileConfig) -> ContextAwareSAHIConfig:
    """
    Map a ProfileConfig to a ContextAwareSAHIConfig.
    Single place where profile fields become runtime knobs.
    """
    return ContextAwareSAHIConfig(
        enabled=profile.sahi_enabled,
        conf=profile.conf_threshold,
        high_conf_skip_threshold=profile.conf_threshold,
        slice_w=profile.sahi_slice_size,
        slice_h=profile.sahi_slice_size,
        overlap_ratio=profile.sahi_overlap_ratio,
        max_slices_per_frame=4,
        temporal_radius_px=112,
        temporal_max_radius_px=360,
        temporal_expand_step_px=24,
    )


def run_pipeline(
    video: Path,
    output: Path,
    profile: ProfileConfig,
) -> int:
    """
    Entry point for the full pipeline.
    Resolves profile into runtime config and passes it downstream.
    """
    log.info("=== Pipeline starting ===")
    log.info("Active profile  : %s", profile.name)
    log.info("Video           : %s", video)
    log.info("Output          : %s", output)
    log.info("Profile detail  : %s", profile)

    sahi_config = build_sahi_config_from_profile(profile)

    log.info("SAHI enabled    : %s", sahi_config.enabled)
    log.info("Confidence      : %s", sahi_config.conf)
    log.info("Slice size      : %sx%s", sahi_config.slice_w, sahi_config.slice_h)
    log.info("Overlap ratio   : %s", sahi_config.overlap_ratio)
    log.info("Frame skip      : %s", profile.frame_skip)
    log.info("Image size      : %s", profile.imgsz)

    # TODO: initialize model and call OptimizedSAHIWrapper with sahi_config
    # model = YOLO(...)
    # wrapper = OptimizedSAHIWrapper(model, ball_class_ids=[0],
    #     config=sahi_config, device=None, use_half=False)

    log.info("=== Pipeline config resolved successfully ===")
    return 0
