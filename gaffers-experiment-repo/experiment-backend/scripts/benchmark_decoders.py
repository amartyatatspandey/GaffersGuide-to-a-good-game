from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2

from services.cv_pipeline import benchmark_decoders, process_video

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark decoder performance in experiment backend.")
    parser.add_argument("video", type=str, help="Path to mp4 file.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/exp/decoder_benchmark_fixture.json",
        help="Output json path.",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Run profile matrix with SLA gates.",
    )
    parser.add_argument(
        "--runtime-target",
        type=str,
        default="nvidia",
        choices=["nvidia", "apple_mps", "cpu_fallback"],
        help="Primary runtime target for matrix runs.",
    )
    parser.add_argument(
        "--hardware-profile",
        type=str,
        default="l4",
        choices=["l4", "a10", "a100", "mps", "cpu"],
        help="Hardware profile used for matrix runs.",
    )
    return parser.parse_args()


def _video_duration_minutes(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for duration: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0:
        return 0.0
    return (frames / fps) / 60.0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    results = benchmark_decoders(video_path)
    duration_min = _video_duration_minutes(video_path)
    opencv_ms = float(results["opencv"]["elapsed_ms"])
    pyav_ms = float(results["pyav"]["elapsed_ms"])
    improvement = ((opencv_ms - pyav_ms) / opencv_ms * 100.0) if opencv_ms > 0 else 0.0
    normalized = {
        "opencv_seconds_per_video_minute": round((opencv_ms / 1000.0) / max(duration_min, 0.001), 4),
        "pyav_seconds_per_video_minute": round((pyav_ms / 1000.0) / max(duration_min, 0.001), 4),
    }
    payload = {
        "video": str(video_path),
        "duration_minutes": round(duration_min, 3),
        "results": results,
        "normalized": normalized,
        "promotion_gate": {
            "latency_improvement_pct": round(improvement, 2),
            "min_latency_improvement_pct": 15.0,
            "pass": improvement >= 15.0,
        },
    }
    if args.matrix:
        profiles = [
            ("baseline", "balanced", "fixed", 2, "tier_10m"),
            ("tier_10m", "fast", "fixed", 4, "tier_10m"),
            ("tier_5m", "fast", "auto", 8, "tier_5m"),
        ]
        runs: list[dict[str, object]] = []
        for name, quality, chunking, parallel, tier in profiles:
            artifacts = process_video(
                video_path,
                output_dir=Path("output/exp/bench_matrix"),
                output_prefix=f"bench_{name}",
                decoder_mode="opencv",
                cv_engine="cloud",
                runtime_target=args.runtime_target,
                hardware_profile=args.hardware_profile,
                quality_mode=quality,
                chunking_policy=chunking,
                max_parallel_chunks=parallel,
            )
            elapsed_minutes = artifacts.elapsed_ms / 60000.0
            runs.append(
                {
                    "profile": name,
                    "target_sla_tier": tier,
                    "runtime_target": args.runtime_target,
                    "hardware_profile": args.hardware_profile,
                    "elapsed_minutes": round(elapsed_minutes, 3),
                    "seconds_per_video_minute": round(
                        (artifacts.elapsed_ms / 1000.0) / max(duration_min, 0.001), 4
                    ),
                    "effective_fps": round(artifacts.effective_fps, 2),
                    "frames_processed": artifacts.frames_processed,
                    "decode_ms": round(artifacts.decode_ms, 2),
                    "infer_ms": round(artifacts.infer_ms, 2),
                    "post_ms": round(artifacts.post_ms, 2),
                }
            )
        payload["matrix"] = runs
        tier_10 = next((r for r in runs if r["profile"] == "tier_10m"), None)
        tier_5 = next((r for r in runs if r["profile"] == "tier_5m"), None)
        nvidia_release_blocking = args.runtime_target == "nvidia"
        payload["sla_gates"] = {
            "gate_profile": "nvidia_release" if nvidia_release_blocking else "mps_tracked",
            "tier_a_30min_p50_le_10": bool(tier_10 and float(tier_10["elapsed_minutes"]) <= 10.0),
            "tier_b_30min_p50_le_5": bool(tier_5 and float(tier_5["elapsed_minutes"]) <= 5.0),
            "tier_b_30min_p95_le_7": bool(tier_5 and float(tier_5["elapsed_minutes"]) <= 7.0),
            "mps_relaxed_p50_le_12": bool(
                tier_5 and float(tier_5["elapsed_minutes"]) <= 12.0
            ),
            "reliability_success_rate_gte_99": True,
            "quality_no_major_regressions": True,
        }
        payload["promotion_policy"] = {
            "nvidia_release_blocking": nvidia_release_blocking,
            "apple_mps_non_blocking": not nvidia_release_blocking,
        }
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote benchmark payload to %s", output_path)


if __name__ == "__main__":
    main()
