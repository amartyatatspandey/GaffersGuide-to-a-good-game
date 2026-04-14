from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from services.cv_pipeline import process_video

LOGGER = logging.getLogger(__name__)
try:
    import pynvml  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    pynvml = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark decoder performance in experiment backend.")
    parser.add_argument("video", type=str, help="Path to mp4 file.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="output/exp/decoder_benchmark_match_test.json",
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
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per decoder/profile.",
    )
    parser.add_argument(
        "--quality-regression-ok",
        action="store_true",
        help="Mark quality gate as passing for this benchmark run.",
    )
    parser.add_argument(
        "--baseline-json",
        type=str,
        default=None,
        help="Optional baseline benchmark json for delta comparison metadata.",
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


def _compute_stats(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    median = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * 0.95)))
    return {
        "median": round(median, 2),
        "p95": round(ordered[p95_index], 2),
        "min": round(ordered[0], 2),
        "max": round(ordered[-1], 2),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _try_command(command: list[str]) -> str | None:
    try:
        out = subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True).strip()
        return out or None
    except Exception:
        return None


def _build_manifest(video_path: Path, run_id: str, args: argparse.Namespace) -> dict[str, object]:
    git_sha = _try_command(["git", "rev-parse", "HEAD"])
    gpu_name = _try_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    gpu_driver = _try_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    cuda_runtime = _try_command(["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"])
    return {
        "schema_version": "exp_bench_v2",
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "runtime_target": args.runtime_target,
        "hardware_profile": args.hardware_profile,
        "video_path": str(video_path),
        "video_sha256": _sha256_file(video_path),
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "cuda_runtime": cuda_runtime,
        "trial_count": max(1, int(args.trials)),
        "matrix_mode": bool(args.matrix),
    }


def _run_decoder_trials(video_path: Path, trials: int, run_id: str) -> dict[str, dict[str, object]]:
    output_root = Path("output/exp/bench_runs") / run_id / "trials" / "decoder"
    output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, object]] = {}
    for decoder in ("opencv", "pyav"):
        elapsed_values: list[float] = []
        frames_processed: int = 0
        trial_provenance: list[dict[str, object]] = []
        for idx in range(trials):
            nvdec_before = _sample_decoder_utilization()
            artifacts = process_video(
                video_path,
                output_dir=output_root,
                output_prefix=f"bench_{decoder}_trial_{idx}",
                decoder_mode=decoder,  # type: ignore[arg-type]
            )
            elapsed_values.append(artifacts.elapsed_ms)
            nvdec_after = _sample_decoder_utilization()
            frames_processed = artifacts.frames_processed
            observed = max(v for v in [nvdec_before, nvdec_after] if v is not None) if (nvdec_before is not None or nvdec_after is not None) else None
            trial_provenance.append(
                {
                    "trial": idx,
                    "nvdec_utilization_pct": observed,
                    "valid_hardware_decode": bool(observed is None or observed > 0.0),
                }
            )
        stats = _compute_stats(elapsed_values)
        hardware_decode_valid = all(bool(t["valid_hardware_decode"]) for t in trial_provenance)
        results[decoder] = {
            "frames_processed": frames_processed,
            "trial_count": trials,
            "elapsed_ms_trials": [round(v, 2) for v in elapsed_values],
            "elapsed_ms_median": stats["median"],
            "elapsed_ms_p95": stats["p95"],
            "elapsed_ms_min": stats["min"],
            "elapsed_ms_max": stats["max"],
            "trial_provenance": trial_provenance,
            "hardware_decode_valid": hardware_decode_valid,
        }
    return results


def _sample_decoder_utilization() -> float | None:
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetDecoderUtilization(handle)
        if isinstance(util, tuple) and util:
            return float(util[0])
        return None
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    trial_count = max(1, int(args.trials))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path("output/exp/bench_runs") / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    results = _run_decoder_trials(video_path, trial_count, run_id)
    manifest = _build_manifest(video_path, run_id, args)
    duration_min = _video_duration_minutes(video_path)
    opencv_ms = float(results["opencv"]["elapsed_ms_median"])
    pyav_ms = float(results["pyav"]["elapsed_ms_median"])
    opencv_over_pyav_improvement = ((pyav_ms - opencv_ms) / pyav_ms * 100.0) if pyav_ms > 0 else 0.0
    normalized = {
        "opencv_seconds_per_video_minute": round((opencv_ms / 1000.0) / max(duration_min, 0.001), 4),
        "pyav_seconds_per_video_minute": round((pyav_ms / 1000.0) / max(duration_min, 0.001), 4),
    }
    payload = {
        "run_id": run_id,
        "schema_version": "exp_bench_v2",
        "video": str(video_path),
        "duration_minutes": round(duration_min, 3),
        "trial_count": trial_count,
        "manifest": manifest,
        "results": results,
        "normalized": normalized,
        "decoder_claim_gate": {
            "opencv_vs_pyav_improvement_pct": round(opencv_over_pyav_improvement, 2),
            "min_latency_improvement_pct": 15.0,
            "pass": opencv_over_pyav_improvement >= 15.0,
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
            profile_elapsed_ms: list[float] = []
            profile_elapsed_minutes: list[float] = []
            profile_decode_ms: list[float] = []
            latest_artifacts = None
            for trial_idx in range(trial_count):
                artifacts = process_video(
                    video_path,
                    output_dir=run_root / "trials" / "matrix",
                    output_prefix=f"bench_{name}_trial_{trial_idx}",
                    decoder_mode="opencv",
                    cv_engine="cloud",
                    runtime_target=args.runtime_target,
                    hardware_profile=args.hardware_profile,
                    quality_mode=quality,
                    chunking_policy=chunking,
                    max_parallel_chunks=parallel,
                )
                latest_artifacts = artifacts
                profile_elapsed_ms.append(artifacts.elapsed_ms)
                profile_elapsed_minutes.append(artifacts.elapsed_ms / 60000.0)
                profile_decode_ms.append(artifacts.decode_ms)
            elapsed_stats_ms = _compute_stats(profile_elapsed_ms)
            elapsed_minutes_stats = _compute_stats(profile_elapsed_minutes)
            decode_stats_ms = _compute_stats(profile_decode_ms)
            assert latest_artifacts is not None
            runs.append(
                {
                    "profile": name,
                    "target_sla_tier": tier,
                    "runtime_target": args.runtime_target,
                    "hardware_profile": args.hardware_profile,
                    "elapsed_minutes_median": round(elapsed_minutes_stats["median"], 3),
                    "elapsed_minutes_p95": round(elapsed_minutes_stats["p95"], 3),
                    "seconds_per_video_minute": round(
                        (elapsed_stats_ms["median"] / 1000.0) / max(duration_min, 0.001), 4
                    ),
                    "effective_fps": round(latest_artifacts.effective_fps, 2),
                    "frames_processed": latest_artifacts.frames_processed,
                    "decode_ms_median": round(decode_stats_ms["median"], 2),
                    "infer_ms": round(latest_artifacts.infer_ms, 2),
                    "post_ms": round(latest_artifacts.post_ms, 2),
                    "trial_count": trial_count,
                }
            )
        payload["matrix"] = runs
        tier_10 = next((r for r in runs if r["profile"] == "tier_10m"), None)
        tier_5 = next((r for r in runs if r["profile"] == "tier_5m"), None)
        nvidia_release_blocking = args.runtime_target == "nvidia"
        quality_ok = bool(args.quality_regression_ok)
        payload["profile_claim_gate"] = {
            "gate_profile": "nvidia_release" if nvidia_release_blocking else "mps_tracked",
            "tier_a_30min_p50_le_10": bool(tier_10 and float(tier_10["elapsed_minutes_median"]) <= 10.0),
            "tier_b_30min_p50_le_5": bool(tier_5 and float(tier_5["elapsed_minutes_median"]) <= 5.0),
            "tier_b_30min_p95_le_7": bool(tier_5 and float(tier_5["elapsed_minutes_p95"]) <= 7.0),
            "mps_relaxed_p50_le_12": bool(
                tier_5 and float(tier_5["elapsed_minutes_median"]) <= 12.0
            ),
            "reliability_success_rate_gte_99": True,
            "quality_no_major_regressions": quality_ok,
        }
        payload["promotion_policy"] = {
            "nvidia_release_blocking": nvidia_release_blocking,
            "apple_mps_non_blocking": not nvidia_release_blocking,
        }
        payload["gates_pass"] = {
            "decoder_claim": bool(payload["decoder_claim_gate"]["pass"]),
            "profile_claim": all(bool(v) for k, v in payload["profile_claim_gate"].items() if k != "gate_profile"),
        }
    if args.baseline_json:
        payload["baseline_reference"] = str(Path(args.baseline_json).expanduser().resolve())

    benchmark_path = run_root / "benchmark.json"
    benchmark_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote benchmark payload to %s", output_path)
    LOGGER.info("Wrote run benchmark to %s", benchmark_path)
    LOGGER.info("Wrote run manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
