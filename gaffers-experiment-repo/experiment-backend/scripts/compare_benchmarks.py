from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
try:
    from scipy import stats  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    stats = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare benchmark run against baseline.")
    parser.add_argument("--current", required=True, help="Current benchmark json path.")
    parser.add_argument("--baseline", required=True, help="Baseline benchmark json path.")
    parser.add_argument(
        "--output-json",
        default="output/exp/benchmark_delta_report.json",
        help="Delta report output path.",
    )
    parser.add_argument(
        "--min-improvement-pct",
        type=float,
        default=0.0,
        help="Minimum % improvement required for sec/video-minute metric.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _matrix_by_profile(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = payload.get("matrix", [])
    out: dict[str, dict[str, object]] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "profile" in row:
                out[str(row["profile"])] = row
    return out


def _delta_pct(current: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return ((baseline - current) / baseline) * 100.0


def _bootstrap_ci(values: list[float], samples: int = 500, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(42)
    medians = []
    for _ in range(samples):
        sample = [values[rng.randrange(0, len(values))] for _ in range(len(values))]
        medians.append(statistics.median(sample))
    medians.sort()
    lo_idx = int((alpha / 2.0) * (len(medians) - 1))
    hi_idx = int((1.0 - alpha / 2.0) * (len(medians) - 1))
    return (medians[lo_idx], medians[hi_idx])


def _welch_ttest(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2 or stats is None:
        return None
    res = stats.ttest_ind(a, b, equal_var=False)
    return float(res.pvalue)


def main() -> None:
    args = parse_args()
    current = _read_json(Path(args.current).expanduser().resolve())
    baseline = _read_json(Path(args.baseline).expanduser().resolve())

    current_norm = current.get("normalized", {})
    baseline_norm = baseline.get("normalized", {})
    current_spm = float(current_norm.get("opencv_seconds_per_video_minute", 0.0)) if isinstance(current_norm, dict) else 0.0
    baseline_spm = float(baseline_norm.get("opencv_seconds_per_video_minute", 0.0)) if isinstance(baseline_norm, dict) else 0.0

    matrix_current = _matrix_by_profile(current)
    matrix_baseline = _matrix_by_profile(baseline)
    profile_deltas: dict[str, dict[str, float]] = {}
    for profile, cur in matrix_current.items():
        base = matrix_baseline.get(profile)
        if base is None:
            continue
        cur_spm = float(cur.get("seconds_per_video_minute", 0.0))
        base_spm = float(base.get("seconds_per_video_minute", 0.0))
        cur_fps = float(cur.get("effective_fps", 0.0))
        base_fps = float(base.get("effective_fps", 0.0))
        profile_deltas[profile] = {
            "seconds_per_video_minute_improvement_pct": round(_delta_pct(cur_spm, base_spm), 2),
            "effective_fps_improvement_pct": round(_delta_pct(base_fps, cur_fps), 2),
        }

    top_level_improvement = _delta_pct(current_spm, baseline_spm)
    current_trials = current.get("results", {}).get("opencv", {}).get("elapsed_ms_trials", []) if isinstance(current.get("results", {}), dict) else []
    baseline_trials = baseline.get("results", {}).get("opencv", {}).get("elapsed_ms_trials", []) if isinstance(baseline.get("results", {}), dict) else []
    current_trials_f = [float(v) for v in current_trials] if isinstance(current_trials, list) else []
    baseline_trials_f = [float(v) for v in baseline_trials] if isinstance(baseline_trials, list) else []
    current_ci = _bootstrap_ci(current_trials_f)
    baseline_ci = _bootstrap_ci(baseline_trials_f)
    pvalue = _welch_ttest(current_trials_f, baseline_trials_f)
    passes_improvement = top_level_improvement >= float(args.min_improvement_pct)

    report = {
        "current": str(Path(args.current).expanduser().resolve()),
        "baseline": str(Path(args.baseline).expanduser().resolve()),
        "opencv_seconds_per_video_minute": {
            "current": current_spm,
            "baseline": baseline_spm,
            "improvement_pct": round(top_level_improvement, 2),
            "min_required_pct": float(args.min_improvement_pct),
            "pass": passes_improvement,
        },
        "statistics": {
            "current_trials": len(current_trials_f),
            "baseline_trials": len(baseline_trials_f),
            "current_median_ci_ms": [round(current_ci[0], 2), round(current_ci[1], 2)],
            "baseline_median_ci_ms": [round(baseline_ci[0], 2), round(baseline_ci[1], 2)],
            "welch_ttest_pvalue": None if pvalue is None else round(pvalue, 6),
            "minimum_sample_count_met": len(current_trials_f) >= 3 and len(baseline_trials_f) >= 3,
        },
        "profile_deltas": profile_deltas,
        "current_gates_pass": current.get("gates_pass", {}),
    }

    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote: {output_path}")
    if not passes_improvement:
        raise SystemExit(1)
    if not report["statistics"]["minimum_sample_count_met"]:
        raise SystemExit(1)
    gates = current.get("gates_pass", {})
    if isinstance(gates, dict) and not all(bool(v) for v in gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
