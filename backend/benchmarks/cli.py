"""
benchmarks/cli.py — CLI entry points for the benchmarking framework.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
import asyncio
import cv2

from services.parallel_pipeline import run_e2e_parallel

from . import config
from .models import BenchmarkReport, VideoProfile
from .regression import compare_reports
from .scripts.render_report import render_report
from .profiler import PerformanceProfiler

def cmd_run(args: argparse.Namespace) -> int:
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file {video_path} not found.", file=sys.stderr)
        return 1

    # Extract basic video properties for the profile
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: Could not read video file.", file=sys.stderr)
        return 1
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    import hashlib
    with open(video_path, "rb") as f:
        file_hash = hashlib.sha256(f.read(1024 * 1024 * 10)).hexdigest()  # Hash first 10MB

    video_profile = VideoProfile(
        filename=video_path.name,
        sha256=file_hash,
        dataset_version=args.dataset_version,
        duration_s=total_frames / fps if fps > 0 else 0.0,
        fps=fps,
        width_px=width,
        height_px=height,
        total_frames=total_frames,
        file_size_bytes=video_path.stat().st_size,
    )

    job_id = f"bench_{video_path.stem}"
    profiler = PerformanceProfiler(job_id=job_id, config=config)
    profiler.set_video_profile(video_profile)
    
    print(f"Starting Benchmark Run: {profiler.run_id}")
    print(f"Target Video: {video_path.name} ({total_frames} frames, {video_profile.duration_s:.1f}s)")
    
    try:
        with profiler.run():
            asyncio.run(
                run_e2e_parallel(
                    video=video_path,
                    output_prefix=job_id,
                    profiler=profiler,
                )
            )
            
        out_dir = Path(args.runs_dir)
        report_path = profiler.write_report(out_dir)
        print(f"\n✅ Benchmark completed successfully.")
        print(f"Report saved to: {report_path}")
        
        # Automatically render it
        render_report(report_path)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ Benchmark failed: {e}", file=sys.stderr)
        return 1


def cmd_designate_baseline(args: argparse.Namespace) -> int:
    run_file = Path(args.runs_dir) / f"{args.run_id}_report.json"
    if not run_file.exists():
        print(f"Error: Could not find report for run {args.run_id} at {run_file}", file=sys.stderr)
        return 1
        
    baseline_dir = Path("backend/benchmarks/baseline")
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baseline_file = baseline_dir / "baseline.json"
    
    shutil.copy2(run_file, baseline_file)
    print(f"Successfully designated {args.run_id} as the new baseline.")
    print(f"Baseline saved to {baseline_file}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    current_file = Path(args.runs_dir) / f"{args.run_id}_report.json"
    baseline_file = Path(args.baseline)
    
    if not current_file.exists():
        print(f"Error: Current run report not found at {current_file}", file=sys.stderr)
        return 1
    if not baseline_file.exists():
        print(f"Error: Baseline report not found at {baseline_file}", file=sys.stderr)
        return 1
        
    try:
        current = BenchmarkReport.model_validate_json(current_file.read_text())
        baseline = BenchmarkReport.model_validate_json(baseline_file.read_text())
    except Exception as e:
        print(f"Error parsing reports: {e}", file=sys.stderr)
        return 1
        
    flags = compare_reports(current, baseline)
    
    if not flags:
        print("✅ No regressions detected against baseline.")
        return 0
        
    print(f"⚠️  Detected {len(flags)} regressions against baseline:")
    has_critical = False
    has_warning = False
    
    for f in flags:
        print(f"  [{f.severity}] {f.description}")
        if f.severity == "CRITICAL":
            has_critical = True
        elif f.severity == "WARNING":
            has_warning = True
            
    # Inject flags back into current run report and save
    current.regression_flags = flags
    current_file.write_text(current.model_dump_json(indent=2))
    
    if args.fail_on == "CRITICAL" and has_critical:
        return 2
    if args.fail_on == "WARNING" and (has_critical or has_warning):
        return 1
        
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    report_file = Path(args.runs_dir) / f"{args.run_id}_report.json"
    render_report(report_file)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Gaffer's Guide V2 Benchmarking CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 1. designate-baseline
    p_baseline = subparsers.add_parser("designate-baseline", help="Set a run as the regression baseline")
    p_baseline.add_argument("--run-id", required=True, help="UUID of the benchmark run")
    p_baseline.add_argument("--runs-dir", default="backend/benchmarks/runs/", help="Directory containing run reports")
    
    # 2. compare
    p_compare = subparsers.add_parser("compare", help="Compare a run against the baseline")
    p_compare.add_argument("--run-id", required=True, help="UUID of the benchmark run to check")
    p_compare.add_argument("--baseline", default="backend/benchmarks/baseline/baseline.json", help="Path to baseline report")
    p_compare.add_argument("--runs-dir", default="backend/benchmarks/runs/", help="Directory containing run reports")
    p_compare.add_argument("--fail-on", choices=["WARNING", "CRITICAL", "NONE"], default="WARNING", help="Exit code threshold")
    
    # 3. render
    p_render = subparsers.add_parser("render", help="Render a run report to the terminal")
    p_render.add_argument("--run-id", required=True, help="UUID of the benchmark run")
    p_render.add_argument("--runs-dir", default="backend/benchmarks/runs/", help="Directory containing run reports")

    # 4. run
    p_run = subparsers.add_parser("run", help="Execute a benchmark on the V1 pipeline")
    p_run.add_argument("--video", required=True, help="Path to the video file")
    p_run.add_argument("--dataset-version", default="custom", help="Dataset identifier")
    p_run.add_argument("--runs-dir", default="backend/benchmarks/runs/", help="Directory to output reports")
    
    args = parser.parse_args()
    
    if args.command == "designate-baseline":
        return cmd_designate_baseline(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "render":
        return cmd_render(args)
    elif args.command == "run":
        return cmd_run(args)
        
    return 1


if __name__ == "__main__":
    sys.exit(main())
