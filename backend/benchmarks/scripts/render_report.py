"""
benchmarks/scripts/render_report.py — Terminal rendering for BenchmarkReport.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_report(report_path: Path) -> None:
    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        return

    try:
        with report_path.open() as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading report: {e}")
        return

    agg = data.get("aggregate", {})
    hw = data.get("hardware_profile", {})
    vid = data.get("video_profile") or {}
    
    print("\n=== GAFFER'S GUIDE BENCHMARK REPORT ===")
    print(f"Run ID        : {data.get('benchmark_run_id', 'unknown')}")
    print(f"Pipeline Ver  : {data.get('pipeline_version', 'unknown')} ({data.get('timestamp_utc', '')[:10]})")
    print(f"Hardware      : {hw.get('cpu_model', 'unknown')}, {hw.get('ram_total_gb', 0)}GB RAM, {hw.get('device_type', 'cpu').upper()}")
    if vid:
        print(f"Video         : {vid.get('filename')} ({vid.get('duration_s', 0):.1f}s, {vid.get('fps', 0)} FPS, {vid.get('total_frames', 0)} frames)")
    else:
        print("Video         : Not available")

    print("\nSTAGE TIMINGS")
    print("─────────────────────────────────────────────")
    
    e2e_ms = agg.get("end_to_end_ms", 0.0)
    
    stages = [
        ("calibration", agg.get("calibration_ms", 0.0)),
        ("parallel_cv", agg.get("parallel_cv_ms", 0.0)),
        ("chunk_merge", agg.get("chunk_merge_ms", 0.0)),
        ("spatial_math", agg.get("spatial_math_ms", 0.0)),
        ("metrics_timeline", agg.get("metrics_timeline_ms", 0.0)),
        ("event_intelligence", agg.get("event_intelligence_ms", 0.0)),
        ("llm_advice", agg.get("llm_total_ms", 0.0)),
        ("report_assembly", agg.get("report_assembly_ms", 0.0)),
        ("gcs_upload", agg.get("gcs_total_ms", 0.0)),
    ]
    
    for name, ms in stages:
        s = ms / 1000.0
        pct = (ms / e2e_ms * 100.0) if e2e_ms > 0 else 0.0
        print(f"  {name:<20}: {s:>7.1f} s  ({pct:>5.1f}%)")
        
    print("─────────────────────────────────────────────")
    print(f"  END-TO-END          : {e2e_ms / 1000.0:>7.1f} s  (100.0%)")

    print("\nRESOURCE USAGE")
    print(f"  Peak RAM       : {agg.get('peak_ram_mb', 0):,.0f} MB")
    vram = agg.get("peak_vram_mb")
    if vram is not None:
        print(f"  Peak VRAM      : {vram:,.0f} MB")
    else:
        print("  Peak VRAM      : N/A")
    print(f"  Peak CPU       : {agg.get('peak_cpu_pct', 0):>6.1f} %")
    print(f"  Overall FPS    : {agg.get('overall_fps', 0):>6.1f} frames/sec")

    print("\nI/O")
    print(f"  Disk Read      : {agg.get('total_io_read_bytes', 0) / (1024**2):,.0f} MB")
    print(f"  Disk Write     : {agg.get('total_io_write_bytes', 0) / (1024**2):,.0f} MB")
    print(f"  GCS Upload     : {agg.get('gcs_upload_bytes', 0) / (1024**2):,.0f} MB")

    print("\nLLM")
    print(f"  Total calls    : {agg.get('total_llm_calls', 0)}")
    print(f"  Total latency  : {agg.get('total_llm_latency_ms', 0):,.0f} ms")
    
    flags = data.get("regression_flags", [])
    if flags:
        print("\nREGRESSIONS DETECTED")
        for f in flags:
            sev = f.get('severity', '')
            desc = f.get('description', '')
            print(f"  [{sev}] {desc}")
    
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render benchmark report to terminal")
    parser.add_argument("report", type=Path, help="Path to *_report.json")
    args = parser.parse_args()
    render_report(args.report)
