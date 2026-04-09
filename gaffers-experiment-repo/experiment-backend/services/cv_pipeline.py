from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import cv2
import numpy as np

from services.dense_pass import run_dense_pass
from services.fast_pass import run_fast_pass
from services.gpu_runtime import select_gpu_runtime
from services.merge import merge_chunk_rows
from services.reid_budget import run_reid_budget_controller
from services.splitter import ChunkSpec, build_chunks
from services.window_selector import select_windows

try:
    import av
except Exception:  # noqa: BLE001
    av = None  # type: ignore[assignment]

DecoderMode = Literal["opencv", "pyav"]


@dataclass(slots=True)
class PipelineArtifacts:
    report_path: Path
    tracking_path: Path
    decoder_used: DecoderMode
    elapsed_ms: float
    frames_processed: int
    decode_ms: float
    infer_ms: float
    post_ms: float
    reid_invocations: int
    reid_ms: float
    id_switch_rate: float
    effective_fps: float
    chunks: list[dict[str, object]]


def _iter_opencv_frames(video_path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _iter_pyav_frames(video_path: Path) -> Iterator[np.ndarray]:
    if av is None:
        raise RuntimeError("PyAV is not installed.")
    container = av.open(str(video_path))
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        container.close()
        raise RuntimeError("No video stream found.")
    try:
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="bgr24")
    finally:
        container.close()


def _iter_frames(video_path: Path, decoder_mode: DecoderMode) -> tuple[Iterator[np.ndarray], DecoderMode]:
    if decoder_mode == "pyav":
        try:
            return _iter_pyav_frames(video_path), "pyav"
        except Exception:  # noqa: BLE001
            return _iter_opencv_frames(video_path), "opencv"
    return _iter_opencv_frames(video_path), "opencv"


def process_video(
    video_path: Path,
    *,
    output_dir: Path,
    output_prefix: str,
    decoder_mode: DecoderMode,
    cv_engine: str = "local",
    runtime_target: str = "nvidia",
    hardware_profile: str = "l4",
    quality_mode: str = "balanced",
    chunking_policy: str = "fixed",
    max_parallel_chunks: int = 2,
) -> PipelineArtifacts:
    start = time.perf_counter()
    decode_start = time.perf_counter()
    frame_iter, decoder_used = _iter_frames(video_path, decoder_mode)
    frames = list(frame_iter)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    frames_processed = len(frames)

    if frames_processed == 0:
        raise RuntimeError("No frames decoded from video.")

    runtime_cfg = select_gpu_runtime(
        runtime_target=runtime_target,
        hardware_profile=hardware_profile,
        cv_engine=cv_engine,
    )
    fps = 25
    chunks = build_chunks(
        total_frames=frames_processed,
        chunking_policy=chunking_policy,
        fps=fps,
    )

    async_rows: list[list[dict[str, object]]] = []
    total_fast_ms = 0.0
    total_infer_ms = 0.0
    total_post_ms = 0.0

    def _run_chunk(chunk: ChunkSpec) -> tuple[list[dict[str, object]], float, float, float]:
        chunk_frames = frames[chunk.start_frame : chunk.end_frame + 1]
        fast = run_fast_pass(chunk_frames, quality_mode=quality_mode)
        windows = select_windows(
            fast.event_frames,
            total_frames=len(chunk_frames),
            fps=fps,
            quality_mode=quality_mode,
        )
        dense = run_dense_pass(
            chunk_frames,
            windows=windows,
            runtime_backend=runtime_cfg.backend,
        )
        adjusted_rows: list[dict[str, object]] = []
        for row in dense.tracked_rows:
            r = dict(row)
            r["frame_idx"] = int(r["frame_idx"]) + chunk.start_frame
            r["chunk_id"] = chunk.chunk_id
            adjusted_rows.append(r)
        return adjusted_rows, fast.elapsed_ms, dense.infer_ms, dense.post_ms

    # Chunk-parallel processing with bounded worker fanout.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(max_parallel_chunks))
    ) as executor:
        futures = [executor.submit(_run_chunk, c) for c in chunks]
        for fut in futures:
            rows, fast_ms, infer_ms_chunk, post_ms_chunk = fut.result()
            async_rows.append(rows)
            total_fast_ms += fast_ms
            total_infer_ms += infer_ms_chunk
            total_post_ms += post_ms_chunk

    merged = merge_chunk_rows(async_rows)
    tracked_rows = merged.frames
    reid = run_reid_budget_controller(
        frames_processed=frames_processed,
        quality_mode=quality_mode,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    tracking_path = output_dir / f"{output_prefix}_tracking_data.json"
    report_path = output_dir / f"{output_prefix}_report.json"

    tracking_payload = {
        "video_path": str(video_path),
        "telemetry": {
            "total_frames_processed": frames_processed,
            "decode_ms": round(decode_ms, 2),
            "fast_pass_ms": round(total_fast_ms, 2),
            "infer_ms": round(total_infer_ms, 2),
            "post_ms": round(total_post_ms, 2),
            "runtime_backend": runtime_cfg.backend,
            "chunk_count": len(chunks),
        },
        "frames": tracked_rows,
    }
    tracking_path.write_text(json.dumps(tracking_payload, indent=2), encoding="utf-8")

    report_payload = [
        {
            "frame_idx": 0,
            "team": "team_0",
            "flaw": "Experimental spacing signal",
            "severity": "medium",
            "evidence": (
                f"Processed {frames_processed} frames using {decoder_used} + {runtime_cfg.backend}. "
                f"Chunks={len(chunks)} mode={quality_mode}"
            ),
            "matched_philosophy_author": "Experiment Engine",
            "tactical_instruction": "1. Compact midfield shape.\n2. Track weak-side runner.\n3. Trigger press on back-pass.",
        }
    ]
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    effective_fps = frames_processed / max(0.001, elapsed_ms / 1000.0)
    return PipelineArtifacts(
        report_path=report_path,
        tracking_path=tracking_path,
        decoder_used=decoder_used,
        elapsed_ms=elapsed_ms,
        frames_processed=frames_processed,
        decode_ms=decode_ms,
        infer_ms=total_infer_ms + total_fast_ms,
        post_ms=total_post_ms,
        reid_invocations=reid.invocations,
        reid_ms=reid.reid_ms,
        id_switch_rate=reid.id_switch_rate,
        effective_fps=effective_fps,
        chunks=[
            {
                "chunk_id": c.chunk_id,
                "start_frame": c.start_frame,
                "end_frame": c.end_frame,
            }
            for c in chunks
        ],
    )


def benchmark_decoders(video_path: Path) -> dict[str, dict[str, float | int]]:
    output_dir = Path(__file__).resolve().parents[1] / "output" / "bench_tmp"
    results: dict[str, dict[str, float | int]] = {}
    for decoder in ("opencv", "pyav"):
        artifacts = process_video(
            video_path,
            output_dir=output_dir,
            output_prefix=f"bench_{decoder}",
            decoder_mode=decoder,  # type: ignore[arg-type]
        )
        results[decoder] = {
            "elapsed_ms": round(artifacts.elapsed_ms, 2),
            "frames_processed": artifacts.frames_processed,
        }
    return results
