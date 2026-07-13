"""
benchmarks/profiler.py — Central coordinator for performance instrumentation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from core.interfaces import PipelineObserver

from .collectors import (
    IoSnapshot,
    PsutilIoCollector,
    PsutilResourceCollector,
    TorchCudaGpuCollector,
    make_gpu_collector,
)
from .models import (
    AggregateMetrics,
    BenchmarkConfig,
    BenchmarkReport,
    FrameProfileRecord,
    HardwareProfile,
    LLMCallRecord,
    ModuleSummary,
    QualityMetrics,
    STAGE_MODULE_MAP,
    StageResult,
    VideoProfile,
)

# Standard perf logger used by the existing observability pipeline
PERF_LOGGER = logging.getLogger("gaffer.perf")


class StageTimer:
    """
    Lightweight timing context manager.
    Measures wall-clock time and optional memory snapshots.
    Emits a PERF_STAGE log and returns a StageResult to the profiler.
    """

    def __init__(
        self,
        profiler: "PerformanceProfiler",
        stage_id: str,
        module_id: str,
        chunk_idx: int | None = None,
        snapshot_memory: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self._profiler = profiler
        self.stage_id = stage_id
        self.module_id = module_id
        self.chunk_idx = chunk_idx
        self.snapshot_memory = snapshot_memory
        self.extra = extra or {}
        
        self._t0: float = 0.0
        self._rss_start_mb: float | None = None
        self._io_start: IoSnapshot | None = None

    def __enter__(self) -> "StageTimer":
        if self.snapshot_memory and self._profiler._resource_collector:
            sample = self._profiler._resource_collector.sample()
            self._rss_start_mb = sample.rss_mb
        
        if self._profiler._io_collector:
            self._io_start = self._profiler._io_collector.snapshot()

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        duration_s = time.perf_counter() - self._t0
        status = "error" if exc_type else "ok"
        
        rss_end_mb: float | None = None
        ram_delta_mb: float | None = None
        if self.snapshot_memory and self._profiler._resource_collector:
            sample = self._profiler._resource_collector.sample()
            rss_end_mb = sample.rss_mb
            if self._rss_start_mb is not None and rss_end_mb is not None:
                ram_delta_mb = round(rss_end_mb - self._rss_start_mb, 2)

        io_read_bytes: int | None = None
        io_write_bytes: int | None = None
        if self._profiler._io_collector and self._io_start:
            io_end = self._profiler._io_collector.snapshot()
            io_delta = self._profiler._io_collector.delta(self._io_start, io_end)
            io_read_bytes = io_delta.read_bytes
            io_write_bytes = io_delta.write_bytes

        # Fast path for VRAM check — Torch collector is fast
        vram_peak_mb: float | None = None
        if self._profiler._torch_gpu_collector and self._profiler._torch_gpu_collector.is_available():
            vram_peak_mb = self._profiler._torch_gpu_collector.get_peak_vram_mb()
            self._profiler._torch_gpu_collector.reset_peak()
            
        # Emit standard PERF_STAGE log
        log_extra = {
            "job_id": self._profiler.job_id,
            "stage": self.stage_id,
            "duration_seconds": round(duration_s, 3),
            "status": status,
            "schema_version": "perf.v1",
            "benchmark_run_id": self._profiler.run_id,
            "module_id": self.module_id,
            "chunk_idx": self.chunk_idx,
            "worker_pid": os.getpid(),
        }
        if rss_end_mb is not None:
            log_extra["ram_rss_mb"] = rss_end_mb
        if vram_peak_mb is not None:
            log_extra["vram_mb"] = vram_peak_mb
        if io_read_bytes:
            log_extra["io_read_bytes"] = io_read_bytes
        if io_write_bytes:
            log_extra["io_write_bytes"] = io_write_bytes
        if self.extra:
            log_extra.update(self.extra)
            
        PERF_LOGGER.info("PERF_STAGE", extra=log_extra)

        # Record StageResult in profiler
        result = StageResult(
            stage_id=self.stage_id,
            stage_name=self.stage_id.split(".")[-1],
            module_id=self.module_id,
            duration_ms=round(duration_s * 1000.0, 3),
            status=status,
            chunk_idx=self.chunk_idx,
            worker_pid=os.getpid(),
            ram_start_mb=self._rss_start_mb,
            ram_end_mb=rss_end_mb,
            ram_delta_mb=ram_delta_mb,
            vram_peak_mb=vram_peak_mb,
            io_read_bytes=io_read_bytes,
            io_write_bytes=io_write_bytes,
            extra=self.extra,
        )
        self._profiler._add_stage_result(result)


class PerformanceProfiler(PipelineObserver):
    """
    Orchestrates performance instrumentation for a benchmark run.
    Uses thread-safe collections for main-process metrics.
    Worker subprocess metrics must be merged explicitly via merge_worker_results.
    """

    def __init__(self, job_id: str, config: BenchmarkConfig) -> None:
        self.job_id = job_id
        self.run_id = str(uuid.uuid4())
        self.config = config
        self.is_active = False
        
        self.hardware = HardwareProfile.detect()
        self.video_profile: VideoProfile | None = None
        self.pipeline_version: str = self._get_git_sha()

        self._stage_results: list[StageResult] = []
        self._llm_records: list[LLMCallRecord] = []
        self._frame_records: list[FrameProfileRecord] = []
        self._lock = threading.Lock()
        
        self._start_time_s: float = 0.0
        self._end_time_s: float = 0.0

        if self.config.benchmark_mode:
            self._resource_collector = PsutilResourceCollector()
            self._gpu_collector = make_gpu_collector(self.hardware.device_type)
            self._torch_gpu_collector = TorchCudaGpuCollector()
            self._io_collector = PsutilIoCollector()
        else:
            self._resource_collector = None
            self._gpu_collector = None
            self._torch_gpu_collector = None
            self._io_collector = None

    def _get_git_sha(self) -> str:
        try:
            from pathlib import Path
            import subprocess
            repo_root = Path(__file__).resolve().parent.parent.parent
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], 
                cwd=str(repo_root), 
                text=True, 
                stderr=subprocess.DEVNULL
            ).strip()
            return sha
        except Exception:
            return "unknown"

    def set_video_profile(self, profile: VideoProfile) -> None:
        self.video_profile = profile

    @contextmanager
    def run(self) -> Generator["PerformanceProfiler", None, None]:
        """Main lifecycle context manager for the benchmark run."""
        self.is_active = True
        self._start_time_s = time.perf_counter()
        
        if self._resource_collector:
            # Record baseline memory
            self._resource_collector.sample()
            self._resource_collector.start_background_sampling(
                interval_s=self.config.gpu_sample_interval_s
            )
            
        try:
            yield self
        finally:
            self._end_time_s = time.perf_counter()
            self.is_active = False
            
            if self._resource_collector:
                self._resource_collector.stop_background_sampling()

    @contextmanager
    def stage(
        self, stage_id: str, chunk_idx: int | None = None, **extra: Any
    ) -> Generator[StageTimer | None, None, None]:
        """
        Creates a StageTimer if benchmark mode is active.
        If inactive, yields None to avoid overhead.
        """
        if not self.config.benchmark_mode:
            yield None
            return

        module_id = STAGE_MODULE_MAP.get(stage_id, "MOD-UNK")
        timer = StageTimer(
            profiler=self,
            stage_id=stage_id,
            module_id=module_id,
            chunk_idx=chunk_idx,
            snapshot_memory=True,
            extra=extra,
        )
        with timer:
            yield timer

    def _add_stage_result(self, result: StageResult) -> None:
        with self._lock:
            self._stage_results.append(result)

    def record_llm_call(self, record: LLMCallRecord) -> None:
        if not self.config.benchmark_mode:
            return
        with self._lock:
            self._llm_records.append(record)

    def record_frame(self, record: FrameProfileRecord) -> None:
        if not self.config.frame_profiling:
            return
        with self._lock:
            self._frame_records.append(record)

    def merge_worker_results(self, worker_results: list[dict[str, Any]]) -> None:
        """
        Merge metrics returned from subprocess parallel workers into the main profiler.
        """
        with self._lock:
            for w in worker_results:
                if "stages" in w:
                    # Deserialize stage results
                    for s_dict in w["stages"]:
                        self._stage_results.append(StageResult(**s_dict))
                if "frames" in w and self.config.frame_profiling:
                    for f_dict in w["frames"]:
                        self._frame_records.append(FrameProfileRecord(**f_dict))

    def _compute_module_summary(self) -> list[ModuleSummary]:
        summary: dict[str, ModuleSummary] = {}
        
        with self._lock:
            for s in self._stage_results:
                if s.module_id not in summary:
                    summary[s.module_id] = ModuleSummary(
                        module_id=s.module_id,
                        total_duration_ms=0.0,
                        pct_of_total=0.0,
                        stage_count=0,
                        peak_ram_mb=0.0,
                        peak_vram_mb=0.0,
                    )
                m = summary[s.module_id]
                m.total_duration_ms += s.duration_ms
                m.stage_count += 1
                if s.ram_end_mb and (m.peak_ram_mb is None or s.ram_end_mb > m.peak_ram_mb):
                    m.peak_ram_mb = s.ram_end_mb
                if s.vram_peak_mb and (m.peak_vram_mb is None or s.vram_peak_mb > m.peak_vram_mb):
                    m.peak_vram_mb = s.vram_peak_mb
                    
        total_e2e_ms = (self._end_time_s - self._start_time_s) * 1000.0
        if total_e2e_ms > 0:
            for m in summary.values():
                m.pct_of_total = round((m.total_duration_ms / total_e2e_ms) * 100.0, 2)
                m.total_duration_ms = round(m.total_duration_ms, 2)
                
        return sorted(list(summary.values()), key=lambda x: x.total_duration_ms, reverse=True)

    def get_report(self) -> BenchmarkReport:
        """Assembles the final benchmark report."""
        if not self.config.benchmark_mode:
            raise RuntimeError("Cannot generate report: BENCHMARK_MODE is false.")
            
        end_to_end_ms = (self._end_time_s - self._start_time_s) * 1000.0
        
        # Aggregate timings by prefix (simple sum for now)
        def sum_stages(prefix: str) -> float:
            return sum(s.duration_ms for s in self._stage_results if s.stage_id.startswith(prefix))
            
        cal_ms = sum_stages("stage.calibration")
        par_ms = sum_stages("stage.parallel_cv")
        mrg_ms = sum_stages("stage.chunk_merge")
        spi_ms = sum_stages("stage.spatial_math")
        met_ms = sum_stages("stage.metrics_timeline")
        evt_ms = sum_stages("stage.event_intelligence")
        rpt_ms = sum_stages("stage.report_assembly")
        
        # LLM aggregation
        llm_total_ms = sum(r.latency_ms for r in self._llm_records)
        llm_calls = len(self._llm_records)
        prompt_t = sum((r.prompt_tokens or 0) for r in self._llm_records)
        comp_t = sum((r.completion_tokens or 0) for r in self._llm_records)
        
        # Throughput
        total_frames = self.video_profile.total_frames if self.video_profile else 0
        overall_fps = total_frames / (end_to_end_ms / 1000.0) if end_to_end_ms > 0 else 0.0
        cv_fps = total_frames / (par_ms / 1000.0) if par_ms > 0 else 0.0
        
        # Resource peaks
        peak_ram_mb = self._resource_collector.get_peak_ram_mb() if self._resource_collector else 0.0
        peak_vram_mb = self._gpu_collector.get_peak_vram_mb() if self._gpu_collector else None
        
        # We don't have true CPU peak from background sampler yet, approximate or set 0
        samples = self._resource_collector.get_samples() if self._resource_collector else []
        peak_cpu_pct = max([s.cpu_pct for s in samples], default=0.0)
        
        # I/O
        total_io_read = sum((s.io_read_bytes or 0) for s in self._stage_results)
        total_io_write = sum((s.io_write_bytes or 0) for s in self._stage_results)

        aggregate = AggregateMetrics(
            end_to_end_ms=round(end_to_end_ms, 3),
            calibration_ms=round(cal_ms, 3),
            parallel_cv_ms=round(par_ms, 3),
            chunk_merge_ms=round(mrg_ms, 3),
            spatial_math_ms=round(spi_ms, 3),
            metrics_timeline_ms=round(met_ms, 3),
            event_intelligence_ms=round(evt_ms, 3),
            llm_total_ms=round(llm_total_ms, 3),
            report_assembly_ms=round(rpt_ms, 3),
            overall_fps=round(overall_fps, 2),
            cv_fps=round(cv_fps, 2),
            peak_ram_mb=round(peak_ram_mb, 2),
            peak_vram_mb=peak_vram_mb,
            peak_cpu_pct=round(peak_cpu_pct, 2),
            total_io_read_bytes=total_io_read,
            total_io_write_bytes=total_io_write,
            total_llm_calls=llm_calls,
            total_llm_latency_ms=round(llm_total_ms, 3),
            total_prompt_tokens=prompt_t,
            total_completion_tokens=comp_t,
            hardware_profile_id=self.hardware.profile_id,
            pipeline_version=self.pipeline_version,
        )

        from .models import PIPELINE_DAG
        
        report = BenchmarkReport(
            benchmark_run_id=self.run_id,
            job_id=self.job_id,
            pipeline_version=self.pipeline_version,
            hardware_profile=self.hardware,
            video_profile=self.video_profile,
            stages=sorted(self._stage_results, key=lambda x: x.duration_ms, reverse=True),
            module_summary=self._compute_module_summary(),
            aggregate=aggregate,
            llm_calls=self._llm_records,
            pipeline_dag=PIPELINE_DAG,
            benchmark_config=self.config,
        )
        return report

    def write_report(self, output_dir: Path) -> Path:
        report = self.get_report()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Human readable full report
        report_path = output_dir / f"{self.run_id}_report.json"
        with report_path.open("w") as f:
            f.write(report.model_dump_json(indent=2))
            
        # 2. PERF_STAGE JSONL
        perf_path = output_dir / f"{self.run_id}.jsonl"
        with perf_path.open("w") as f:
            for s in self._stage_results:
                f.write(s.model_dump_json() + "\n")
                
        # 3. Frame records
        if self.config.frame_profiling and self._frame_records:
            frames_path = output_dir / f"{self.run_id}_frames.jsonl"
            with frames_path.open("w") as f:
                for fr in self._frame_records:
                    f.write(fr.model_dump_json() + "\n")
                    
        return report_path
