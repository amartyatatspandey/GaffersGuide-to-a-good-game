"""
benchmarks/collectors.py — Metric collector implementations.

Protocol-based collectors for RAM, GPU, CPU, and I/O metrics.
All collectors handle missing dependencies gracefully (return null values).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ── Data transfer objects ─────────────────────────────────────────────────────

@dataclass
class ResourceSample:
    timestamp_s: float
    rss_mb: float
    cpu_pct: float
    cpu_per_core: list[float] = field(default_factory=list)


@dataclass
class GpuSample:
    timestamp_s: float
    gpu_utilisation_pct: float | None
    vram_used_mb: float | None
    vram_total_mb: float | None
    source: str = "unknown"  # "nvml", "torch_cuda", "not_available_mps", "not_available"


@dataclass
class IoSnapshot:
    read_bytes: int = 0
    write_bytes: int = 0
    timestamp_s: float = 0.0


# ── Collector protocols ───────────────────────────────────────────────────────

@runtime_checkable
class ResourceCollector(Protocol):
    def sample(self) -> ResourceSample: ...
    def start_background_sampling(self, interval_s: float) -> None: ...
    def stop_background_sampling(self) -> None: ...
    def get_peak_ram_mb(self) -> float: ...
    def get_samples(self) -> list[ResourceSample]: ...


@runtime_checkable
class GpuCollector(Protocol):
    def is_available(self) -> bool: ...
    def sample(self) -> GpuSample | None: ...
    def get_peak_vram_mb(self) -> float | None: ...
    def reset_peak(self) -> None: ...


@runtime_checkable
class IoCollector(Protocol):
    def snapshot(self) -> IoSnapshot: ...
    def delta(self, start: IoSnapshot, end: IoSnapshot) -> IoSnapshot: ...


# ── Resource (RAM + CPU) collector ───────────────────────────────────────────

class PsutilResourceCollector:
    """
    Collects RAM and CPU utilisation using psutil.
    Gracefully handles missing psutil by returning zero values.
    """

    def __init__(self) -> None:
        self._samples: list[ResourceSample] = []
        self._peak_ram_mb: float = 0.0
        self._lock = threading.Lock()
        self._sampler_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        try:
            import psutil  # noqa: F401
            self._psutil_available = True
            self._process = psutil.Process(os.getpid())
            # Prime the cpu_percent call so the first real call is accurate
            self._process.cpu_percent(interval=None)
        except ImportError:
            self._psutil_available = False
            self._process = None

    def sample(self) -> ResourceSample:
        ts = time.monotonic()
        rss_mb = 0.0
        cpu_pct = 0.0
        cpu_per_core: list[float] = []

        if self._psutil_available:
            try:
                import psutil
                info = self._process.memory_info()
                rss_mb = round(info.rss / (1024 ** 2), 2)
                cpu_pct = self._process.cpu_percent(interval=None)
                cpu_per_core = psutil.cpu_percent(percpu=True)
            except Exception:
                pass

        s = ResourceSample(
            timestamp_s=ts,
            rss_mb=rss_mb,
            cpu_pct=cpu_pct,
            cpu_per_core=cpu_per_core if isinstance(cpu_per_core, list) else [],
        )
        with self._lock:
            self._samples.append(s)
            if rss_mb > self._peak_ram_mb:
                self._peak_ram_mb = rss_mb
        return s

    def start_background_sampling(self, interval_s: float = 2.0) -> None:
        self._stop_event.clear()

        def _loop() -> None:
            while not self._stop_event.wait(timeout=interval_s):
                self.sample()

        self._sampler_thread = threading.Thread(
            target=_loop, name="bench-resource-sampler", daemon=True
        )
        self._sampler_thread.start()

    def stop_background_sampling(self) -> None:
        self._stop_event.set()
        if self._sampler_thread and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=5.0)
        self._sampler_thread = None
        # Final sample
        self.sample()

    def get_peak_ram_mb(self) -> float:
        with self._lock:
            return self._peak_ram_mb

    def get_samples(self) -> list[ResourceSample]:
        with self._lock:
            return list(self._samples)


# ── GPU collectors ────────────────────────────────────────────────────────────

class NullGpuCollector:
    """No-op GPU collector for CPU and MPS devices."""

    def __init__(self, reason: str = "not_available") -> None:
        self._reason = reason

    def is_available(self) -> bool:
        return False

    def sample(self) -> GpuSample | None:
        return GpuSample(
            timestamp_s=time.monotonic(),
            gpu_utilisation_pct=None,
            vram_used_mb=None,
            vram_total_mb=None,
            source=self._reason,
        )

    def get_peak_vram_mb(self) -> float | None:
        return None

    def reset_peak(self) -> None:
        pass


class NvmlGpuCollector:
    """
    GPU collector using pynvml for NVIDIA CUDA devices.
    Falls back to NullGpuCollector behaviour if pynvml is unavailable.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index
        self._available = False
        self._handle = None
        self._peak_vram_mb: float = 0.0
        self._lock = threading.Lock()

        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._available = True
            self._pynvml = pynvml
        except Exception:
            pass

    def is_available(self) -> bool:
        return self._available

    def sample(self) -> GpuSample | None:
        if not self._available:
            return GpuSample(
                timestamp_s=time.monotonic(),
                gpu_utilisation_pct=None,
                vram_used_mb=None,
                vram_total_mb=None,
                source="not_available",
            )
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            vram_used = round(mem.used / (1024 ** 2), 2)
            vram_total = round(mem.total / (1024 ** 2), 2)
            with self._lock:
                if vram_used > self._peak_vram_mb:
                    self._peak_vram_mb = vram_used
            return GpuSample(
                timestamp_s=time.monotonic(),
                gpu_utilisation_pct=float(util.gpu),
                vram_used_mb=vram_used,
                vram_total_mb=vram_total,
                source="nvml",
            )
        except Exception:
            return None

    def get_peak_vram_mb(self) -> float | None:
        with self._lock:
            return self._peak_vram_mb if self._peak_vram_mb > 0 else None

    def reset_peak(self) -> None:
        with self._lock:
            self._peak_vram_mb = 0.0


class TorchCudaGpuCollector:
    """
    GPU collector using torch.cuda memory tracking.
    Complements NvmlGpuCollector for PyTorch-specific VRAM metrics.
    """

    def __init__(self) -> None:
        self._available = False
        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                self._available = True
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def is_available(self) -> bool:
        return self._available

    def sample(self) -> GpuSample | None:
        if not self._available:
            return None
        try:
            allocated = self._torch.cuda.memory_allocated()
            vram_mb = round(allocated / (1024 ** 2), 2)
            return GpuSample(
                timestamp_s=time.monotonic(),
                gpu_utilisation_pct=None,  # Not available from torch
                vram_used_mb=vram_mb,
                vram_total_mb=None,
                source="torch_cuda",
            )
        except Exception:
            return None

    def get_peak_vram_mb(self) -> float | None:
        if not self._available:
            return None
        try:
            peak = self._torch.cuda.max_memory_allocated()
            return round(peak / (1024 ** 2), 2) if peak > 0 else None
        except Exception:
            return None

    def reset_peak(self) -> None:
        if self._available:
            try:
                self._torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


# ── I/O collector ─────────────────────────────────────────────────────────────

class PsutilIoCollector:
    """
    Tracks disk I/O byte counts using psutil for the current process.
    Note: subprocess workers have separate PIDs and are not tracked here.
    """

    def __init__(self) -> None:
        self._psutil_available = False
        self._process = None
        try:
            import psutil
            self._process = psutil.Process(os.getpid())
            # Verify io_counters is available on this platform
            _ = self._process.io_counters()
            self._psutil_available = True
        except Exception:
            pass

    def snapshot(self) -> IoSnapshot:
        if not self._psutil_available:
            return IoSnapshot(read_bytes=0, write_bytes=0, timestamp_s=time.monotonic())
        try:
            counters = self._process.io_counters()
            return IoSnapshot(
                read_bytes=counters.read_bytes,
                write_bytes=counters.write_bytes,
                timestamp_s=time.monotonic(),
            )
        except Exception:
            return IoSnapshot(read_bytes=0, write_bytes=0, timestamp_s=time.monotonic())

    def delta(self, start: IoSnapshot, end: IoSnapshot) -> IoSnapshot:
        return IoSnapshot(
            read_bytes=max(0, end.read_bytes - start.read_bytes),
            write_bytes=max(0, end.write_bytes - start.write_bytes),
            timestamp_s=end.timestamp_s,
        )


# ── Factory ───────────────────────────────────────────────────────────────────

def make_gpu_collector(device_type: str) -> NvmlGpuCollector | NullGpuCollector:
    """Return the best available GPU collector for the detected device type."""
    if device_type == "cuda":
        collector = NvmlGpuCollector()
        if collector.is_available():
            return collector
        # Fall back to null with explanation
        return NullGpuCollector(reason="nvml_unavailable")
    elif device_type == "mps":
        return NullGpuCollector(reason="not_available_mps")
    else:
        return NullGpuCollector(reason="not_available")
