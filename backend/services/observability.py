from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(slots=True)
class _TimerSample:
    name: str
    elapsed_ms: float


class PipelineMetricsRegistry:
    """In-memory metrics registry for beta baselining and SLO gates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def incr(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] += value

    def observe_ms(self, key: str, elapsed_ms: float) -> None:
        with self._lock:
            self._timers[key].append(float(elapsed_ms))

    @contextmanager
    def timed(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.observe_ms(key, elapsed_ms)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            timers: dict[str, dict[str, float | int]] = {}
            for key, samples in self._timers.items():
                if not samples:
                    continue
                values = sorted(samples)
                count = len(values)
                p50 = values[int((count - 1) * 0.50)]
                p95 = values[int((count - 1) * 0.95)]
                timers[key] = {
                    "count": count,
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "avg_ms": round(sum(values) / count, 2),
                }
            return {
                "counters": dict(self._counters),
                "timers": timers,
            }
