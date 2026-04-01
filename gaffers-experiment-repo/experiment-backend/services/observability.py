from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TimerStat:
    count: int
    p50_ms: float
    p95_ms: float
    avg_ms: float


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._timers: dict[str, list[float]] = defaultdict(list)

    def incr(self, key: str, value: int = 1) -> None:
        with self._lock:
            self._counters[key] += value

    @contextmanager
    def timed(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                self._timers[key].append(elapsed_ms)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            timers: dict[str, TimerStat] = {}
            for key, values in self._timers.items():
                if not values:
                    continue
                ordered = sorted(values)
                p50 = statistics.median(ordered)
                p95_index = max(0, int(len(ordered) * 0.95) - 1)
                p95 = ordered[p95_index]
                timers[key] = TimerStat(
                    count=len(values),
                    p50_ms=round(p50, 2),
                    p95_ms=round(p95, 2),
                    avg_ms=round(sum(values) / len(values), 2),
                )
            return {
                "counters": dict(self._counters),
                "timers": {k: asdict(v) for k, v in timers.items()},
            }
