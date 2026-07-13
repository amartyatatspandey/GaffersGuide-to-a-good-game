"""
benchmarks/ — Performance Instrumentation Framework for Gaffer's Guide V2
=========================================================================

This package provides production-grade benchmarking and profiling tools.
It has ZERO import dependency on the backend service layer.

Activation: set BENCHMARK_MODE=true environment variable.
When BENCHMARK_MODE is not set or false, all instrumentation is no-ops.

Entry point: python -m benchmarks.cli
"""
from __future__ import annotations

from .models import BenchmarkConfig

# Global benchmark configuration
config = BenchmarkConfig.from_env()

BENCHMARK_MODE: bool = config.benchmark_mode
FRAME_PROFILING: bool = config.frame_profiling
ENABLE_CPROFILE: bool = config.enable_cprofile

__all__ = ["BENCHMARK_MODE", "FRAME_PROFILING", "ENABLE_CPROFILE", "config"]
