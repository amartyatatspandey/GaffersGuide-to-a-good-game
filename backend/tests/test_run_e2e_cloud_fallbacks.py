from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from scripts import run_e2e as run_e2e_wrapper  # noqa: E402
from scripts import run_e2e_cloud  # noqa: E402


class _DummyCuda:
    def __init__(self, available: bool) -> None:
        self._available = available
        self.clear_calls = 0

    def is_available(self) -> bool:
        return self._available

    def empty_cache(self) -> None:
        self.clear_calls += 1


class _DummyMpsBackend:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _DummyMpsDevice:
    def __init__(self) -> None:
        self.clear_calls = 0

    def empty_cache(self) -> None:
        self.clear_calls += 1


class _DummyTorch:
    def __init__(self, *, cuda_available: bool, mps_available: bool) -> None:
        self.cuda = _DummyCuda(cuda_available)
        self.backends = type("Backends", (), {"mps": _DummyMpsBackend(mps_available)})()
        self.mps = _DummyMpsDevice()


def test_infer_device_falls_back_to_cpu_when_no_torch(monkeypatch) -> None:
    monkeypatch.setattr(run_e2e_cloud, "torch", None)
    assert run_e2e_cloud._infer_device(None) is None


def test_infer_device_prefers_mps_without_cuda(monkeypatch) -> None:
    dummy_torch = _DummyTorch(cuda_available=False, mps_available=True)
    monkeypatch.setattr(run_e2e_cloud, "torch", dummy_torch)
    assert run_e2e_cloud._infer_device("auto") == "mps"


def test_clear_device_cache_safe_on_cpu(monkeypatch) -> None:
    dummy_torch = _DummyTorch(cuda_available=False, mps_available=False)
    monkeypatch.setattr(run_e2e_cloud, "torch", dummy_torch)
    run_e2e_cloud._clear_device_cache("cpu")
    assert dummy_torch.cuda.clear_calls == 0
    assert dummy_torch.mps.clear_calls == 0


def test_wrapper_run_e2e_delegates_to_cloud(monkeypatch, tmp_path: Path) -> None:
    expected = tmp_path / "report.json"
    calls: dict[str, object] = {}

    def _fake_run_e2e_cloud(
        video: str | Path,
        *,
        output_prefix: str = "test_mp4",
        progress_callback=None,
        batch_size: int = run_e2e_cloud.DEFAULT_BATCH_SIZE,
        flow_max_width: int = run_e2e_cloud.DEFAULT_FLOW_MAX_WIDTH,
        device: str | None = None,
    ) -> Path:
        calls["video"] = video
        calls["output_prefix"] = output_prefix
        calls["progress_callback"] = progress_callback
        calls["batch_size"] = batch_size
        calls["flow_max_width"] = flow_max_width
        calls["device"] = device
        return expected

    monkeypatch.setattr(run_e2e_wrapper, "run_e2e_cloud", _fake_run_e2e_cloud)
    out = run_e2e_wrapper.run_e2e("sample.mp4", output_prefix="job_1")
    assert out == expected
    assert calls["video"] == "sample.mp4"
    assert calls["output_prefix"] == "job_1"
