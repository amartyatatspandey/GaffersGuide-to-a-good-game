from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

if "google" not in sys.modules:
    import types

    sys.modules["google"] = types.ModuleType("google")
if "google.generativeai" not in sys.modules:
    import types

    mod = types.ModuleType("google.generativeai")
    setattr(mod, "configure", lambda **_: None)
    setattr(mod, "GenerativeModel", lambda *_args, **_kwargs: None)
    sys.modules["google.generativeai"] = mod

from main import app  # noqa: E402


def test_list_datasets_empty_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DATASETS_ROOT", str(tmp_path))
    client = TestClient(app)
    res = client.get("/api/datasets")
    assert res.status_code == 200
    assert res.json() == {"datasets": []}


def test_list_datasets_scans_subdirs(tmp_path: Path, monkeypatch) -> None:
    ds_root = tmp_path / "datasets"
    a = ds_root / "ds_a"
    b = ds_root / "ds_b"
    (a / "nested").mkdir(parents=True)
    (a / "nested" / "1.txt").write_text("x", encoding="utf-8")
    (a / "nested" / "2.txt").write_text("y", encoding="utf-8")
    b.mkdir(parents=True)
    (b / "empty_sub").mkdir()

    monkeypatch.setenv("DATASETS_ROOT", str(ds_root))
    client = TestClient(app)
    res = client.get("/api/datasets")
    assert res.status_code == 200
    body = res.json()
    assert "datasets" in body
    rows = {d["name"]: d for d in body["datasets"]}
    assert set(rows) == {"ds_a", "ds_b"}
    assert rows["ds_a"]["split"] == "all"
    assert rows["ds_a"]["num_samples"] == 2
    assert rows["ds_a"]["root_dir"] == str(a.resolve())
    assert rows["ds_b"]["num_samples"] == 0
