# Building Windows Wheels

This guide explains how to build a Windows wheel for `gaffers-guide` with the
protected Cython modules compiled into `.pyd` native extensions.

The goal is to produce a wheel that works on Windows and does **not** ship the
readable `.py`, `.pyx`, or generated `.c` sources for protected modules.

## What Gets Compiled

The protected modules are compiled from Cython into native extensions:

- `gaffers_guide.cv.temporal_ball_prior`
- `gaffers_guide.cv.pitch_roi_provider`
- `gaffers_guide.cv.ball_candidate_fuser`
- `gaffers_guide.cv.slice_batch_inferencer`
- `gaffers_guide.cv.optimized_sahi_wrapper`
- `gaffers_guide.pipeline.advanced_pitch_calibration`
- `gaffers_guide.pipeline.dynamic_homography`
- `gaffers_guide.pipeline.track_teams`
- `gaffers_guide.pipeline.track_teams_reid_hybrid`
- `gaffers_guide.pipeline.reid_healer`
- `gaffers_guide.pipeline.global_refiner`
- `gaffers_guide.pipeline.e2e_shared_impl`
- `gaffers_guide.pipeline.tactical_radar`
- `gaffers_guide.pipeline.generate_analytics`

On Windows these become `.pyd` files inside the wheel.

## Prerequisites

1. Install **Python 3.11** from <https://www.python.org/downloads/>.
   - Check **Add python.exe to PATH** during installation.
   - Use 64-bit Python.

2. Install **Microsoft C++ Build Tools** from:
   <https://visualstudio.microsoft.com/visual-cpp-build-tools/>

3. During Build Tools installation, select:
   - **Desktop development with C++**
   - MSVC v143 or newer
   - Windows 10/11 SDK

4. Open:

```bat
x64 Native Tools Command Prompt for VS 2022
```

Do not use a normal Command Prompt for the first build. The Native Tools prompt
sets compiler environment variables that Python needs to compile `.pyd` files.

## Build The Wheel

From the repository root:

```bat
cd path\to\phoenix-work
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install cython numpy build setuptools wheel
python setup_ext.py build_ext --inplace
python -m build --wheel --outdir dist_windows
```

The wheel should be written to `dist_windows` and look similar to:

```text
gaffers_guide-2.0.2-cp311-cp311-win_amd64.whl
```

## Verify The Wheel Does Not Leak Protected Sources

List the wheel contents:

```bat
python -m zipfile -l dist_windows\gaffers_guide-*.whl
```

Protected modules should appear as `.pyd` files, for example:

```text
gaffers_guide/cv/optimized_sahi_wrapper.cp311-win_amd64.pyd
gaffers_guide/pipeline/track_teams.cp311-win_amd64.pyd
```

Protected modules should **not** appear as:

```text
gaffers_guide/cv/optimized_sahi_wrapper.py
gaffers_guide/cv/optimized_sahi_wrapper.pyx
gaffers_guide/cv/optimized_sahi_wrapper.c
gaffers_guide/pipeline/track_teams.py
gaffers_guide/pipeline/track_teams.pyx
gaffers_guide/pipeline/track_teams.c
```

You can run this quick audit:

```bat
python - <<PY
from pathlib import Path
from zipfile import ZipFile

wheel = sorted(Path("dist_windows").glob("gaffers_guide-*.whl"))[-1]
protected = {
    "gaffers_guide/cv/temporal_ball_prior",
    "gaffers_guide/cv/pitch_roi_provider",
    "gaffers_guide/cv/ball_candidate_fuser",
    "gaffers_guide/cv/slice_batch_inferencer",
    "gaffers_guide/cv/optimized_sahi_wrapper",
    "gaffers_guide/pipeline/advanced_pitch_calibration",
    "gaffers_guide/pipeline/dynamic_homography",
    "gaffers_guide/pipeline/track_teams",
    "gaffers_guide/pipeline/track_teams_reid_hybrid",
    "gaffers_guide/pipeline/reid_healer",
    "gaffers_guide/pipeline/global_refiner",
    "gaffers_guide/pipeline/e2e_shared_impl",
    "gaffers_guide/pipeline/tactical_radar",
    "gaffers_guide/pipeline/generate_analytics",
}

with ZipFile(wheel) as zf:
    names = zf.namelist()

compiled = [
    name for name in names
    if any(name.startswith(module + ".") and name.endswith(".pyd") for module in protected)
]
leaks = [
    name for name in names
    if name.rsplit(".", 1)[0] in protected and name.endswith((".py", ".pyx", ".c"))
]

print(f"wheel={wheel}")
print(f"compiled_extensions={len(compiled)}")
print(f"source_leaks={len(leaks)}")
for leak in leaks:
    print(f"LEAK: {leak}")

raise SystemExit(0 if len(compiled) == len(protected) and not leaks else 1)
PY
```

Expected result:

```text
compiled_extensions=14
source_leaks=0
```

## Install And Smoke-Test The Wheel

Create a clean test environment:

```bat
cd path\to\phoenix-work
python -m venv .venv-wheel-test
.venv-wheel-test\Scripts\activate
python -m pip install --upgrade pip
python -m pip install dist_windows\gaffers_guide-*.whl
```

Verify compiled imports:

```bat
python -c "from gaffers_guide.cv.optimized_sahi_wrapper import OptimizedSAHIWrapper; print('cv ok')"
python -c "from gaffers_guide.pipeline.track_teams import TeamClassifier; print('pipeline ok')"
gaffers-guide --help
```

## Common Windows Issues

### `Microsoft Visual C++ 14.0 or greater is required`

Install or repair **Microsoft C++ Build Tools**, then reopen **x64 Native Tools
Command Prompt for VS 2022**.

### `Python.h: No such file or directory`

Use the official Python installer from python.org and make sure it is a full
CPython install, not the Microsoft Store shim.

### Wheel imports locally but fails on another Windows machine

Build and test on the same Python minor version. A wheel built for `cp311` is
for Python 3.11. It will not install on Python 3.12.

## Recommended Release Workflow

For repeatable releases, use GitHub Actions with `cibuildwheel` to build:

- macOS arm64 / x86_64 wheels
- Windows amd64 wheels
- Linux wheels if needed later

This is more reliable than asking teammates to build manually every release.
