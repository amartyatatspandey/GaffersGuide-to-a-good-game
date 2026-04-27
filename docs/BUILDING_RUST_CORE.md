# Building The Optional Rust Core

`rust_core/` contains an optional PyO3 extension named `gaffers_core_math`.
It is not required for the protected Cython wheel. The Cython build remains the
primary supported IP-protection path.

The Rust extension currently provides:

- `rank_candidates_rs`
- `TemporalBallPrior`

Python imports these opportunistically. If `gaffers_core_math` is unavailable,
the compiled Cython implementations are used automatically.

## Prerequisites

Install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install maturin:

```bash
python -m pip install maturin
```

## Build Locally

```bash
cd rust_core
maturin develop
```

Then verify:

```bash
python -c "from gaffers_core_math import rank_candidates_rs, TemporalBallPrior; print('rust ok')"
```

## Build A Wheel

```bash
cd rust_core
maturin build --release
```

The Rust wheel will be written under `rust_core/target/wheels/`.

## Notes

- This extension is deliberately narrow. It does not own YOLO, OpenCV, SAHI,
  or supervision objects.
- It is safe to skip this build. The package still works through the Cython
  protected modules.
