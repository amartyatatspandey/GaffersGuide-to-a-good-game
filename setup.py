"""Setuptools bridge for compiled gaffers-guide modules."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup


_setup_ext_path = Path(__file__).with_name("setup_ext.py")
_spec = importlib.util.spec_from_file_location("_gaffers_setup_ext", _setup_ext_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Could not load build helper: {_setup_ext_path}")
_setup_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_setup_ext)


build_setup_kwargs = _setup_ext.build_setup_kwargs


setup(**build_setup_kwargs())
