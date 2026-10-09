"""Compatibility import for the canonical supervised paired sampler."""

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parent / "Ablation" / "PairedClassSampler.py"
_MODULE = sys.modules.get("jmda_paired_sampler_core")
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location(
        "jmda_paired_sampler_core",
        _CORE_PATH,
    )
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load paired sampler from {_CORE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

PairedClassSampler = _MODULE.PairedClassSampler

__all__ = ["PairedClassSampler"]
