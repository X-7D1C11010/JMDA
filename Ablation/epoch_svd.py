"""Compatibility import for the canonical epoch-level SVD updater."""

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parents[1] / "epoch_svd.py"
_MODULE = sys.modules.get("jmda_epoch_svd_core")
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location("jmda_epoch_svd_core", _CORE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load epoch SVD utilities from {_CORE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

build_epoch_pairs = _MODULE.build_epoch_pairs
update_epoch_projections = _MODULE.update_epoch_projections
format_svd_update = _MODULE.format_svd_update

__all__ = [
    "build_epoch_pairs",
    "update_epoch_projections",
    "format_svd_update",
]
