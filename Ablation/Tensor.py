"""Compatibility import for the project's canonical SVD tensor alignment."""

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parents[1] / "Tensor.py"
_MODULE = sys.modules.get("jmda_tensor_core")
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location("jmda_tensor_core", _CORE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load tensor alignment module from {_CORE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

TensorBasedAlignmentStable = _MODULE.TensorBasedAlignmentStable

__all__ = ["TensorBasedAlignmentStable"]
