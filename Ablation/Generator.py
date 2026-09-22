"""Compatibility import for the canonical neural transport implementation."""

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parents[1] / "Generator.py"
_MODULE = sys.modules.get("jmda_generator_core")
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location("jmda_generator_core", _CORE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load generator module from {_CORE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

CostNet = _MODULE.CostNet
TransmissionNetwork = _MODULE.TransmissionNetwork
NeuralOptimalTransportGenerator = _MODULE.NeuralOptimalTransportGenerator
TRANSPORT_MODES = _MODULE.TRANSPORT_MODES

__all__ = [
    "CostNet",
    "TransmissionNetwork",
    "NeuralOptimalTransportGenerator",
    "TRANSPORT_MODES",
]
