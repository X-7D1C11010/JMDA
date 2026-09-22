"""Compatibility import for the canonical domain discriminator."""

import importlib.util
import sys
from pathlib import Path


_CORE_PATH = Path(__file__).resolve().parents[2] / "Discriminator.py"
_MODULE = sys.modules.get("jmda_discriminator_core")
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location("jmda_discriminator_core", _CORE_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Cannot load discriminator module from {_CORE_PATH}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

GradientReversal = _MODULE.GradientReversal
DomainDiscriminator = _MODULE.DomainDiscriminator
compute_discriminator_loss = _MODULE.compute_discriminator_loss
compute_generator_loss = _MODULE.compute_generator_loss

__all__ = [
    "GradientReversal",
    "DomainDiscriminator",
    "compute_discriminator_loss",
    "compute_generator_loss",
]
