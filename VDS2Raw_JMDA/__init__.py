"""Dedicated VDS2Raw training package for JMDA-Net.

This package is isolated from the historical weather-dataset code under
``Ablation``. The public entry point is ``module_ablation.py`` in this folder.
"""

from .vds2raw_dataset import (
    CLASS_NAMES,
    CLASS_TO_ID,
    PARTITIONS,
    TRAIN_PARTITIONS,
    VDS2RawDataset,
    VDS2RawManifest,
    compute_train_image_stats,
)

__all__ = [
    "CLASS_NAMES",
    "CLASS_TO_ID",
    "PARTITIONS",
    "TRAIN_PARTITIONS",
    "VDS2RawDataset",
    "VDS2RawManifest",
    "compute_train_image_stats",
]
