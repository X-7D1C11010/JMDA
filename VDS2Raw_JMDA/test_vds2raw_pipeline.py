"""Unit and integration tests for the isolated VDS2Raw training path."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Tensor import TensorBasedAlignmentStable
from VDS2Raw_JMDA.models import (
    AISFeatureExtractor,
    NIRFeatureExtractor,
    RGBFeatureExtractor,
)
from VDS2Raw_JMDA.paired_sampler import SameClassPairedBatcher
from VDS2Raw_JMDA.vds2raw_dataset import (
    TRAIN_PARTITIONS,
    VDS2RawDataset,
    VDS2RawManifest,
    compute_train_image_stats,
)


DATASET_ROOT = Path(
    os.environ.get("VDS2RAW_ROOT", r"D:\Downloads\VDS2Raw\threeclass_ready")
)


class _LabelDataset:
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class PairingTests(unittest.TestCase):
    def test_target_is_covered_once_without_balancing(self):
        source = _LabelDataset([0, 0, 1, 1, 2, 2])
        target = _LabelDataset([0, 0, 0, 1, 2])
        batcher = SameClassPairedBatcher(source, target, batch_size=2, seed=17)
        pairs = batcher.pairs_for_epoch(0)
        self.assertEqual(len(pairs), len(target))
        self.assertEqual(sorted(target_index for _, target_index in pairs), list(range(len(target))))
        self.assertTrue(
            all(source.labels[source_index] == target.labels[target_index] for source_index, target_index in pairs)
        )


class ModelTests(unittest.TestCase):
    def test_three_encoder_shapes(self):
        with torch.no_grad():
            rgb = RGBFeatureExtractor(output_dim=32, pretrained=False)(
                torch.randn(2, 3, 96, 96)
            )
            nir = NIRFeatureExtractor(output_dim=32)(torch.randn(2, 1, 96, 96))
            ais = AISFeatureExtractor(output_dim=32)(torch.randn(2, 26))
        self.assertEqual(tuple(rgb.shape), (2, 32))
        self.assertEqual(tuple(nir.shape), (2, 32))
        self.assertEqual(tuple(ais.shape), (2, 32))

    def test_three_modal_tensor_projection(self):
        module = TensorBasedAlignmentStable(
            input_dims=[8, 8, 8],
            output_dims=[3, 3, 3],
            num_modalities=3,
            max_svd_sweeps=2,
        )
        source = [torch.randn(10, 8) for _ in range(3)]
        target = [torch.randn(10, 8) for _ in range(3)]
        information = module.update_projections(source, target)
        projected_source, projected_target, loss = module(source, target)
        self.assertEqual(information["sample_count"], 10)
        self.assertEqual([tuple(value.shape) for value in projected_source], [(10, 3)] * 3)
        self.assertEqual([tuple(value.shape) for value in projected_target], [(10, 3)] * 3)
        self.assertTrue(torch.isfinite(loss))


@unittest.skipUnless(DATASET_ROOT.is_dir(), "VDS2Raw prepared dataset is not available")
class PreparedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = VDS2RawManifest(DATASET_ROOT)
        cls.audit = cls.manifest.audit(validate_npz=True)
        cls.stats = compute_train_image_stats(cls.manifest)

    def test_fixed_partition_and_leakage_audit(self):
        self.assertEqual(self.audit["status"], "PASS")
        self.assertEqual(
            self.audit["partition_counts"],
            {"source_train": 116, "target_train": 94, "target_val": 28, "target_test": 33},
        )
        self.assertTrue(all(value == 0 for value in self.audit["scene_overlap_counts"].values()))
        self.assertTrue(all(value == 0 for value in self.audit["mmsi_overlap_counts"].values()))

    def test_training_only_statistics(self):
        self.assertEqual(tuple(self.stats["fit_partitions"]), TRAIN_PARTITIONS)
        self.assertEqual(self.stats["band_order"], ["B4", "B3", "B2", "B8"])
        self.assertTrue(np.isfinite(self.stats["mean"]).all())
        self.assertTrue(np.asarray(self.stats["std"]).min() > 0)

    def test_npz_loader_shapes(self):
        dataset = VDS2RawDataset(self.manifest, "target_train", self.stats)
        sample = dataset[0]
        self.assertEqual(tuple(sample["rgb"].shape), (3, 96, 96))
        self.assertEqual(tuple(sample["nir"].shape), (1, 96, 96))
        self.assertEqual(tuple(sample["ais"].shape), (13,))
        self.assertEqual(tuple(sample["ais_mask"].shape), (13,))
        self.assertEqual(tuple(sample["ais_input"].shape), (26,))
        self.assertEqual(sample["rgb"].dtype, torch.float32)
        self.assertEqual(sample["nir"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample["rgb"]).all())
        self.assertTrue(torch.isfinite(sample["nir"]).all())

    def test_padding_is_neutral_after_normalization(self):
        dataset = VDS2RawDataset(self.manifest, "source_train", self.stats)
        checked = False
        for index, record in enumerate(dataset.records):
            with np.load(self.manifest.resolve_npz(record), allow_pickle=False) as payload:
                raw = np.concatenate([payload["rgb"], payload["nir"]], axis=0)
            padding = np.all(raw == 0, axis=0)
            if padding.any():
                sample = dataset[index]
                normalized = torch.cat([sample["rgb"], sample["nir"]], dim=0)
                self.assertTrue(torch.equal(normalized[:, torch.from_numpy(padding)], torch.zeros_like(normalized[:, torch.from_numpy(padding)])))
                checked = True
                break
        self.assertTrue(checked, "Expected at least one padded source_train crop")


if __name__ == "__main__":
    unittest.main()
