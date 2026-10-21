"""Unit and integration tests for the isolated VDS2Raw training path."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
from VDS2Raw_JMDA.complexity import parameter_report
from VDS2Raw_JMDA.engine import (
    checkpoint_selection_start_epoch,
    historical_classification_loss,
)
from VDS2Raw_JMDA.module_ablation import summarize_numeric_values
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
    def test_target_classes_are_balanced_and_all_samples_are_covered(self):
        source = _LabelDataset([0, 0, 1, 1, 2, 2])
        target = _LabelDataset([0, 0, 0, 1, 2])
        batcher = SameClassPairedBatcher(source, target, batch_size=2, seed=17)
        pairs = batcher.pairs_for_epoch(0)
        self.assertEqual(len(pairs), 9)
        observed_target_indices = [target_index for _, target_index in pairs]
        self.assertTrue(set(range(len(target))).issubset(observed_target_indices))
        balanced_counts = {
            label: sum(target.labels[index] == label for index in observed_target_indices)
            for label in (0, 1, 2)
        }
        self.assertEqual(balanced_counts, {0: 3, 1: 3, 2: 3})
        self.assertTrue(
            all(source.labels[source_index] == target.labels[target_index] for source_index, target_index in pairs)
        )

    def test_balancing_can_be_disabled_for_diagnostics(self):
        source = _LabelDataset([0, 0, 1, 1, 2, 2])
        target = _LabelDataset([0, 0, 0, 1, 2])
        batcher = SameClassPairedBatcher(
            source,
            target,
            batch_size=2,
            seed=17,
            balance_classes=False,
        )
        pairs = batcher.pairs_for_epoch(0)
        self.assertEqual(sorted(target_index for _, target_index in pairs), list(range(len(target))))


class ModelTests(unittest.TestCase):
    def test_three_encoder_shapes(self):
        with torch.no_grad():
            rgb = RGBFeatureExtractor(output_dim=32, pretrained=False)(
                torch.randn(2, 3, 224, 224)
            )
            nir = NIRFeatureExtractor(output_dim=32)(torch.randn(2, 1, 224, 224))
            ais = AISFeatureExtractor(output_dim=32)(torch.randn(2, 26))
        self.assertEqual(tuple(rgb.shape), (2, 32))
        self.assertEqual(tuple(nir.shape), (2, 32))
        self.assertEqual(tuple(ais.shape), (2, 32))

    def test_rgb_freeze_matches_historical_effective_policy(self):
        model = RGBFeatureExtractor(output_dim=32, pretrained=False)
        model.freeze_backbone()
        self.assertTrue(all(not parameter.requires_grad for parameter in model.features.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.proj.parameters()))

    def test_three_modal_tensor_projection(self):
        module = TensorBasedAlignmentStable(
            input_dims=[8, 8, 8],
            output_dims=[3, 3, 2],
            num_modalities=3,
            max_svd_sweeps=2,
        )
        source = [torch.randn(10, 8) for _ in range(3)]
        target = [torch.randn(10, 8) for _ in range(3)]
        information = module.update_projections(source, target)
        projected_source, projected_target, loss = module(source, target)
        self.assertEqual(information["sample_count"], 10)
        self.assertEqual([tuple(value.shape) for value in projected_source], [(10, 3), (10, 3), (10, 2)])
        self.assertEqual([tuple(value.shape) for value in projected_target], [(10, 3), (10, 3), (10, 2)])
        self.assertTrue(torch.isfinite(loss))


class ReportingTests(unittest.TestCase):
    def test_multi_seed_statistics_include_mean_and_variance(self):
        summary = summarize_numeric_values([0.8, 0.9, 1.0])
        self.assertEqual(summary["count"], 3)
        self.assertAlmostEqual(summary["mean"], 0.9)
        self.assertEqual(summary["minimum"], 0.8)
        self.assertEqual(summary["maximum"], 1.0)
        self.assertAlmostEqual(summary["variance_population"], 2.0 / 300.0)
        self.assertAlmostEqual(summary["variance_sample"], 0.01)

    def test_parameter_report_separates_training_only_modules(self):
        class Modules:
            rgb = nn.Linear(2, 3)
            nir = nn.Linear(2, 3)
            ais = nn.Linear(2, 3)
            tensor = None
            classifier = nn.Linear(9, 2)
            generator = nn.Linear(9, 9)
            discriminator = nn.Linear(9, 1)

        report = parameter_report(Modules())
        expected_inference = sum(
            p.numel()
            for module in (Modules.rgb, Modules.nir, Modules.ais, Modules.classifier)
            for p in module.parameters()
        )
        self.assertEqual(report["inference_parameters"], expected_inference)
        self.assertGreater(report["total_parameters"], report["inference_parameters"])

    def test_full_checkpoint_selection_starts_after_ot_ramp(self):
        self.assertEqual(checkpoint_selection_start_epoch(10, 5, True, True), 15)
        self.assertEqual(checkpoint_selection_start_epoch(10, 5, True, False), 1)
        self.assertEqual(checkpoint_selection_start_epoch(10, 5, False, True), 1)

    def test_historical_classification_terms_are_summed(self):
        criterion = nn.CrossEntropyLoss()
        labels = torch.tensor([0, 1])
        source = torch.tensor([[2.0, -1.0], [-1.0, 2.0]])
        target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        intermediate = torch.tensor([[0.5, 0.0], [0.0, 0.5]])
        expected = (
            criterion(source, labels)
            + criterion(target, labels)
            + 0.4 * criterion(intermediate, labels)
        )
        actual = historical_classification_loss(
            criterion,
            source,
            labels,
            target,
            labels,
            intermediate,
            0.4,
        )
        self.assertTrue(torch.allclose(actual, expected))


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
        self.assertEqual(tuple(sample["rgb"].shape), (3, 224, 224))
        self.assertEqual(tuple(sample["nir"].shape), (1, 224, 224))
        self.assertEqual(tuple(sample["ais"].shape), (13,))
        self.assertEqual(tuple(sample["ais_mask"].shape), (13,))
        self.assertEqual(tuple(sample["ais_input"].shape), (26,))
        self.assertEqual(sample["rgb"].dtype, torch.float32)
        self.assertEqual(sample["nir"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(sample["rgb"]).all())
        self.assertTrue(torch.isfinite(sample["nir"]).all())

    def test_training_augmentation_preserves_heading_magnitude(self):
        dataset = VDS2RawDataset(
            self.manifest,
            "target_train",
            self.stats,
            augment=True,
        )
        record = dataset.records[0]
        with np.load(self.manifest.resolve_npz(record), allow_pickle=False) as payload:
            raw_heading_sin = float(payload["ais"][0])
            raw_heading_mask = float(payload["ais_mask"][0])
        torch.manual_seed(123)
        sample = dataset[0]
        self.assertEqual(tuple(sample["rgb"].shape), (3, 224, 224))
        self.assertEqual(tuple(sample["nir"].shape), (1, 224, 224))
        self.assertEqual(float(sample["ais_mask"][0]), raw_heading_mask)
        self.assertAlmostEqual(abs(float(sample["ais"][0])), abs(raw_heading_sin))

    def test_validation_augmentation_is_rejected(self):
        with self.assertRaises(ValueError):
            VDS2RawDataset(self.manifest, "target_val", self.stats, augment=True)

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
                fully_padded_after_resize = F.interpolate(
                    torch.from_numpy(padding.astype(np.float32))[None, None],
                    size=(224, 224),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )[0, 0] == 1.0
                if fully_padded_after_resize.any():
                    selected = normalized[:, fully_padded_after_resize]
                    self.assertTrue(torch.equal(selected, torch.zeros_like(selected)))
                    checked = True
                    break
        self.assertTrue(checked, "Expected at least one padded source_train crop")


if __name__ == "__main__":
    unittest.main()
