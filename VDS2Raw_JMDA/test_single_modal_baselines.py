"""Tests for the VDS2Raw single-modality baseline experiment."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import torch


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VDS2Raw_JMDA.single_modal_baselines import (
    AIS_INPUT_DIM,
    AISConvNeXt,
    AISResNet,
    AISViT,
    SingleModalityDataset,
    architecture_name,
    build_parser,
    checkpoint_key,
    classification_metrics,
    domain_class_balanced_sampler,
    replace_input_conv,
)
from VDS2Raw_JMDA.vds2raw_dataset import (
    TRAIN_PARTITIONS,
    VDS2RawManifest,
    compute_train_image_stats,
)


DATASET_ROOT = Path(
    os.environ.get("VDS2RAW_ROOT", r"D:\Downloads\VDS2Raw\threeclass_ready")
)


class MetricTests(unittest.TestCase):
    def test_acc_and_macro_metrics(self):
        metrics = classification_metrics([0, 1, 2, 2], [0, 1, 1, 2])
        self.assertAlmostEqual(metrics["acc"], 0.75)
        self.assertAlmostEqual(metrics["precision_macro"], (1.0 + 0.5 + 1.0) / 3)
        self.assertAlmostEqual(metrics["recall_macro"], (1.0 + 1.0 + 0.5) / 3)
        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [0, 1, 1]])

    def test_tuned_image_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.lr_image, 3e-4)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.patience, 20)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.image_size, 96)
        self.assertEqual(args.train_resize_size, 112)
        self.assertEqual(args.label_smoothing, 0.05)
        self.assertFalse(args.amp)
        self.assertEqual(args.amp_dtype, "bfloat16")
        self.assertFalse(args.channels_last)
        self.assertEqual(args.backbone_profile, "baseline")
        self.assertFalse(args.pretrained)
        self.assertEqual(args.freeze_backbone_epochs, 0)
        self.assertEqual(args.checkpoint_metric, "acc")
        self.assertFalse(args.domain_class_balanced_sampling)
        self.assertIsNone(args.lr_backbone)

    def test_macro_f1_checkpoint_order_is_not_accuracy_order(self):
        high_acc = {"acc": 0.8, "f1_macro": 0.4}
        high_f1 = {"acc": 0.7, "f1_macro": 0.6}
        self.assertGreater(
            checkpoint_key(high_acc, "acc"), checkpoint_key(high_f1, "acc")
        )
        self.assertGreater(
            checkpoint_key(high_f1, "macro_f1"),
            checkpoint_key(high_acc, "macro_f1"),
        )

    def test_domain_class_sampler_equalizes_cell_mass(self):
        labels = [0, 0, 1, 2, 0, 1, 1, 2]
        domains = [0, 0, 0, 0, 1, 1, 1, 1]
        sampler = domain_class_balanced_sampler(labels, domains, seed=42)
        masses = {}
        for domain, label, weight in zip(domains, labels, sampler.weights.tolist()):
            masses[(domain, label)] = masses.get((domain, label), 0.0) + weight
        self.assertEqual(set(masses), {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)})
        for mass in masses.values():
            self.assertAlmostEqual(mass, 1.0)

    def test_pretrained_nir_stem_matches_rgb_replication(self):
        source = torch.nn.Conv2d(3, 2, kernel_size=1, bias=True)
        with torch.no_grad():
            source.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3, 1, 1))
            source.bias.copy_(torch.tensor([0.25, -0.5]))
        adapted = replace_input_conv(source, 1, preserve_pretrained=True)
        self.assertTrue(torch.equal(adapted.weight, source.weight.sum(dim=1, keepdim=True)))
        self.assertTrue(torch.equal(adapted.bias, source.bias))

    def test_transfer_profile_uses_smaller_convolutional_backbones(self):
        self.assertEqual(
            architecture_name("resnet", "rgb", "transfer-small"),
            "torchvision ResNet-18",
        )
        self.assertEqual(
            architecture_name("convnext", "nir", "transfer-small"),
            "torchvision ConvNeXt-Small",
        )


class AISModelTests(unittest.TestCase):
    def test_all_architecture_families_accept_heading_only_input(self):
        self.assertEqual(AIS_INPUT_DIM, 3)
        inputs = torch.randn(3, AIS_INPUT_DIM)
        with torch.no_grad():
            for model_class in (AISResNet, AISViT, AISConvNeXt):
                outputs = model_class(num_classes=3)(inputs)
                self.assertEqual(tuple(outputs.shape), (3, 3))


@unittest.skipUnless(DATASET_ROOT.is_dir(), "prepared VDS2Raw dataset is unavailable")
class PreparedDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = VDS2RawManifest(DATASET_ROOT)
        cls.image_stats = compute_train_image_stats(cls.manifest)

    def test_train_only_statistics_and_native_image_shapes(self):
        self.assertEqual(tuple(self.image_stats["fit_partitions"]), TRAIN_PARTITIONS)
        rgb = SingleModalityDataset(
            self.manifest,
            "target_val",
            self.image_stats,
            "rgb",
            augment=False,
        )[0][0]
        nir = SingleModalityDataset(
            self.manifest,
            "target_val",
            self.image_stats,
            "nir",
            augment=False,
        )[0][0]
        self.assertEqual(tuple(rgb.shape), (3, 96, 96))
        self.assertEqual(tuple(nir.shape), (1, 96, 96))

    def test_configurable_224_image_shape_without_random_crop(self):
        rgb = SingleModalityDataset(
            self.manifest,
            "source_train",
            self.image_stats,
            "rgb",
            augment=True,
            image_size=224,
            train_resize_size=224,
        )[0][0]
        nir = SingleModalityDataset(
            self.manifest,
            "target_val",
            self.image_stats,
            "nir",
            augment=False,
            image_size=224,
            train_resize_size=224,
        )[0][0]
        self.assertEqual(tuple(rgb.shape), (3, 224, 224))
        self.assertEqual(tuple(nir.shape), (1, 224, 224))

    def test_ais_view_contains_only_requested_values_and_masks(self):
        dataset = SingleModalityDataset(
            self.manifest,
            "target_val",
            self.image_stats,
            "ais",
            augment=False,
        )
        inputs, label, sample_id = dataset[0]
        source = dataset.base[0]
        heading_valid = torch.minimum(source["ais_mask"][0], source["ais_mask"][1])
        expected = torch.cat([source["ais"][[0, 1]], heading_valid.unsqueeze(0)])
        self.assertEqual(tuple(inputs.shape), (AIS_INPUT_DIM,))
        self.assertTrue(torch.equal(inputs, expected))
        self.assertEqual(int(label), int(source["label"]))
        self.assertEqual(sample_id, source["sample_id"])


if __name__ == "__main__":
    unittest.main()
