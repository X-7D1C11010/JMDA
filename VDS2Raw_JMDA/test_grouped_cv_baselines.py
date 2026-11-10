from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torchvision import models

from VDS2Raw_JMDA.grouped_cv_baselines import (
    FoldManifest,
    build_grouped_folds,
    build_sample_stratified_folds,
    connected_group_ids,
    fold_audit,
    set_last_stage_trainable,
    target_priority_sampler,
)
from VDS2Raw_JMDA.vds2raw_dataset import VDS2RawManifest


DATASET_ROOT = Path(r"D:\Downloads\VDS2Raw\threeclass_ready")


class ConnectedGroupTests(unittest.TestCase):
    def test_scene_and_mmsi_are_joined_transitively(self) -> None:
        records = [
            SimpleNamespace(sample_id="a", scene="s1", mmsi="m1"),
            SimpleNamespace(sample_id="b", scene="s1", mmsi="m2"),
            SimpleNamespace(sample_id="c", scene="s2", mmsi="m2"),
            SimpleNamespace(sample_id="d", scene="s3", mmsi=""),
        ]
        groups = connected_group_ids(records)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])


class SamplerTests(unittest.TestCase):
    def test_target_priority_weight_mass_is_two_to_one(self) -> None:
        sampler = target_priority_sampler(116, 104, 2.0, 42)
        weights = sampler.weights
        source_mass = float(weights[:116].sum())
        target_mass = float(weights[116:].sum())
        self.assertAlmostEqual(source_mass, 1.0, places=12)
        self.assertAlmostEqual(target_mass, 2.0, places=12)
        self.assertEqual(sampler.num_samples, 220)


class TrainableScopeTests(unittest.TestCase):
    def test_resnet_only_head_and_layer4_are_trainable(self) -> None:
        model = models.resnet18(weights=None, num_classes=3)
        set_last_stage_trainable(model, "resnet")
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any(name.startswith("layer4.") for name in trainable))
        self.assertTrue(any(name.startswith("fc.") for name in trainable))
        self.assertFalse(any(name.startswith("layer3.") for name in trainable))
        self.assertFalse(any(name.startswith("conv1.") for name in trainable))


@unittest.skipUnless(DATASET_ROOT.is_dir(), "VDS2Raw dataset is not available")
class ActualDatasetFoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = VDS2RawManifest(DATASET_ROOT)
        cls.folds = build_grouped_folds(cls.base, split_seed=20261110)
        cls.sample_folds = build_sample_stratified_folds(
            cls.base, split_seed=20261110
        )

    def test_five_folds_cover_every_target_exactly_once(self) -> None:
        coverage: Counter[str] = Counter()
        target_ids = {record.sample_id for record in self.base.records if record.day != "day3"}
        for fold in self.folds:
            coverage.update(
                sample_id
                for sample_id, partition in fold["assignments"].items()
                if partition == "target_test"
            )
        self.assertEqual(set(coverage), target_ids)
        self.assertEqual(set(coverage.values()), {1})

    def test_each_fold_passes_scene_and_mmsi_leakage_audit(self) -> None:
        for fold in self.folds:
            manifest = FoldManifest(self.base, fold["assignments"])
            audit = fold_audit(manifest, int(fold["fold"]))
            self.assertEqual(audit["status"], "PASS")
            for overlap in audit["overlaps"].values():
                self.assertEqual(overlap, {"sample": 0, "scene": 0, "mmsi": 0})

    def test_fold_partitions_have_expected_scope_and_all_classes(self) -> None:
        for fold in self.folds:
            manifest = FoldManifest(self.base, fold["assignments"])
            self.assertEqual(len(manifest.by_partition["source_train"]), 116)
            self.assertEqual(
                sum(len(manifest.by_partition[name]) for name in ("target_train", "target_val", "target_test")),
                155,
            )
            self.assertLessEqual(abs(len(manifest.by_partition["target_val"]) - 20), 2)
            for partition in ("source_train", "target_train", "target_val", "target_test"):
                self.assertEqual(
                    {record.label for record in manifest.by_partition[partition]},
                    {"Cargo", "Fishing", "Sailing & Pleasure"},
                )

    def test_sample_stratified_comparison_is_explicitly_warning_marked(self) -> None:
        coverage: Counter[str] = Counter()
        for fold in self.sample_folds:
            manifest = FoldManifest(self.base, fold["assignments"])
            self.assertEqual(len(manifest.by_partition["target_train"]), 104)
            self.assertEqual(len(manifest.by_partition["target_val"]), 20)
            self.assertEqual(len(manifest.by_partition["target_test"]), 31)
            audit = fold_audit(
                manifest,
                int(fold["fold"]),
                allow_scene_mmsi_overlap=True,
            )
            self.assertEqual(audit["status"], "WARNING_SCENE_OR_MMSI_OVERLAP")
            self.assertTrue(audit["warnings"])
            coverage.update(
                record.sample_id for record in manifest.by_partition["target_test"]
            )
        self.assertEqual(len(coverage), 155)
        self.assertEqual(set(coverage.values()), {1})


if __name__ == "__main__":
    unittest.main()
