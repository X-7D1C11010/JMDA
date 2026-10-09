import unittest

import numpy as np
import torch
from PIL import Image

from Generator import NeuralOptimalTransportGenerator
from Tensor import TensorBasedAlignmentStable
from Ablation.Models import Classifier
from Ablation.PairedClassSampler import PairedClassSampler
from Ablation.module_ablation import (
    BinaryDomainDiscriminator,
    compute_joint_ot_scale,
    select_report_metrics,
    transport_projected_feature_basis,
)
from Ablation.paired_dataset import PairedModalTransform
from epoch_svd import build_epoch_pairs


class _LabelOnlyDataset:
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "label": torch.tensor(self.labels[index], dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }


class EpochPairingTests(unittest.TestCase):
    def test_supervised_pairs_cover_target_once_and_match_classes(self):
        source = _LabelOnlyDataset([1, 1, 2, 2, 2])
        target = _LabelOnlyDataset([1, 2, 2, 1])
        pairs = build_epoch_pairs(source, target, seed=13, class_paired=True)

        target_indices = [target_index for _, target_index in pairs]
        self.assertEqual(sorted(target_indices), list(range(len(target))))
        self.assertEqual(len(set(target_indices)), len(target))
        self.assertTrue(
            all(
                source.labels[source_index] == target.labels[target_index]
                for source_index, target_index in pairs
            )
        )

    def test_supervised_training_sampler_covers_and_balances_target(self):
        source = _LabelOnlyDataset([1, 1, 1, 2, 2])
        target = _LabelOnlyDataset([1, 2, 2, 2])
        sampler = PairedClassSampler(source, target, batch_size=3)

        observed_target_indices = []
        observed_target_labels = []
        for source_batch, target_batch in sampler:
            self.assertTrue(
                torch.equal(source_batch["label"], target_batch["label"])
            )
            observed_target_indices.extend(target_batch["sample_index"].tolist())
            observed_target_labels.extend(target_batch["label"].tolist())

        self.assertEqual(set(observed_target_indices), set(range(len(target))))
        self.assertEqual(observed_target_labels.count(1), 3)
        self.assertEqual(observed_target_labels.count(2), 3)

    def test_paired_transform_uses_identical_random_geometry(self):
        grid = np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        image = Image.fromarray(grid, mode="L").convert("RGB")
        transform = PairedModalTransform(phase="train")
        torch.manual_seed(123)

        vis, ir = transform(image, image)
        vis_raw = vis * torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        vis_raw = vis_raw + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        ir_raw = ir * 0.5 + 0.5

        self.assertTrue(torch.allclose(vis_raw, ir_raw, atol=1e-6))


class ResultReportingTests(unittest.TestCase):
    def test_joint_ot_warmup_and_ramp_schedule(self):
        scales = [
            compute_joint_ot_scale(
                epoch, warmup_epochs=3, ramp_epochs=2, enabled=True
            )
            for epoch in range(6)
        ]
        self.assertEqual(scales, [0.0, 0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertEqual(
            compute_joint_ot_scale(0, 3, 2, enabled=False),
            1.0,
        )

    def test_best_strategy_keeps_metrics_from_one_checkpoint(self):
        history = [
            {
                "accuracy": 0.4,
                "precision_macro": 0.9,
                "recall_macro": 0.8,
                "f1_macro": 0.7,
            },
            {
                "accuracy": 0.8,
                "precision_macro": 0.3,
                "recall_macro": 0.4,
                "f1_macro": 0.35,
            },
            {
                "accuracy": 0.6,
                "precision_macro": 0.7,
                "recall_macro": 0.6,
                "f1_macro": 0.65,
            },
        ]

        selected = select_report_metrics(history, strategy="best")

        self.assertIs(selected, history[1])
        self.assertEqual(selected["accuracy"], 0.8)
        self.assertEqual(selected["precision_macro"], 0.3)


class TensorEpochUpdateTests(unittest.TestCase):
    def test_downstream_basis_transport_preserves_pure_rotation(self):
        torch.manual_seed(3)
        modal_rank = 3
        feature_dim = 2 * modal_rank
        old_source = [
            torch.linalg.qr(torch.randn(7, modal_rank)).Q,
            torch.linalg.qr(torch.randn(8, modal_rank)).Q,
        ]
        old_target = [matrix.clone() for matrix in old_source]
        rotations = [
            torch.linalg.qr(torch.randn(modal_rank, modal_rank)).Q
            for _ in range(2)
        ]
        new_source = [
            projection.matmul(rotation)
            for projection, rotation in zip(old_source, rotations)
        ]
        new_target = [matrix.clone() for matrix in new_source]
        basis_map = torch.block_diag(*rotations)

        classifier = Classifier(input_dim=feature_dim, num_classes=4).eval()
        discriminator = BinaryDomainDiscriminator(feature_dim=feature_dim).eval()
        generator = NeuralOptimalTransportGenerator(
            feature_dim=feature_dim,
            hidden_dim=10,
            transport_mode="sinkhorn",
        ).eval()
        old_features = torch.randn(5, feature_dim)
        new_features = old_features.matmul(basis_map)

        classifier_expected = classifier(old_features)
        discriminator_expected = discriminator(old_features)
        cost_expected = generator.cost_net(old_features, old_features)
        residual_input = torch.cat(
            [old_features, old_features, torch.full((5, 1), 0.4)], dim=1
        )
        residual_expected = generator.mlp[:3](residual_input).matmul(basis_map)

        diagnostics = transport_projected_feature_basis(
            classifier,
            discriminator,
            generator,
            old_source,
            old_target,
            new_source,
            new_target,
        )

        self.assertTrue(
            torch.allclose(classifier(new_features), classifier_expected, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(
                discriminator(new_features), discriminator_expected, atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                generator.cost_net(new_features, new_features),
                cost_expected,
                atol=1e-5,
            )
        )
        new_residual_input = torch.cat(
            [new_features, new_features, torch.full((5, 1), 0.4)], dim=1
        )
        self.assertTrue(
            torch.allclose(
                generator.mlp[:3](new_residual_input), residual_expected, atol=1e-5
            )
        )
        self.assertAlmostEqual(
            diagnostics["source_subspace_overlap"], 1.0, places=5
        )
        self.assertAlmostEqual(
            diagnostics["target_subspace_overlap"], 1.0, places=5
        )
        self.assertLess(diagnostics["basis_map_non_orthogonality"], 1e-5)

    def test_paired_procrustes_alignment_removes_subspace_rotation(self):
        torch.manual_seed(5)
        new_source = torch.linalg.qr(torch.randn(9, 4)).Q
        new_target = torch.linalg.qr(torch.randn(8, 4)).Q
        rotation = torch.linalg.qr(torch.randn(4, 4)).Q
        old_source = new_source.matmul(rotation)
        old_target = new_target.matmul(rotation)

        aligned_source, aligned_target = (
            TensorBasedAlignmentStable._align_paired_subspace(
                new_source,
                new_target,
                old_source,
                old_target,
            )
        )

        self.assertTrue(torch.allclose(aligned_source, old_source, atol=1e-5))
        self.assertTrue(torch.allclose(aligned_target, old_target, atol=1e-5))

    def test_epoch_update_is_orthogonal_and_forward_is_read_only(self):
        torch.manual_seed(7)
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6],
            output_dims=[4, 4],
            num_modalities=2,
            max_svd_sweeps=3,
        )
        source_bank = [torch.randn(16, 8), torch.randn(16, 6)]
        target_bank = [
            source_bank[0] + 0.05 * torch.randn(16, 8),
            source_bank[1] + 0.05 * torch.randn(16, 6),
        ]

        info = module.update_projections(source_bank, target_bank)
        self.assertEqual(info["update_count"], 1)
        self.assertEqual(info["sample_count"], 16)
        for projection in (*module.U_matrices, *module.V_matrices):
            identity = torch.eye(projection.shape[1])
            self.assertTrue(
                torch.allclose(
                    projection.transpose(0, 1).matmul(projection),
                    identity,
                    atol=1e-5,
                    rtol=1e-5,
                )
            )

        before = [matrix.clone() for matrix in (*module.U_matrices, *module.V_matrices)]
        source_batch = [torch.randn(5, 8, requires_grad=True), torch.randn(5, 6, requires_grad=True)]
        target_batch = [torch.randn(5, 8, requires_grad=True), torch.randn(5, 6, requires_grad=True)]
        _, _, loss = module(source_batch, target_batch)
        loss.backward()

        self.assertTrue(all(feature.grad is not None for feature in source_batch + target_batch))
        after = [matrix for matrix in (*module.U_matrices, *module.V_matrices)]
        self.assertTrue(all(torch.equal(old, new) for old, new in zip(before, after)))
        self.assertEqual(int(module.projection_update_count.item()), 1)

    def test_per_batch_update_is_rejected(self):
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6], output_dims=[4, 4], num_modalities=2
        )
        source = [torch.randn(5, 8), torch.randn(5, 6)]
        target = [torch.randn(5, 8), torch.randn(5, 6)]
        with self.assertRaisesRegex(RuntimeError, "Per-mini-batch"):
            module(source, target, update_projections=True)

    def test_rank_guard_rejects_unsupported_projection(self):
        module = TensorBasedAlignmentStable(
            input_dims=[8, 6], output_dims=[4, 4], num_modalities=2
        )
        source = [torch.randn(4, 8), torch.randn(4, 6)]
        target = [torch.randn(4, 8), torch.randn(4, 6)]
        with self.assertRaisesRegex(ValueError, "maximum centered rank=3"):
            module.update_projections(source, target)


class SinkhornTransportTests(unittest.TestCase):
    def test_legacy_mode_keeps_original_transnet_row_softmax(self):
        torch.manual_seed(9)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=10,
            hidden_dim=12,
            transport_mode="legacy_row_softmax",
        )
        generator.eval()
        source = torch.randn(6, 10)
        target = torch.randn(6, 10)

        cost = generator.cost_net(source, target)
        expected = torch.softmax(generator.transmission_net(cost), dim=1)
        _, details = generator(source, target, return_details=True)
        self.assertTrue(
            torch.allclose(details["conditional_plan"], expected, atol=1e-7)
        )

    def test_sinkhorn_enforces_both_marginals_and_backpropagates(self):
        torch.manual_seed(11)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=12,
            hidden_dim=16,
            transport_mode="sinkhorn",
            epsilon=0.1,
            sinkhorn_iterations=100,
            correction_scale=0.1,
        )
        source = torch.randn(7, 12, requires_grad=True)
        target = torch.randn(5, 12, requires_grad=True)
        intermediate, details = generator(source, target, return_details=True)

        plan = details["mass_plan"]
        expected_rows = details["source_marginal"]
        expected_columns = details["target_marginal"]
        self.assertTrue(
            torch.allclose(
                details["softmax_kernel"].sum(dim=1),
                torch.ones(7),
                atol=1e-6,
            )
        )
        self.assertTrue(torch.allclose(plan.sum(dim=1), expected_rows, atol=2e-4))
        self.assertTrue(torch.allclose(plan.sum(dim=0), expected_columns, atol=2e-4))
        self.assertLess(float(details["marginal_residual"]), 2e-4)
        self.assertTrue(
            torch.allclose(
                details["conditional_plan"].sum(dim=1),
                torch.ones(7),
                atol=2e-4,
            )
        )

        loss = intermediate.square().mean() + details[
            "correction_regularization"
        ]
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNotNone(target.grad)
        transnet_gradients = [
            parameter.grad
            for parameter in generator.transmission_net.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(any(gradient is not None for gradient in transnet_gradients))

    def test_supervised_sinkhorn_has_zero_off_class_mass(self):
        torch.manual_seed(17)
        generator = NeuralOptimalTransportGenerator(
            feature_dim=6,
            hidden_dim=8,
            transport_mode="sinkhorn",
            epsilon=0.1,
            sinkhorn_iterations=100,
        )
        source = torch.randn(4, 6)
        target = torch.randn(4, 6)
        labels = torch.tensor([0, 0, 1, 1])

        _, details = generator(
            source,
            target,
            return_details=True,
            source_labels=labels,
            target_labels=labels,
        )

        plan = details["mass_plan"]
        off_class = labels[:, None].ne(labels[None, :])
        self.assertEqual(float(plan.masked_select(off_class).sum()), 0.0)
        self.assertLess(float(details["marginal_residual"]), 2e-4)


if __name__ == "__main__":
    unittest.main()
