import unittest

import torch

from Generator import NeuralOptimalTransportGenerator
from Tensor import TensorBasedAlignmentStable
from epoch_svd import build_epoch_pairs


class _LabelOnlyDataset:
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)


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


class TensorEpochUpdateTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
