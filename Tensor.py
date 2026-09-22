"""Tensor alignment with alternating closed-form SVD projection updates.

The encoders remain end-to-end trainable. The source/target projection
matrices are not optimizer parameters: they are updated from the current
mini-batch by alternating over tensor modes and solving each conditional
subproblem with an SVD of the source-target cross-covariance matrix.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FactorizedOuterProductTensor:
    """Exact rank-one representation of per-sample multimodal outer products.

    For factors ``x_1, ..., x_M`` this object represents
    ``x_1 ⨀ ... ⨀ x_M`` without allocating the exponentially large
    dense tensor. All contractions used by TAL are evaluated exactly from the
    factors, which makes the same method practical for VIS/IR/AIS inputs.
    """

    factors: Tuple[torch.Tensor, ...]

    def detach(self) -> "FactorizedOuterProductTensor":
        return FactorizedOuterProductTensor(
            tuple(factor.detach() for factor in self.factors)
        )


class TensorBasedAlignmentStable(nn.Module):
    """High-order tensor alignment using alternating SVD updates.

    Each domain first constructs the sample-wise outer product of all modal
    features. When mode ``m`` is updated, all other projection matrices are
    held fixed and used to contract their corresponding modes. The left/right
    singular vectors of the resulting source-target cross-covariance directly
    replace ``U_m`` and ``V_m``.

    ``U_m`` and ``V_m`` are buffers rather than ``Parameter`` objects. Thus,
    SVD is their only update rule; back-propagation updates the encoders while
    treating the projections as fixed during that optimizer step.
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dims: List[int],
        num_modalities: int = 3,
        max_svd_sweeps: int = 3,
        svd_tolerance: float = 1e-4,
        eps: float = 1e-8,
    ):
        super().__init__()

        if len(input_dims) != num_modalities or len(output_dims) != num_modalities:
            raise ValueError("input_dims/output_dims must match num_modalities.")
        if max_svd_sweeps < 1:
            raise ValueError("max_svd_sweeps must be at least 1.")
        if svd_tolerance <= 0 or eps <= 0:
            raise ValueError("svd_tolerance and eps must be positive.")
        for mode, (input_dim, output_dim) in enumerate(zip(input_dims, output_dims)):
            if output_dim > input_dim:
                raise ValueError(
                    f"Mode {mode}: output_dim ({output_dim}) cannot exceed "
                    f"input_dim ({input_dim})."
                )

        self.num_modalities = num_modalities
        self.input_dims = list(input_dims)
        self.output_dims = list(output_dims)
        self.max_svd_sweeps = max_svd_sweeps
        self.svd_tolerance = svd_tolerance
        self.eps = eps
        self.last_update_info: Dict[str, object] = {}

        # Buffers are checkpointed and moved by module.to(device), but are
        # deliberately excluded from gradient-based optimizers.
        for mode, (input_dim, output_dim) in enumerate(zip(input_dims, output_dims)):
            source_projection = torch.empty(input_dim, output_dim)
            nn.init.orthogonal_(source_projection)
            # Start both domains in the same latent coordinate system. Every
            # subsequent SVD update produces a paired left/right basis.
            target_projection = source_projection.clone()
            self.register_buffer(f"U_{mode}", source_projection)
            self.register_buffer(f"V_{mode}", target_projection)
            self.register_buffer(f"singular_values_{mode}", torch.zeros(output_dim))

    @property
    def U_matrices(self) -> Tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"U_{mode}") for mode in range(self.num_modalities))

    @property
    def V_matrices(self) -> Tuple[torch.Tensor, ...]:
        return tuple(getattr(self, f"V_{mode}") for mode in range(self.num_modalities))

    def _validate_modalities(
        self,
        source_modalities: Sequence[torch.Tensor],
        target_modalities: Sequence[torch.Tensor],
    ) -> None:
        if len(source_modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} source modalities, "
                f"got {len(source_modalities)}."
            )
        if len(target_modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} target modalities, "
                f"got {len(target_modalities)}."
            )

        source_batch = source_modalities[0].shape[0]
        target_batch = target_modalities[0].shape[0]
        if source_batch != target_batch:
            raise ValueError(
                "Alternating SVD requires paired source/target mini-batches "
                f"of equal size, got {source_batch} and {target_batch}."
            )

        for mode, (source, target, expected_dim) in enumerate(
            zip(source_modalities, target_modalities, self.input_dims)
        ):
            if source.ndim != 2 or target.ndim != 2:
                raise ValueError(f"Mode {mode} features must be rank-2 [batch, dim].")
            if source.shape != target.shape:
                raise ValueError(
                    f"Mode {mode} source/target shapes differ: "
                    f"{tuple(source.shape)} vs {tuple(target.shape)}."
                )
            if source.shape[1] != expected_dim:
                raise ValueError(
                    f"Mode {mode} expected feature dim {expected_dim}, "
                    f"got {source.shape[1]}."
                )

    def create_multimodal_tensor(
        self,
        modalities: Sequence[torch.Tensor],
    ) -> FactorizedOuterProductTensor:
        """Construct the exact factorized form of Eq. (4)'s outer product."""
        return FactorizedOuterProductTensor(tuple(modalities))

    @staticmethod
    def mode_n_product(
        tensor: torch.Tensor,
        matrix: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        """Multiply tensor mode ``mode`` (excluding batch) by a projection."""
        tensor_axis = mode + 1
        if tensor.shape[tensor_axis] != matrix.shape[0]:
            raise ValueError(
                f"Mode {mode} dimension mismatch: "
                f"{tensor.shape[tensor_axis]} vs {matrix.shape[0]}."
            )
        moved = tensor.movedim(tensor_axis, -1)
        projected = torch.matmul(moved, matrix)
        return projected.movedim(-1, tensor_axis)

    def _contract_other_modes(
        self,
        tensor: FactorizedOuterProductTensor,
        matrices: Sequence[torch.Tensor],
        current_mode: int,
    ) -> torch.Tensor:
        """Fix/project every mode except ``current_mode``.

        The remaining projected axes are summed out, which is the matrix-basis
        generalization of Eq. (5)'s contraction by projection vectors. The
        returned aggregated feature matrix has shape ``[B, d_m]``.
        """
        current_features = tensor.factors[current_mode]
        contraction_weight = current_features.new_ones(
            current_features.shape[0], 1
        )
        for mode, (features, matrix) in enumerate(zip(tensor.factors, matrices)):
            if mode == current_mode:
                continue
            # Dense equivalence:
            #   (x_1 o ... o x_M) x_j U_j, followed by contraction of the
            #   projected axis. Mean differs from sum only by a constant rank
            #   scale and avoids numerical growth for three or more modalities.
            projected = features.matmul(matrix)
            contraction_weight = contraction_weight * projected.mean(
                dim=1, keepdim=True
            )
        return current_features * contraction_weight

    @staticmethod
    def _cross_covariance(
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        if source_features.shape[0] != target_features.shape[0]:
            raise ValueError("Contracted source/target observation counts must match.")
        source_centered = source_features - source_features.mean(dim=0, keepdim=True)
        target_centered = target_features - target_features.mean(dim=0, keepdim=True)
        denominator = max(source_centered.shape[0] - 1, 1)
        covariance = source_centered.transpose(0, 1).matmul(target_centered) / denominator
        if not torch.isfinite(covariance).all():
            raise FloatingPointError("Non-finite values encountered in SVD cross-covariance.")
        return covariance

    @staticmethod
    def _orient_singular_vectors(
        new_source: torch.Tensor,
        new_target: torch.Tensor,
        old_source: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resolve the arbitrary joint sign of each singular-vector pair."""
        signs = torch.sign((new_source * old_source).sum(dim=0))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        return new_source * signs, new_target * signs

    @torch.no_grad()
    def _alternating_svd_update(
        self,
        source_tensor: FactorizedOuterProductTensor,
        target_tensor: FactorizedOuterProductTensor,
    ) -> Dict[str, object]:
        """Run Gauss-Seidel mode updates until singular values converge."""
        source_tensor = source_tensor.detach()
        target_tensor = target_tensor.detach()
        previous_singular_values = None
        relative_change = float("inf")
        converged = False

        for sweep in range(self.max_svd_sweeps):
            current_singular_values = []

            for mode in range(self.num_modalities):
                # Other modal projections remain fixed at their latest values.
                source_aggregated = self._contract_other_modes(
                    source_tensor, self.U_matrices, mode
                )
                target_aggregated = self._contract_other_modes(
                    target_tensor, self.V_matrices, mode
                )
                covariance = self._cross_covariance(source_aggregated, target_aggregated)

                left, singular_values, right_h = torch.linalg.svd(
                    covariance, full_matrices=False
                )
                rank = self.output_dims[mode]
                new_source = left[:, :rank]
                new_target = right_h.transpose(-2, -1)[:, :rank]
                new_source, new_target = self._orient_singular_vectors(
                    new_source, new_target, self.U_matrices[mode]
                )

                self.U_matrices[mode].copy_(new_source)
                self.V_matrices[mode].copy_(new_target)
                retained = singular_values[:rank]
                getattr(self, f"singular_values_{mode}").copy_(retained)
                current_singular_values.append(retained.clone())

            if previous_singular_values is not None:
                changes = []
                for current, previous in zip(current_singular_values, previous_singular_values):
                    numerator = torch.linalg.vector_norm(current - previous)
                    denominator = torch.linalg.vector_norm(previous).clamp_min(self.eps)
                    changes.append((numerator / denominator).item())
                relative_change = max(changes)
                if relative_change < self.svd_tolerance:
                    converged = True
                    break

            previous_singular_values = current_singular_values

        return {
            "sweeps": sweep + 1,
            "converged": converged,
            "relative_singular_change": relative_change,
            "singular_values": tuple(values.detach().clone() for values in current_singular_values),
        }

    def _mean_cosine_similarity(
        self,
        source_projected: torch.Tensor,
        target_projected: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (7): batch mean of paired-sample cosine similarities."""
        return F.cosine_similarity(
            source_projected,
            target_projected,
            dim=1,
            eps=self.eps,
        ).mean()

    def forward(
        self,
        source_modalities: List[torch.Tensor],
        target_modalities: List[torch.Tensor],
        update_projections: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """Build domain tensors, optionally update SVD bases, and align them.

        Training code passes ``update_projections=True``. Evaluation uses the
        last training update and leaves it at the default False.
        """
        self._validate_modalities(source_modalities, target_modalities)

        # Paper-ordered step: each domain first constructs its own high-order
        # multimodal tensor using sample-wise outer products.
        source_tensor = self.create_multimodal_tensor(source_modalities)
        target_tensor = self.create_multimodal_tensor(target_modalities)

        if update_projections:
            self.last_update_info = self._alternating_svd_update(source_tensor, target_tensor)

        projected_source = [
            feature.matmul(projection)
            for feature, projection in zip(source_modalities, self.U_matrices)
        ]
        projected_target = [
            feature.matmul(projection)
            for feature, projection in zip(target_modalities, self.V_matrices)
        ]
        correlations = [
            self._mean_cosine_similarity(source, target)
            for source, target in zip(projected_source, projected_target)
        ]
        # Eq. (8): negative mean of the modality-wise correlations.
        alignment_loss = -torch.stack(correlations).mean()
        return projected_source, projected_target, alignment_loss
