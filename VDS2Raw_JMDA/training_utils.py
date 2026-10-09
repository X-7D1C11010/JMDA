"""Training utilities isolated for the VDS2Raw JMDA-Net pipeline."""

from __future__ import annotations

import random
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = requires_grad


def compute_joint_ot_scale(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    joint_tensor_ot: bool,
) -> float:
    if warmup_epochs < 0 or ramp_epochs < 1:
        raise ValueError("warmup_epochs must be >= 0 and ramp_epochs must be >= 1")
    if not joint_tensor_ot:
        return 1.0
    if epoch < warmup_epochs:
        return 0.0
    return min((epoch - warmup_epochs + 1) / ramp_epochs, 1.0)


def binary_domain_loss(
    source_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> torch.Tensor:
    source_labels = torch.zeros_like(source_logits)
    target_labels = torch.ones_like(target_logits)
    return 0.5 * (
        F.binary_cross_entropy_with_logits(source_logits, source_labels)
        + F.binary_cross_entropy_with_logits(target_logits, target_labels)
    )


def _reset_optimizer_state(
    optimizer: Optional[torch.optim.Optimizer],
    parameters: Iterable[nn.Parameter],
) -> None:
    if optimizer is None:
        return
    for parameter in parameters:
        optimizer.state.pop(parameter, None)


@torch.no_grad()
def transport_projected_feature_basis(
    classifier: nn.Module,
    discriminator: nn.Module,
    generator: Optional[nn.Module],
    old_source_projections: Sequence[torch.Tensor],
    old_target_projections: Sequence[torch.Tensor],
    new_source_projections: Sequence[torch.Tensor],
    new_target_projections: Sequence[torch.Tensor],
    optimizer_g: Optional[torch.optim.Optimizer] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Transport downstream weights after an epoch-SVD basis update."""

    groups = (
        old_source_projections,
        old_target_projections,
        new_source_projections,
        new_target_projections,
    )
    if len({len(group) for group in groups}) != 1:
        raise ValueError("Old/new source/target projection counts must match")

    blocks = []
    source_overlaps = []
    target_overlaps = []
    for old_source, old_target, new_source, new_target in zip(*groups):
        if old_source.shape != new_source.shape or old_target.shape != new_target.shape:
            raise ValueError("Old/new projection shapes must match")
        source_map = old_source.transpose(0, 1).matmul(new_source)
        target_map = old_target.transpose(0, 1).matmul(new_target)
        blocks.append(0.5 * (source_map + target_map))
        rank = max(source_map.shape[0], 1)
        source_overlaps.append(float(source_map.square().sum().div(rank).clamp(0.0, 1.0)))
        target_overlaps.append(float(target_map.square().sum().div(rank).clamp(0.0, 1.0)))

    basis_map = torch.block_diag(*blocks)
    feature_dim = basis_map.shape[0]
    transformed = []

    def transport_input(linear: nn.Linear) -> None:
        if not isinstance(linear, nn.Linear) or linear.in_features != feature_dim:
            raise ValueError("Projected-feature consumer has incompatible input width")
        linear.weight.copy_(linear.weight.matmul(basis_map))
        transformed.append(linear.weight)

    transport_input(classifier.fc[0])
    transport_input(discriminator.discriminator[0])

    if generator is not None:
        transport_input(generator.cost_net.proj[0])
        residual_input = generator.mlp[0]
        if residual_input.in_features != 2 * feature_dim + 1:
            raise ValueError("Neural-OT residual input width is incompatible")
        residual_input.weight[:, :feature_dim].copy_(
            residual_input.weight[:, :feature_dim].matmul(basis_map)
        )
        residual_input.weight[:, feature_dim : 2 * feature_dim].copy_(
            residual_input.weight[:, feature_dim : 2 * feature_dim].matmul(basis_map)
        )
        transformed.append(residual_input.weight)

        residual_output = generator.mlp[2]
        if residual_output.out_features != feature_dim:
            raise ValueError("Neural-OT residual output width is incompatible")
        residual_output.weight.copy_(basis_map.transpose(0, 1).matmul(residual_output.weight))
        transformed.append(residual_output.weight)
        if residual_output.bias is not None:
            residual_output.bias.copy_(basis_map.transpose(0, 1).matmul(residual_output.bias))
            transformed.append(residual_output.bias)

    discriminator_parameters = set(discriminator.parameters())
    _reset_optimizer_state(
        optimizer_g,
        [parameter for parameter in transformed if parameter not in discriminator_parameters],
    )
    _reset_optimizer_state(
        optimizer_d,
        [parameter for parameter in transformed if parameter in discriminator_parameters],
    )
    non_orthogonality = torch.linalg.matrix_norm(
        basis_map.transpose(0, 1).matmul(basis_map)
        - torch.eye(feature_dim, device=basis_map.device, dtype=basis_map.dtype)
    ) / max(feature_dim, 1)
    return {
        "source_subspace_overlap": float(np.mean(source_overlaps)),
        "target_subspace_overlap": float(np.mean(target_overlaps)),
        "basis_map_non_orthogonality": float(non_orthogonality),
    }


__all__ = [
    "seed_everything",
    "set_requires_grad",
    "compute_joint_ot_scale",
    "binary_domain_loss",
    "transport_projected_feature_basis",
]
