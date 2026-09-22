"""Neural entropic optimal transport and legacy row-Softmax ablations."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


TRANSPORT_MODES = ("legacy_row_softmax", "row_softmax", "sinkhorn")


class CostNet(nn.Module):
    """Learn a metric space and construct the squared Euclidean cost matrix."""

    def __init__(self, feature_dim: int):
        super().__init__()
        hidden_dim = max(feature_dim // 2, 1)
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        source_mapped = self.proj(source_features)
        target_mapped = self.proj(target_features)
        return (
            source_mapped.unsqueeze(1) - target_mapped.unsqueeze(0)
        ).square().sum(dim=2)


class TransmissionNetwork(nn.Module):
    """Predict batch-dependent logits used as transport-cost corrections."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        if cost_matrix.ndim != 2:
            raise ValueError("The transport cost matrix must have shape [B_s, B_t].")
        source_count, target_count = cost_matrix.shape
        logits = self.net(cost_matrix.reshape(1, 1, source_count, target_count))
        return logits.reshape(source_count, target_count)


class NeuralOptimalTransportGenerator(nn.Module):
    """Learn a cost and construct an intermediate domain through transport.

    Modes:
      - ``legacy_row_softmax`` reproduces the original TransNet + row-Softmax
        implementation and is retained only as the historical ablation.
      - ``row_softmax`` uses the revised geometric cost plus neural correction,
        but omits the target-marginal Sinkhorn projection.
      - ``sinkhorn`` is the main method: TransNet corrects the cost, Softmax
        creates a positive row-normalized kernel, and differentiable Sinkhorn
        scaling enforces both source and target marginals.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        transport_mode: str = "sinkhorn",
        epsilon: float = 0.1,
        sinkhorn_iterations: int = 50,
        correction_scale: float = 0.1,
        numerical_eps: float = 1e-8,
    ):
        super().__init__()
        if transport_mode not in TRANSPORT_MODES:
            raise ValueError(
                f"Unknown transport_mode={transport_mode!r}; "
                f"expected one of {TRANSPORT_MODES}."
            )
        if epsilon <= 0 or numerical_eps <= 0:
            raise ValueError("epsilon and numerical_eps must be positive.")
        if sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be at least 1.")
        if correction_scale < 0:
            raise ValueError("correction_scale cannot be negative.")

        self.transport_mode = transport_mode
        self.epsilon = float(epsilon)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.correction_scale = float(correction_scale)
        self.numerical_eps = float(numerical_eps)

        self.cost_net = CostNet(feature_dim)
        self.transmission_net = TransmissionNetwork()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh(),
        )
        self.last_transport_info: Dict[str, float] = {}

    def _sinkhorn_from_log_kernel(
        self,
        log_kernel: torch.Tensor,
        source_marginal: torch.Tensor,
        target_marginal: torch.Tensor,
    ) -> torch.Tensor:
        """Log-domain Sinkhorn scaling with a fixed differentiable iteration count."""
        log_source = torch.log(source_marginal.clamp_min(self.numerical_eps))
        log_target = torch.log(target_marginal.clamp_min(self.numerical_eps))
        log_u = torch.zeros_like(log_source)
        log_v = torch.zeros_like(log_target)

        for _ in range(self.sinkhorn_iterations):
            log_u = log_source - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0), dim=1
            )
            log_v = log_target - torch.logsumexp(
                log_kernel + log_u.unsqueeze(1), dim=0
            )

        return torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))

    def compute_transport(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return a row-conditional map and differentiable diagnostics."""
        if source_features.ndim != 2 or target_features.ndim != 2:
            raise ValueError("Transport features must have shape [batch, feature_dim].")
        if source_features.shape[1] != target_features.shape[1]:
            raise ValueError("Source and target transport feature dimensions must match.")

        source_count = source_features.shape[0]
        target_count = target_features.shape[0]
        if source_count == 0 or target_count == 0:
            raise ValueError("Transport batches cannot be empty.")

        geometric_cost = self.cost_net(source_features, target_features)
        transnet_logits = self.transmission_net(geometric_cost)
        source_marginal = geometric_cost.new_full(
            (source_count,), 1.0 / source_count
        )
        target_marginal = geometric_cost.new_full(
            (target_count,), 1.0 / target_count
        )

        if self.transport_mode == "legacy_row_softmax":
            # Exact historical behavior: TransNet logits are treated as the
            # transport logits and row-Softmax is the final conditional map.
            log_kernel = F.log_softmax(transnet_logits, dim=1)
            conditional_plan = log_kernel.exp()
            mass_plan = source_marginal.unsqueeze(1) * conditional_plan
            corrected_cost = geometric_cost
            cost_correction = torch.zeros_like(geometric_cost)
        else:
            # Revised neural cost: a bounded correction prevents TransNet from
            # arbitrarily replacing the geometric metric.
            cost_correction = self.correction_scale * torch.tanh(transnet_logits)
            corrected_cost = geometric_cost + cost_correction
            log_kernel = F.log_softmax(-corrected_cost / self.epsilon, dim=1)

            if self.transport_mode == "sinkhorn":
                mass_plan = self._sinkhorn_from_log_kernel(
                    log_kernel,
                    source_marginal,
                    target_marginal,
                )
                conditional_plan = mass_plan / source_marginal.unsqueeze(1)
            else:
                conditional_plan = log_kernel.exp()
                mass_plan = source_marginal.unsqueeze(1) * conditional_plan

        row_residual = (
            mass_plan.sum(dim=1) - source_marginal
        ).abs().max()
        column_residual = (
            mass_plan.sum(dim=0) - target_marginal
        ).abs().max()
        marginal_residual = torch.maximum(row_residual, column_residual)
        transport_cost = (mass_plan * corrected_cost).sum()
        entropy = -(
            mass_plan * torch.log(mass_plan.clamp_min(self.numerical_eps))
        ).sum()
        # The Sinkhorn branch solves the entropy-regularized inner problem
        #   min_P <P, C_tilde> - epsilon H(P)
        #   s.t. P 1 = a and P^T 1 = b.
        regularized_ot_objective = transport_cost - self.epsilon * entropy
        correction_regularization = cost_correction.square().mean()

        details = {
            "mass_plan": mass_plan,
            "conditional_plan": conditional_plan,
            "softmax_kernel": log_kernel.exp(),
            "geometric_cost": geometric_cost,
            "corrected_cost": corrected_cost,
            "cost_correction": cost_correction,
            "transport_cost": transport_cost,
            "entropy": entropy,
            "regularized_ot_objective": regularized_ot_objective,
            "row_marginal_residual": row_residual,
            "column_marginal_residual": column_residual,
            "marginal_residual": marginal_residual,
            "correction_regularization": correction_regularization,
        }
        self.last_transport_info = {
            "marginal_residual": float(marginal_residual.detach()),
            "transport_cost": float(transport_cost.detach()),
            "entropy": float(entropy.detach()),
            "regularized_ot_objective": float(
                regularized_ot_objective.detach()
            ),
            "row_marginal_residual": float(row_residual.detach()),
            "column_marginal_residual": float(column_residual.detach()),
            "correction_regularization": float(correction_regularization.detach()),
        }
        return conditional_plan, details

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        return_details: bool = False,
    ):
        conditional_plan, details = self.compute_transport(
            source_features, target_features
        )

        # Barycentric target representation. conditional_plan has row sum 1
        # for every mode, including the Sinkhorn plan after division by a_i.
        transported_target = conditional_plan.matmul(target_features)
        source_count = source_features.shape[0]
        tau = torch.distributions.Beta(2.0, 2.0).sample((source_count, 1)).to(
            device=source_features.device,
            dtype=source_features.dtype,
        )
        linear_interpolation = source_features + tau * (
            transported_target - source_features
        )

        # Neural residual correction retained from the original implementation.
        residual_input = torch.cat(
            [source_features, transported_target, tau], dim=1
        )
        residual = self.mlp(residual_input)
        intermediate_features = linear_interpolation + tau * residual

        details["transported_target"] = transported_target
        details["tau"] = tau
        if return_details:
            return intermediate_features, details
        return intermediate_features
