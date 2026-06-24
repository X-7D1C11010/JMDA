"""
张量对齐模块 - 稳定简化版本

主要改进：
1. 移除显式约束损失（避免梯度爆炸）
2. 只通过投影操作满足约束
3. 简化协方差计算
4. 增强数值稳定性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TensorBasedAlignmentStable(nn.Module):
    """
    稳定版张量对齐模块

    核心改进：
    - 不在损失函数中包含约束项
    - 通过正交投影自动满足约束
    - 数值更稳定
    """

    def __init__(self,
                 input_dims: List[int],
                 output_dims: List[int],
                 num_modalities: int = 3):
        """
        Args:
            input_dims: 每种模态的输入维度
            output_dims: 投影后每个模态的输出维度
            num_modalities: 模态个数
        """
        super().__init__()

        self.num_modalities = num_modalities
        self.input_dims = input_dims
        self.output_dims = output_dims

        # 源域和目标域的变换矩阵
        self.U_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(input_dims[n], output_dims[n]))
            for n in range(num_modalities)
        ])

        self.V_matrices = nn.ParameterList([
            nn.Parameter(torch.randn(input_dims[n], output_dims[n]))
            for n in range(num_modalities)
        ])

        self._init_parameters()

        # 收敛检查
        self.prev_loss = None
        self.converged = False

    def _init_parameters(self):
        """正交初始化"""
        for U, V in zip(self.U_matrices, self.V_matrices):
            nn.init.orthogonal_(U)
            nn.init.orthogonal_(V)
        logger.info("参数初始化完成：正交初始化")

    def mode_n_product(self, tensor: torch.Tensor, matrix: torch.Tensor, mode: int) -> torch.Tensor:
        """mode-n 乘积"""
        tensor_mode = mode + 1
        dims = list(range(tensor.dim()))
        dims[tensor_mode], dims[-1] = dims[-1], dims[tensor_mode]
        tensor_permuted = tensor.permute(dims).contiguous()

        original_shape = tensor_permuted.shape
        mode_size = original_shape[-1]
        tensor_2d = tensor_permuted.view(-1, mode_size)

        if matrix.shape[0] != mode_size:
            raise ValueError(f"维度不匹配: {matrix.shape[0]} vs {mode_size}")

        result_2d = torch.matmul(tensor_2d, matrix)
        new_shape = original_shape[:-1] + (matrix.shape[1],)
        result_reshaped = result_2d.view(new_shape)

        inv_dims = [0] * len(dims)
        for i, d in enumerate(dims):
            inv_dims[d] = i
        result = result_reshaped.permute(inv_dims).contiguous()

        return result

    def create_multimodal_tensor(self, modalities: List[torch.Tensor]) -> torch.Tensor:
        """创建多模态张量（外积）"""
        batch_size = modalities[0].size(0)
        result = modalities[0]

        for i in range(1, len(modalities)):
            mod = modalities[i]
            result = result.unsqueeze(-1)
            mod = mod.unsqueeze(1)
            for _ in range(i - 1):
                mod = mod.unsqueeze(1)
            result = result * mod

        return result

    def tensor_contraction(self, tensor: torch.Tensor, exclude_mode: int) -> torch.Tensor:
        """张量收缩"""
        dims_to_contract = []
        for i in range(1, tensor.dim()):
            if i != exclude_mode + 1:
                dims_to_contract.append(i)

        if not dims_to_contract:
            return tensor

        result = tensor
        for dim in sorted(dims_to_contract, reverse=True):
            result = torch.sum(result, dim=dim, keepdim=False)

        return result

    def compute_correlation_score(self, X_features: torch.Tensor, Y_features: torch.Tensor) -> torch.Tensor:
        """
        计算相关性得分（简化版）

        使用标准化的点积作为相关性度量，避免复杂的SVD

        Args:
            X_features: [batch_size, dim]
            Y_features: [batch_size, dim]

        Returns:
            相关性得分（标量）
        """
        # L2 归一化
        X_norm = F.normalize(X_features, p=2, dim=1)
        Y_norm = F.normalize(Y_features, p=2, dim=1)

        # 计算余弦相似度矩阵
        similarity = torch.mm(X_norm, Y_norm.t())  # [batch, batch]

        # 平均相关性（对角线元素的平均）
        correlation = torch.diagonal(similarity).mean()

        return correlation

    def forward(self,
                source_modalities: List[torch.Tensor],
                target_modalities: List[torch.Tensor]) -> Tuple:
        """
        前向传播（简化版）

        只计算对齐目标，不包含约束损失

        Returns:
            projected_source: 投影后的源域特征
            projected_target: 投影后的目标域特征
            alignment_loss: 对齐损失
        """
        batch_size = source_modalities[0].shape[0]

        # 创建多模态张量
        X_tensor = self.create_multimodal_tensor(source_modalities)
        Y_tensor = self.create_multimodal_tensor(target_modalities)

        total_correlation = 0.0
        projected_source = []
        projected_target = []

        # 对每个模态计算相关性
        for n in range(self.num_modalities):
            # 应用其他模态的变换
            X_projected = X_tensor
            Y_projected = Y_tensor

            for i in range(self.num_modalities):
                if i != n:
                    X_projected = self.mode_n_product(X_projected, self.U_matrices[i], i)
                    Y_projected = self.mode_n_product(Y_projected, self.V_matrices[i], i)

            # 张量收缩
            X_contracted = self.tensor_contraction(X_projected, n)
            Y_contracted = self.tensor_contraction(Y_projected, n)

            # 计算相关性得分（替代复杂的CCA）
            correlation = self.compute_correlation_score(X_contracted, Y_contracted)
            total_correlation += correlation

            # 投影当前模态
            proj_source = torch.mm(source_modalities[n], self.U_matrices[n])
            proj_target = torch.mm(target_modalities[n], self.V_matrices[n])

            projected_source.append(proj_source)
            projected_target.append(proj_target)

        # 对齐损失 = 负相关性（我们要最大化相关性）
        alignment_loss = -total_correlation / self.num_modalities

        return projected_source, projected_target, alignment_loss

    def apply_orthogonal_projection(self):
        """
        应用正交投影

        通过 QR 分解强制参数正交化
        这自动满足约束条件（当协方差接近单位矩阵时）
        """
        with torch.no_grad():
            for n in range(self.num_modalities):
                # QR 分解
                Q_u, _ = torch.linalg.qr(self.U_matrices[n].data)
                self.U_matrices[n].data = Q_u

                Q_v, _ = torch.linalg.qr(self.V_matrices[n].data)
                self.V_matrices[n].data = Q_v

    def check_convergence(self, current_loss: float, threshold: float = 1e-6) -> bool:
        """检查收敛"""
        if self.prev_loss is not None:
            loss_change = abs(current_loss - self.prev_loss)
            if loss_change < threshold:
                self.converged = True
                logger.info(f"收敛! 损失变化: {loss_change:.8f}")

        self.prev_loss = current_loss
        return self.converged