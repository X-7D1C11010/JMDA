import torch
import torch.nn as nn
import torch.nn.functional as F


class CostNet(nn.Module):
    """
    计算代价矩阵 C
    """

    def __init__(self, feature_dim):
        super(CostNet, self).__init__()
        # 特征投影层
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim)
        )

    def forward(self, src_feat, tgt_feat):
        # src_feat: [B, D], tgt_feat: [B, D]
        phi_s = self.proj(src_feat)
        phi_t = self.proj(tgt_feat)

        # 计算成对欧氏距离平方: ||phi(fs) - phi(ft)||^2
        # 利用广播机制计算 [B, B] 矩阵
        dist = (phi_s.unsqueeze(1) - phi_t.unsqueeze(0)).pow(2).sum(dim=2)
        return dist  # [B, B]


class TransmissionNetwork(nn.Module):
    """
    【核心修改】简单的神经网络，用于预测传输矩阵 T
    """

    def __init__(self):
        super(TransmissionNetwork, self).__init__()
        # 使用卷积网络处理代价矩阵
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            # 不用Sigmoid，后面用Softmax
        )

    def forward(self, cost_matrix):
        """
        修正点：将 [B, B] 的矩阵视为 [1, 1, B, B] 的图像输入
        Batch Size = 1 (因为我们是在处理整个batch的全局关系矩阵)
        Channels = 1
        Height = B, Width = B
        """
        batch_size = cost_matrix.size(0)

        # [B, B] -> [1, 1, B, B]
        x = cost_matrix.view(1, 1, batch_size, batch_size)

        # Output: [1, 1, B, B]
        logits = self.net(x)

        # 还原回 [B, B]
        logits = logits.view(batch_size, batch_size)

        # 使用 Softmax 保证每一行和为1（源域样本 i 转移到目标域各样本的概率）
        transport_plan = F.softmax(logits, dim=1)
        return transport_plan


class NeuralOptimalTransportGenerator(nn.Module):
    """
    神经最优传输生成器
    """

    def __init__(self, feature_dim, hidden_dim=256):
        super(NeuralOptimalTransportGenerator, self).__init__()

        self.cost_net = CostNet(feature_dim)
        self.transmission_net = TransmissionNetwork()

        # 最后的混合 MLP，引入非线性
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.Tanh()  # 残差修正
        )

    def forward(self, src_feat, tgt_feat):
        batch_size = src_feat.size(0)

        # 1. 计算代价矩阵 (Cost)
        C = self.cost_net(src_feat, tgt_feat)

        # 2. 神经网络预测传输矩阵 (Transmission)
        P = self.transmission_net(C)

        # 3. 基于传输计划重构目标域特征 (Barycentric Mapping)
        # P: [B, B], tgt_feat: [B, D] -> [B, D]
        transported_tgt = torch.matmul(P, tgt_feat)

        # 4. 位移插值 (Displacement Interpolation)
        delta = transported_tgt - src_feat

        # 采样插值系数 tau (Beta分布)
        # 注意：生成 tensor 需要与 src_feat 在同一设备上
        tau = torch.distributions.Beta(2.0, 2.0).sample((batch_size, 1)).to(src_feat.device)

        # 线性部分
        f_m_linear = src_feat + tau * delta

        # 5. 非线性残差修正
        combined = torch.cat([src_feat, transported_tgt, tau], dim=1)
        residual = self.mlp(combined)

        # 最终生成中间域特征
        mid_feat = f_m_linear + tau * residual

        return mid_feat