"""
Domain Discriminator Module
三分类域判别器，用于区分源域、目标域和生成的中间域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    """
    梯度反转层（Gradient Reversal Layer, GRL）

    前向传播：输出 = 输入（不改变）
    反向传播：梯度 = -alpha * 梯度（翻转并缩放）

    用途：在对抗训练中，让生成器的梯度与判别器相反
    使得生成器试图混淆判别器，而判别器试图正确分类
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """
        Args:
            x: 输入张量
            alpha: 梯度反转强度系数
        """
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: 从后续层传回的梯度
        Returns:
            翻转后的梯度, None (alpha不需要梯度)
        """
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    """
    三分类域判别器

    功能：判断输入特征来自哪个域
    - 类别0: 源域 (Source Domain)
    - 类别1: 目标域 (Target Domain)
    - 类别2: 中间域 (Mixed/Intermediate Domain)

    网络结构：
    Input → [Linear → LayerNorm → ReLU → Dropout] × N → Linear → Output(3)

    设计要点：
    1. 多层结构：逐步提取判别性特征
    2. LayerNorm：稳定训练，防止梯度爆炸/消失
    3. Dropout：防止过拟合，提高泛化能力
    4. 输出3个logits：对应3个域的未归一化分数
    """

    def __init__(self, feature_dim, hidden_dims=[512, 256, 128], dropout=0.3):
        """
        Args:
            feature_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表，例如 [512, 256, 128]
            dropout: Dropout比例，推荐0.3-0.5
        """
        super(DomainDiscriminator, self).__init__()

        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims

        # 构建判别网络
        layers = []
        input_dim = feature_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 归一化，稳定训练
                nn.ReLU(inplace=True),  # 非线性激活
                nn.Dropout(dropout)  # 防止过拟合
            ])
            input_dim = hidden_dim

        # 最终分类层：输出3个类别的logits
        layers.append(nn.Linear(input_dim, 3))

        self.discriminator = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        权重初始化：使用Xavier初始化
        有助于训练初期的梯度稳定性
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, alpha=1.0, use_grl=False):
        """
        前向传播

        Args:
            x: [batch_size, feature_dim] - 输入特征
            alpha: float - 梯度反转强度（默认1.0）
                   通常在训练过程中逐渐增大：alpha = 2/(1+exp(-10*p)) - 1
                   其中 p 从 0 增长到 1
            use_grl: bool - 是否使用梯度反转层
                     训练生成器时设为True，训练判别器时设为False

        Returns:
            logits: [batch_size, 3] - 三个域的logits (未归一化分数)
                    logits[:, 0] → 源域分数
                    logits[:, 1] → 目标域分数
                    logits[:, 2] → 中间域分数
        """
        # 如果启用梯度反转层
        if use_grl:
            x = GradientReversal.apply(x, alpha)

        # 通过判别网络
        logits = self.discriminator(x)

        return logits

    def predict(self, x):
        """
        预测域标签（推理模式）

        Args:
            x: [batch_size, feature_dim] - 输入特征

        Returns:
            predictions: [batch_size] - 预测的域标签
                         0 = 源域, 1 = 目标域, 2 = 中间域
            probabilities: [batch_size, 3] - 每个域的概率分布
        """
        with torch.no_grad():
            logits = self.forward(x, use_grl=False)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

        return predictions, probabilities

    def get_confusion_entropy(self, x):
        """
        计算混淆熵：衡量判别器的不确定性

        熵越高，说明判别器越难区分该样本的域
        生成器的目标是让中间域的熵接近最大值 log(3)

        Args:
            x: [batch_size, feature_dim] - 输入特征

        Returns:
            entropy: [batch_size] - 每个样本的预测熵
        """
        logits = self.forward(x, use_grl=False)
        probs = F.softmax(logits, dim=-1)

        # 计算熵: H = -Σ p(x) * log(p(x))
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy


def compute_discriminator_loss(src_logits, tgt_logits, mid_logits):
    """
    计算判别器损失

    目标：让判别器能够正确分类三个域
    使用交叉熵损失

    Args:
        src_logits: [batch_size, 3] - 源域样本的判别logits
        tgt_logits: [batch_size, 3] - 目标域样本的判别logits
        mid_logits: [batch_size, 3] - 中间域样本的判别logits

    Returns:
        total_loss: 总损失（三个域的平均损失）
        loss_dict: 包含各域损失的字典，便于监控
    """
    batch_size = src_logits.size(0)
    device = src_logits.device

    # 创建标签
    src_labels = torch.zeros(batch_size, dtype=torch.long, device=device)  # 0: 源域
    tgt_labels = torch.ones(batch_size, dtype=torch.long, device=device)  # 1: 目标域
    mid_labels = torch.full((batch_size,), 2, dtype=torch.long, device=device)  # 2: 中间域

    # 计算每个域的交叉熵损失
    loss_src = F.cross_entropy(src_logits, src_labels)
    loss_tgt = F.cross_entropy(tgt_logits, tgt_labels)
    loss_mid = F.cross_entropy(mid_logits, mid_labels)

    # 总损失：三个域的平均损失
    total_loss = (loss_src + loss_tgt + loss_mid) / 3

    # 返回详细损失信息
    loss_dict = {
        'total': total_loss.item(),
        'source': loss_src.item(),
        'target': loss_tgt.item(),
        'mixed': loss_mid.item()
    }

    return total_loss, loss_dict


def compute_generator_loss(mid_logits, loss_type='kl_uniform'):
    """
    计算生成器损失

    目标：让中间域特征混淆判别器
    有多种策略可选

    Args:
        mid_logits: [batch_size, 3] - 中间域样本的判别logits
        loss_type: str - 损失类型
            'kl_uniform': KL散度，希望预测分布接近均匀分布
            'entropy': 最大化熵，让判别器更不确定
            'adversarial': 对抗损失，让中间域被误判为源域或目标域

    Returns:
        loss: 生成器损失
    """
    if loss_type == 'kl_uniform':
        # 策略1: 让预测分布接近均匀分布 [1/3, 1/3, 1/3]
        batch_size = mid_logits.size(0)
        uniform_target = torch.ones_like(mid_logits) / 3

        loss = F.kl_div(
            F.log_softmax(mid_logits, dim=-1),
            uniform_target,
            reduction='batchmean'
        )

    elif loss_type == 'entropy':
        # 策略2: 最大化预测熵（等价于最小化负熵）
        probs = F.softmax(mid_logits, dim=-1)
        log_probs = F.log_softmax(mid_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # 最大化熵 = 最小化负熵
        loss = -entropy.mean()

    elif loss_type == 'adversarial':
        # 策略3: 让中间域被判别为源域（或目标域）
        batch_size = mid_logits.size(0)
        fake_labels = torch.zeros(batch_size, dtype=torch.long, device=mid_logits.device)

        loss = F.cross_entropy(mid_logits, fake_labels)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss