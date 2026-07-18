import torch
import torch.nn as nn


class AISFeatureExtractor(nn.Module):
    """
    AIS信号特征提取器

    使用多层感知机(MLP)提取AIS信号特征
    适用于从.mat文件加载的AIS数据
    """

    def __init__(self, input_dim=None, output_dim=512):
        """
        Args:
            input_dim: AIS信号的输入维度 (需要根据实际数据确定)
            output_dim: 输出特征维度 (默认512，与其他模态保持一致)
        """
        super(AISFeatureExtractor, self).__init__()

        # 如果没有指定input_dim，使用默认值
        if input_dim is None:
            input_dim = 128  # 默认值，需要根据实际AIS数据调整

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 多层MLP特征提取器
        self.features = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # 第二层
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # 第三层
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] - AIS信号输入

        Returns:
            features: [batch_size, output_dim] - 提取的特征
        """
        x = self.features(x)
        return x


class AISFeatureExtractorDeep(nn.Module):
    """
    深层AIS信号特征提取器

    使用更深的网络结构，提取更复杂的特征
    """

    def __init__(self, input_dim=None, output_dim=512):
        super(AISFeatureExtractorDeep, self).__init__()

        if input_dim is None:
            input_dim = 128

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 更深的网络结构
        self.features = nn.Sequential(
            # 第一层
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            # 第二层
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            # 第三层
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            # 第四层
            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.features(x)


class AISFeatureExtractorCNN1D(nn.Module):
    """
    1D CNN AIS信号特征提取器

    如果AIS数据具有时序或序列特性，使用1D CNN可能更合适
    """

    def __init__(self, input_channels=1, sequence_length=128, output_dim=512):
        """
        Args:
            input_channels: 输入通道数 (默认1，单变量时序)
            sequence_length: 序列长度
            output_dim: 输出特征维度
        """
        super(AISFeatureExtractorCNN1D, self).__init__()

        self.input_channels = input_channels
        self.sequence_length = sequence_length

        # 1D卷积层
        self.conv_layers = nn.Sequential(
            # Conv1
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # Conv2
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # Conv3
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

            # Conv4
            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 投影层
        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_channels, sequence_length] - AIS信号输入
               或 [batch_size, sequence_length] - 会自动增加channel维度

        Returns:
            features: [batch_size, output_dim] - 提取的特征
        """
        # 如果输入是2D，增加channel维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, length] -> [batch, 1, length]

        x = self.conv_layers(x)
        x = self.global_avg_pool(x)  # [batch, 512, 1]
        x = x.squeeze(-1)  # [batch, 512]
        x = self.fc(x)  # [batch, output_dim]

        return x


class AISResidualConvBlock1D(nn.Module):
    """Lightweight residual ConvNeXt-style block for long I/Q sequences."""

    def __init__(self, channels, dropout=0.1):
        super(AISResidualConvBlock1D, self).__init__()
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, channels * 3)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.project = nn.Linear(channels * 3, channels)

    def forward(self, x):
        residual = x
        x = self.depthwise(x).transpose(1, 2)
        x = self.norm(x)
        x = self.project(self.dropout(self.activation(self.expand(x))))
        return residual + x.transpose(1, 2)


class AISIQFeatureExtractorCNN1D(nn.Module):
    """Complex-signal extractor that preserves I/Q geometry.

    The AIS file stores one record as concatenated I then Q vectors. The old
    CNN interpreted that vector as one channel, creating an artificial boundary
    between I and Q and losing their sample-wise correspondence. In addition to
    reconstructing [batch, 2, time], this extractor derives amplitude and the
    real/imaginary parts of z[t] * conj(z[t-1]). These three channels are stable
    under a global phase rotation and improve generalization without using any
    validation labels or adding a new adaptation module.
    """

    def __init__(self, input_dim, output_dim=512):
        super(AISIQFeatureExtractorCNN1D, self).__init__()
        if input_dim % 2 != 0:
            raise ValueError(
                f"AIS I/Q input_dim must be even, got {input_dim}"
            )
        self.input_dim = input_dim
        self.sequence_length = input_dim // 2
        self.stem = nn.Sequential(
            nn.Conv1d(5, 64, kernel_size=31, stride=16, padding=15),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.stage1 = nn.Sequential(
            AISResidualConvBlock1D(64, dropout=0.10),
            AISResidualConvBlock1D(64, dropout=0.10),
        )
        self.downsample = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.stage2 = nn.Sequential(
            AISResidualConvBlock1D(128, dropout=0.15),
            AISResidualConvBlock1D(128, dropout=0.15),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(128 * 3),
            nn.Linear(128 * 3, output_dim),
            nn.GELU(),
            nn.Dropout(0.20),
        )

    def forward(self, x):
        if x.dim() != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"Expected AIS input [batch, {self.input_dim}], got {tuple(x.shape)}"
            )
        i_part, q_part = torch.chunk(x, 2, dim=1)

        amplitude = torch.sqrt(i_part.square() + q_part.square() + 1e-8)
        product_real = i_part[:, 1:] * i_part[:, :-1] + q_part[:, 1:] * q_part[:, :-1]
        product_imag = q_part[:, 1:] * i_part[:, :-1] - i_part[:, 1:] * q_part[:, :-1]
        # Keep all derived channels aligned with the original sequence length.
        zeros = torch.zeros_like(i_part[:, :1])
        product_real = torch.cat([zeros, product_real], dim=1)
        product_imag = torch.cat([zeros, product_imag], dim=1)

        x = torch.stack(
            [i_part, q_part, amplitude, product_real, product_imag],
            dim=1,
        )
        x = self.stage1(self.stem(x))
        x = self.stage2(self.downsample(x))
        # Mean alone discards burst strength and local peaks. The three fixed
        # statistics keep output dimensions independent of input record length.
        pooled = torch.cat([
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            x.amax(dim=-1),
        ], dim=1)
        return self.projection(pooled)


def create_ais_feature_extractor(ais_data_shape=None, output_dim=512, architecture='mlp'):
    """
    工厂函数：根据AIS数据形状和架构类型创建合适的特征提取器

    Args:
        ais_data_shape: AIS数据的形状 (例如: (128,) 或 (1, 128))
        output_dim: 输出特征维度
        architecture: 架构类型 ('mlp', 'deep_mlp', 'cnn1d')

    Returns:
        feature_extractor: AIS特征提取器实例
    """
    if ais_data_shape is None:
        print("警告: 未指定AIS数据形状，使用默认配置")
        input_dim = 128
    else:
        if isinstance(ais_data_shape, tuple):
            if len(ais_data_shape) == 1:
                input_dim = ais_data_shape[0]
            else:
                input_dim = ais_data_shape[-1]
        else:
            input_dim = ais_data_shape

    if architecture == 'mlp':
        return AISFeatureExtractor(input_dim=input_dim, output_dim=output_dim)
    elif architecture == 'deep_mlp':
        return AISFeatureExtractorDeep(input_dim=input_dim, output_dim=output_dim)
    elif architecture == 'cnn1d':
        return AISFeatureExtractorCNN1D(
            input_channels=1,
            sequence_length=input_dim,
            output_dim=output_dim
        )
    elif architecture == 'iq_cnn1d':
        return AISIQFeatureExtractorCNN1D(
            input_dim=input_dim,
            output_dim=output_dim,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
