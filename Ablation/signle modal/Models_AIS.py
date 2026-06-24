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
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
