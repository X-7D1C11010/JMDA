import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class VisualFeatureExtractor(nn.Module):
    """ResNet 提取可见光特征"""

    def __init__(self, output_dim=512):
        super(VisualFeatureExtractor, self).__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x


class IRFeatureExtractor(nn.Module):
    """U-Net Encoder 提取红外特征"""

    def __init__(self, input_channels=3, output_dim=512):
        super(IRFeatureExtractor, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.enc1 = conv_block(input_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.proj(x)
        return x


class Classifier(nn.Module):
    """
    改进的分类器 - 增强表达能力
    
    改进点：
    1. 增加隐藏层维度（512 -> 256 -> 128）
    2. 添加LayerNorm稳定训练
    3. 使用更合理的Dropout率
    4. 添加残差连接（可选）
    """
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # 添加LayerNorm稳定训练
            nn.ReLU(),
            nn.Dropout(0.3),  # 降低Dropout率，避免过度正则化
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
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
        return self.fc(x)