"""Three-modality feature encoders for the VDS2Raw JMDA-Net entry point.

The image encoders deliberately match the historical VIS/IR experiment: a
ResNet18 VIS/RGB branch and a four-stage U-Net-style IR branch.  VDS2Raw B8 is
stored as one channel, so the NIR encoder repeats it to the three channels
expected by the historical IR network without changing the prepared dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


FEATURE_DIM = 512
AIS_INPUT_DIM = 26


class RGBFeatureExtractor(nn.Module):
    """Historical ImageNet-pretrained ResNet18 VIS encoder at 224x224."""

    def __init__(
        self,
        output_dim: int = FEATURE_DIM,
        pretrained: bool = True,
        pretrained_weights_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        resnet = models.resnet18(weights=None)
        if pretrained:
            if pretrained_weights_path:
                checkpoint_path = Path(pretrained_weights_path).expanduser().resolve()
            else:
                filename = Path(urlparse(ResNet18_Weights.DEFAULT.url).path).name
                checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "ImageNet ResNet18 weights are not available offline. "
                    f"Expected {checkpoint_path}. Copy resnet18-f37072fd.pth and "
                    "pass --pretrained_weights, or explicitly use --no-pretrained."
                )
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location="cpu")
            resnet.load_state_dict(state)
        # Keep the same module layout as Models.VisualFeatureExtractor.  This
        # also makes the actual freeze policy explicit: the complete ResNet
        # feature stack is frozen and only ``proj`` is optimized.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(512, output_dim)

    def freeze_backbone(self) -> None:
        """Reproduce the old experiment's effective VIS freeze policy."""

        for parameter in self.features.parameters():
            parameter.requires_grad = False
        for parameter in self.proj.parameters():
            parameter.requires_grad = True

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        features = self.features(rgb).flatten(1)
        return self.proj(features)


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Historical IR encoder block (Conv-BN-ReLU twice, then pooling)."""

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class NIRFeatureExtractor(nn.Module):
    """Historical three-channel U-Net-style encoder for Sentinel-2 B8.

    B8 is a scalar spectral band, not thermal RGB.  Repeating the same values
    over three channels is used solely to reproduce the old IR encoder input
    contract; it does not invent additional spectral information.
    """

    def __init__(self, output_dim: int = FEATURE_DIM):
        super().__init__()
        self.enc1 = _conv_block(3, 64)
        self.enc2 = _conv_block(64, 128)
        self.enc3 = _conv_block(128, 256)
        self.enc4 = _conv_block(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, output_dim)

    def forward(self, nir: torch.Tensor) -> torch.Tensor:
        if nir.ndim != 4 or nir.shape[1] not in (1, 3):
            raise ValueError(
                "NIR input must have shape [batch, 1, height, width] or "
                f"[batch, 3, height, width], got {tuple(nir.shape)}"
            )
        if nir.shape[1] == 1:
            nir = nir.repeat(1, 3, 1, 1)
        features = self.enc1(nir)
        features = self.enc2(features)
        features = self.enc3(features)
        features = self.enc4(features)
        features = self.avgpool(features).flatten(1)
        return self.proj(features)


class AISFeatureExtractor(nn.Module):
    """Encode 13 navigation features together with their 13 validity masks."""

    def __init__(self, input_dim: int = AIS_INPUT_DIM, output_dim: int = FEATURE_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, output_dim),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, ais_input: torch.Tensor) -> torch.Tensor:
        if ais_input.ndim != 2 or ais_input.shape[1] != self.input_dim:
            raise ValueError(
                f"AIS input must have shape [batch, {self.input_dim}], "
                f"got {tuple(ais_input.shape)}"
            )
        return self.encoder(ais_input)


class Classifier(nn.Module):
    """Classification head kept compatible with SVD basis transport."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


class BinaryDomainDiscriminator(nn.Module):
    """Source/target discriminator for the no-OT ablation."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.discriminator(features).squeeze(1)


def trainable_parameters(*modules: nn.Module) -> Iterable[nn.Parameter]:
    for module in modules:
        yield from (parameter for parameter in module.parameters() if parameter.requires_grad)


__all__ = [
    "FEATURE_DIM",
    "AIS_INPUT_DIM",
    "RGBFeatureExtractor",
    "NIRFeatureExtractor",
    "AISFeatureExtractor",
    "Classifier",
    "BinaryDomainDiscriminator",
    "trainable_parameters",
]
