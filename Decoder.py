import torch
import torch.nn as nn


class VisDecoder(nn.Module):
    """Decodes 512-dim visible light feature → [B, 3, 224, 224] (Tanh output in [-1, 1])"""

    def __init__(self, input_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, 512 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 7 → 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14 → 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28 → 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56 → 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 112 → 224
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        return self.deconv(x)


class IRDecoder(nn.Module):
    """Decodes 512-dim IR feature → [B, 3, 224, 224] (Tanh output in [-1, 1])"""

    def __init__(self, input_dim=512):
        super().__init__()
        self.fc = nn.Linear(input_dim, 512 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 7 → 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14 → 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28 → 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56 → 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 112 → 224
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        return self.deconv(x)


class MidFeatureDecoder(nn.Module):
    """Decodes 256-dim mid feature → [B, 3, 224, 224] (Tanh output in [-1, 1])"""

    def __init__(self, input_dim=256):
        super().__init__()
        self.fc = nn.Linear(input_dim, 512 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 7 → 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 14 → 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 28 → 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 56 → 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 112 → 224
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        return self.deconv(x)


# ── Helper tensors for denormalization ─────────────────────────────────────────
_VIS_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VIS_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_recon_target_vis(img_tensor):
    """ImageNet-normalized [B,3,H,W] → [-1,1] reconstruction target.

    Denormalizes to [0, 1] using ImageNet statistics, then scales to [-1, 1].
    """
    mean = _VIS_MEAN.to(img_tensor.device)
    std  = _VIS_STD.to(img_tensor.device)
    # Denormalize: x_01 = x_norm * std + mean
    x_01 = img_tensor * std + mean
    x_01 = x_01.clamp(0.0, 1.0)
    # Scale to [-1, 1]
    return x_01 * 2.0 - 1.0


def get_recon_target_ir(img_tensor):
    """IR-normalized (mean=0.5, std=0.5) [B,3,H,W] → [-1,1] reconstruction target.

    Since mean=0.5 and std=0.5, the normalized range is already [-1, 1]:
        x_norm = (x - 0.5) / 0.5  →  x_norm in [-1, 1] when x in [0, 1]
    So this function is an identity pass-through, clamped for safety.
    """
    return img_tensor.clamp(-1.0, 1.0)
