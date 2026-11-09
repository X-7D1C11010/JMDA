"""Run leakage-safe VDS2Raw single-modality baselines.

The experiment grid is three architecture families (ResNet, ViT, ConvNeXt)
by three modalities (RGB, NIR, AIS) by five independent seeds.  Source and
target training partitions are both supervised.  Validation selects one
checkpoint, and target_test is touched only once after that checkpoint has
been restored.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Required by CUDA for reproducible matrix multiplications when deterministic
# algorithms are enabled.  It must be set before CUDA is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from torchvision import models


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from VDS2Raw_JMDA.vds2raw_dataset import (  # noqa: E402
    CLASS_NAMES,
    TRAIN_PARTITIONS,
    VDS2RawDataset,
    VDS2RawManifest,
    compute_train_image_stats,
)


MODEL_NAMES = ("resnet", "vit", "convnext")
MODALITIES = ("rgb", "nir", "ais")
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
NATIVE_IMAGE_SIZE = (96, 96)
DEFAULT_IMAGE_SIZE = 96
DEFAULT_TRAIN_RESIZE_SIZE = 112

# Heading is represented by a sin/cos pair in the prepared feature schema.
# Length and width are intentionally excluded because they strongly encode the
# vessel class in this three-class dataset.
AIS_VALUE_INDICES = (0, 1)
AIS_VALUE_NAMES = ("heading_sin", "heading_cos")
AIS_SEMANTIC_FIELDS = ("heading",)
AIS_INPUT_DIM = 3  # heading sin/cos plus one validity indicator


def parse_csv_values(value: str, allowed: Sequence[str], label: str) -> List[str]:
    values = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(values) - set(allowed))
    if not values or unknown:
        raise argparse.ArgumentTypeError(
            f"{label} must be a comma-separated subset of {list(allowed)}; unknown={unknown}"
        )
    return list(dict.fromkeys(values))


def parse_seeds(value: str) -> List[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be non-empty and unique")
    return seeds


def default_dataset_root() -> Path:
    configured = os.environ.get("VDS2RAW_ROOT")
    if configured:
        return Path(configured)
    windows_path = Path(r"D:\Downloads\VDS2Raw\threeclass_ready")
    if windows_path.is_dir():
        return windows_path
    return Path("/home/lixiang/lx/VDS2Raw/threeclass_ready")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the 3-model x 3-modality VDS2Raw single-modal baseline grid."
    )
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_DIR / "single_modal_runs"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Stable run folder name. Reusing it resumes completed configurations.",
    )
    parser.add_argument(
        "--models",
        type=lambda value: parse_csv_values(value, MODEL_NAMES, "models"),
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--modalities",
        type=lambda value: parse_csv_values(value, MODALITIES, "modalities"),
        default=list(MODALITIES),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=100, help="Image-model epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Image-model patience.")
    parser.add_argument("--ais-epochs", type=int, default=50)
    parser.add_argument("--ais-patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr-image", type=float, default=3e-4)
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help=(
            "Optional image-backbone learning rate. When set, --lr-image is "
            "used for the classification head."
        ),
    )
    parser.add_argument("--lr-ais", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument(
        "--backbone-profile",
        choices=("baseline", "transfer-small"),
        default="baseline",
        help=(
            "baseline uses ResNet-50/ViT-B-16/ConvNeXt-Base; transfer-small "
            "uses ResNet-18/ViT-B-16/ConvNeXt-Small."
        ),
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Initialize image backbones with torchvision ImageNet weights.",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Train only the classification head for this many initial epochs.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=("acc", "macro_f1"),
        default="acc",
        help="Primary target_val metric used for checkpoint selection.",
    )
    parser.add_argument(
        "--domain-class-balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sample the six source/target x class cells with equal expected "
            "mass during training."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUDA automatic mixed precision for image models.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Autocast dtype used when --amp is enabled.",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use channels-last memory format for convolutional image models.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Square RGB/NIR model input size. ViT requires a multiple of 16.",
    )
    parser.add_argument(
        "--train-resize-size",
        type=int,
        default=DEFAULT_TRAIN_RESIZE_SIZE,
        help=(
            "Training resize before cropping to --image-size. Setting both sizes "
            "equal disables spatial random cropping."
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--image-augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a synchronized horizontal flip to training images.",
    )
    parser.add_argument(
        "--class-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate target_test exactly once after validation checkpoint selection.",
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch/one training batch and never access target_test.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke_test:
        args.epochs = 1
        args.patience = 1
        args.ais_epochs = 1
        args.ais_patience = 1
        args.warmup_epochs = 0
        args.evaluate_test = False
    positive_ints = {
        "epochs": args.epochs,
        "patience": args.patience,
        "ais_epochs": args.ais_epochs,
        "ais_patience": args.ais_patience,
        "batch_size": args.batch_size,
    }
    invalid = [name for name, value in positive_ints.items() if value < 1]
    if invalid:
        raise ValueError(f"Arguments must be positive: {invalid}")
    if args.num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup_epochs must be >= 0 and smaller than epochs")
    if args.image_size < 1:
        raise ValueError("image_size must be positive")
    if "vit" in args.models and args.image_size % 16 != 0:
        raise ValueError("ViT image_size must be divisible by its 16-pixel patch size")
    if args.train_resize_size < args.image_size:
        raise ValueError("train_resize_size must be at least image_size")
    if args.lr_image <= 0 or args.lr_ais <= 0:
        raise ValueError("learning rates must be positive")
    if args.lr_backbone is not None and args.lr_backbone <= 0:
        raise ValueError("lr_backbone must be positive when provided")
    if args.freeze_backbone_epochs < 0 or args.freeze_backbone_epochs >= args.epochs:
        raise ValueError("freeze_backbone_epochs must be >= 0 and smaller than epochs")
    if args.freeze_backbone_epochs > 0 and not args.pretrained:
        raise ValueError("freezing a randomly initialized backbone is unsupported")
    if args.weight_decay < 0 or not 0 <= args.label_smoothing < 1:
        raise ValueError("invalid weight_decay or label_smoothing")
    if args.grad_clip < 0:
        raise ValueError("grad_clip must be >= 0")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def utc_free_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in fieldnames} for row in rows)
    temporary.replace(path)


class SingleModalityDataset(Dataset):
    """Expose exactly one modality from an immutable protocol partition."""

    def __init__(
        self,
        manifest: VDS2RawManifest,
        partition: str,
        image_stats: Mapping[str, Sequence[float]],
        modality: str,
        augment: bool,
        image_size: int = DEFAULT_IMAGE_SIZE,
        train_resize_size: int = DEFAULT_TRAIN_RESIZE_SIZE,
    ):
        if modality not in MODALITIES:
            raise ValueError(f"Unknown modality: {modality}")
        if modality == "ais":
            augment = False
        self.modality = modality
        self.base = VDS2RawDataset(
            manifest,
            partition,
            image_stats,
            augment=augment,
            model_image_size=(image_size, image_size),
            train_resize_size=(train_resize_size, train_resize_size),
        )
        self.labels = list(self.base.labels)
        self.records = self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        sample = self.base[index]
        if self.modality == "rgb":
            inputs = sample["rgb"]
        elif self.modality == "nir":
            inputs = sample["nir"]
        else:
            values = sample["ais"][list(AIS_VALUE_INDICES)]
            masks = sample["ais_mask"][list(AIS_VALUE_INDICES)]
            heading_valid = torch.minimum(masks[0], masks[1]).unsqueeze(0)
            inputs = torch.cat([values, heading_valid], dim=0)
            if inputs.shape != (AIS_INPUT_DIM,):
                raise RuntimeError(f"Unexpected AIS input shape: {tuple(inputs.shape)}")
        return inputs, sample["label"], sample["sample_id"]


class ResidualMLPBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.block(inputs)


class AISResNet(nn.Module):
    """Residual MLP analogue for the three structured AIS fields."""

    def __init__(self, num_classes: int):
        super().__init__()
        width = 128
        self.stem = nn.Sequential(nn.Linear(AIS_INPUT_DIM, width), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualMLPBlock(width) for _ in range(4)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(inputs)))


def ais_field_tokens(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim != 2 or inputs.shape[1] != AIS_INPUT_DIM:
        raise ValueError(f"AIS inputs must have shape [batch, {AIS_INPUT_DIM}]")
    return inputs.unsqueeze(1)


class AISViT(nn.Module):
    """Transformer over heading, length, and width field tokens."""

    def __init__(self, num_classes: int):
        super().__init__()
        width = 128
        self.token_projection = nn.Linear(3, width)
        self.class_token = nn.Parameter(torch.zeros(1, 1, width))
        self.position = nn.Parameter(torch.zeros(1, 2, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 4,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_classes))
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.token_projection(ais_field_tokens(inputs))
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([class_token, tokens], dim=1) + self.position)
        return self.head(encoded[:, 0])


class ConvNeXt1DBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.15):
        super().__init__()
        self.depthwise = nn.Conv1d(
            width, width, kernel_size=3, padding=1, groups=width
        )
        self.norm = nn.LayerNorm(width)
        self.pointwise1 = nn.Linear(width, width * 4)
        self.pointwise2 = nn.Linear(width * 4, width)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.full((width,), 1e-6))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        outputs = self.depthwise(inputs).transpose(1, 2)
        outputs = self.norm(outputs)
        outputs = self.pointwise2(self.dropout(torch.nn.functional.gelu(self.pointwise1(outputs))))
        return residual + (outputs * self.scale).transpose(1, 2)


class AISConvNeXt(nn.Module):
    """One-dimensional ConvNeXt analogue over the three AIS fields."""

    def __init__(self, num_classes: int):
        super().__init__()
        width = 128
        self.stem = nn.Conv1d(3, width, kernel_size=1)
        self.blocks = nn.Sequential(*[ConvNeXt1DBlock(width) for _ in range(6)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = ais_field_tokens(inputs).transpose(1, 2)
        features = self.blocks(self.stem(tokens)).mean(dim=-1)
        return self.head(features)


def replace_input_conv(
    module: nn.Module, in_channels: int, preserve_pretrained: bool = False
) -> nn.Conv2d:
    if not isinstance(module, nn.Conv2d):
        raise TypeError(f"Expected Conv2d, got {type(module).__name__}")
    replacement = nn.Conv2d(
        in_channels,
        module.out_channels,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
        padding_mode=module.padding_mode,
    )
    with torch.no_grad():
        if preserve_pretrained:
            if in_channels != 1 or module.in_channels != 3:
                raise ValueError(
                    "Pretrained input adaptation currently supports RGB-to-one-channel only"
                )
            # This is equivalent to feeding the normalized NIR band into all
            # three RGB channels while avoiding a physical 3x tensor copy.
            replacement.weight.copy_(module.weight.sum(dim=1, keepdim=True))
            if replacement.bias is not None and module.bias is not None:
                replacement.bias.copy_(module.bias)
        else:
            nn.init.kaiming_normal_(
                replacement.weight, mode="fan_out", nonlinearity="relu"
            )
            if replacement.bias is not None:
                nn.init.zeros_(replacement.bias)
    return replacement


def create_image_model(
    model_name: str,
    modality: str,
    num_classes: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone_profile: str = "baseline",
    pretrained: bool = False,
) -> nn.Module:
    input_channels = 3 if modality == "rgb" else 1
    if model_name == "resnet":
        if backbone_profile == "transfer-small":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if input_channels != 3:
            model.conv1 = replace_input_conv(
                model.conv1, input_channels, preserve_pretrained=pretrained
            )
    elif model_name == "vit":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(
            weights=weights, image_size=image_size
        )
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        if input_channels != 3:
            model.conv_proj = replace_input_conv(
                model.conv_proj, input_channels, preserve_pretrained=pretrained
            )
    elif model_name == "convnext":
        if backbone_profile == "transfer-small":
            weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
            model = models.convnext_small(weights=weights)
        else:
            weights = models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None
            model = models.convnext_base(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        if input_channels != 3:
            model.features[0][0] = replace_input_conv(
                model.features[0][0],
                input_channels,
                preserve_pretrained=pretrained,
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model


def create_model(
    model_name: str,
    modality: str,
    num_classes: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    backbone_profile: str = "baseline",
    pretrained: bool = False,
) -> nn.Module:
    if modality != "ais":
        return create_image_model(
            model_name,
            modality,
            num_classes,
            image_size,
            backbone_profile,
            pretrained,
        )
    builders = {
        "resnet": AISResNet,
        "vit": AISViT,
        "convnext": AISConvNeXt,
    }
    return builders[model_name](num_classes)


def architecture_name(
    model_name: str, modality: str, backbone_profile: str = "baseline"
) -> str:
    if modality == "ais":
        return {
            "resnet": "AISResNet residual MLP",
            "vit": "AISViT field-token Transformer",
            "convnext": "AISConvNeXt 1D",
        }[model_name]
    if backbone_profile == "transfer-small":
        return {
            "resnet": "torchvision ResNet-18",
            "vit": "torchvision ViT-B/16",
            "convnext": "torchvision ConvNeXt-Small",
        }[model_name]
    return {
        "resnet": "torchvision ResNet-50",
        "vit": "torchvision ViT-B/16",
        "convnext": "torchvision ConvNeXt-Base",
    }[model_name]


def classification_head(model: nn.Module, model_name: str) -> nn.Module:
    if model_name == "resnet":
        return model.fc
    if model_name == "vit":
        return model.heads
    if model_name == "convnext":
        return model.classifier
    raise ValueError(f"Unknown model: {model_name}")


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    head_parameter_ids = {
        id(parameter) for parameter in classification_head(model, model_name).parameters()
    }
    for parameter in model.parameters():
        if id(parameter) not in head_parameter_ids:
            parameter.requires_grad = trainable


def image_optimizer_groups(
    model: nn.Module,
    model_name: str,
    backbone_lr: float,
    head_lr: float,
) -> List[Dict[str, object]]:
    head_parameter_ids = {
        id(parameter) for parameter in classification_head(model, model_name).parameters()
    }
    backbone = [
        parameter for parameter in model.parameters() if id(parameter) not in head_parameter_ids
    ]
    head = [
        parameter for parameter in model.parameters() if id(parameter) in head_parameter_ids
    ]
    if not backbone or not head:
        raise RuntimeError("Could not split image backbone and classification head")
    return [
        {"params": backbone, "lr": backbone_lr, "group_name": "backbone"},
        {"params": head, "lr": head_lr, "group_name": "head"},
    ]


def class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Training data misses classes: {counts.tolist()}")
    weights = len(labels) / (num_classes * counts.astype(np.float64))
    return torch.as_tensor(weights, dtype=torch.float32)


def classification_metrics(
    labels: Sequence[int], predictions: Sequence[int]
) -> Dict[str, object]:
    true = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    if true.size == 0 or pred.shape != true.shape:
        raise ValueError("Metrics require non-empty, equally shaped labels and predictions")
    class_ids = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=class_ids, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true, pred, labels=class_ids, average="macro", zero_division=0
    )
    return {
        "acc": float((true == pred).mean()),
        "precision_macro": float(macro_precision),
        "recall_macro": float(macro_recall),
        "f1_macro": float(macro_f1),
        "per_class": {
            CLASS_NAMES[index]: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in class_ids
        },
        "confusion_matrix": confusion_matrix(true, pred, labels=class_ids).tolist(),
    }


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
    sampler: Optional[WeightedRandomSampler] = None,
) -> DataLoader:
    if sampler is not None and shuffle:
        raise ValueError("shuffle and sampler cannot both be enabled")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def domain_class_balanced_sampler(
    labels: Sequence[int], domains: Sequence[int], seed: int
) -> WeightedRandomSampler:
    if len(labels) != len(domains) or not labels:
        raise ValueError("labels and domains must be non-empty and equally sized")
    cells = [(int(domain), int(label)) for domain, label in zip(domains, labels)]
    counts = Counter(cells)
    expected_cells = {
        (domain, class_id)
        for domain in sorted(set(int(value) for value in domains))
        for class_id in range(len(CLASS_NAMES))
    }
    missing = sorted(expected_cells - set(counts))
    if missing:
        raise ValueError(f"Domain/class sampler is missing cells: {missing}")
    weights = torch.as_tensor(
        [1.0 / counts[cell] for cell in cells], dtype=torch.double
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True, generator=generator
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_index, (inputs, labels, _) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        if channels_last and inputs.ndim == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits = model(inputs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum())
        total += batch_size
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    if total == 0:
        raise RuntimeError("Training loader yielded no samples")
    return {"loss": total_loss / total, "acc": correct / total}


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    channels_last: bool = False,
    include_predictions: bool = False,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels: List[int] = []
    all_predictions: List[int] = []
    prediction_rows: List[Dict[str, object]] = []
    for inputs, labels, sample_ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        if channels_last and inputs.ndim == 4:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            logits = model(inputs)
            batch_loss = criterion(logits, labels)
        probabilities = torch.softmax(logits.float(), dim=1)
        predictions = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(batch_loss) * batch_size
        total += batch_size
        labels_cpu = labels.cpu().numpy().astype(int)
        predictions_cpu = predictions.cpu().numpy().astype(int)
        probabilities_cpu = probabilities.cpu().numpy()
        all_labels.extend(labels_cpu.tolist())
        all_predictions.extend(predictions_cpu.tolist())
        if include_predictions:
            for index, sample_id in enumerate(sample_ids):
                row: Dict[str, object] = {
                    "sample_id": sample_id,
                    "label": int(labels_cpu[index]),
                    "prediction": int(predictions_cpu[index]),
                }
                for class_index, class_name in enumerate(CLASS_NAMES):
                    row[f"prob_{class_name.replace(' & ', '_').replace(' ', '_')}"] = float(
                        probabilities_cpu[index, class_index]
                    )
                prediction_rows.append(row)
    metrics = classification_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / max(total, 1)
    metrics["sample_count"] = total
    return metrics, prediction_rows


def compact_metrics(metrics: Mapping[str, object], prefix: str = "") -> Dict[str, float]:
    return {
        f"{prefix}acc": float(metrics["acc"]),
        f"{prefix}precision_macro": float(metrics["precision_macro"]),
        f"{prefix}recall_macro": float(metrics["recall_macro"]),
        f"{prefix}f1_macro": float(metrics["f1_macro"]),
    }


def checkpoint_rule(metric: str) -> str:
    if metric == "macro_f1":
        return "maximum target_val macro F1; ACC tie break"
    if metric == "acc":
        return "maximum target_val ACC; macro F1 tie break"
    raise ValueError(f"Unknown checkpoint metric: {metric}")


def checkpoint_key(metrics: Mapping[str, object], metric: str) -> Tuple[float, float]:
    if metric == "macro_f1":
        return float(metrics["f1_macro"]), float(metrics["acc"])
    if metric == "acc":
        return float(metrics["acc"]), float(metrics["f1_macro"])
    raise ValueError(f"Unknown checkpoint metric: {metric}")


def is_better(
    metrics: Mapping[str, object],
    best_key: Optional[Tuple[float, float]],
    metric: str = "acc",
) -> bool:
    key = checkpoint_key(metrics, metric)
    return best_key is None or key > best_key


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(path.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    file_handler = logging.FileHandler(path, encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_datasets(
    manifest: VDS2RawManifest,
    image_stats: Mapping[str, Sequence[float]],
    modality: str,
    image_augmentation: bool,
    image_size: int,
    train_resize_size: int,
) -> Tuple[Dataset, SingleModalityDataset, List[int], List[int]]:
    source = SingleModalityDataset(
        manifest,
        "source_train",
        image_stats,
        modality,
        augment=image_augmentation,
        image_size=image_size,
        train_resize_size=train_resize_size,
    )
    target = SingleModalityDataset(
        manifest,
        "target_train",
        image_stats,
        modality,
        augment=image_augmentation,
        image_size=image_size,
        train_resize_size=train_resize_size,
    )
    validation = SingleModalityDataset(
        manifest,
        "target_val",
        image_stats,
        modality,
        augment=False,
        image_size=image_size,
        train_resize_size=train_resize_size,
    )
    labels = list(source.labels) + list(target.labels)
    domains = [0] * len(source) + [1] * len(target)
    return ConcatDataset([source, target]), validation, labels, domains


def load_own_checkpoint(path: Path, device: torch.device) -> Mapping[str, object]:
    """Load our checkpoint on both old and new PyTorch releases."""

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_one(
    args: argparse.Namespace,
    manifest: VDS2RawManifest,
    image_stats: Mapping[str, Sequence[float]],
    run_root: Path,
    model_name: str,
    modality: str,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    run_dir = run_root / model_name / modality / f"seed_{seed}"
    result_path = run_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"Skipping completed run: {model_name}/{modality}/seed_{seed}")
            return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "train.log")
    seed_everything(seed)
    train_dataset, val_dataset, train_labels, train_domains = build_datasets(
        manifest,
        image_stats,
        modality,
        args.image_augmentation,
        args.image_size,
        args.train_resize_size,
    )
    train_sampler = (
        domain_class_balanced_sampler(train_labels, train_domains, seed)
        if args.domain_class_balanced_sampling
        else None
    )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        train_sampler is None,
        args.num_workers,
        device,
        seed,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device, seed
    )
    model = create_model(
        model_name,
        modality,
        len(CLASS_NAMES),
        image_size=args.image_size,
        backbone_profile=args.backbone_profile,
        pretrained=args.pretrained and modality != "ais",
    ).to(device)
    run_amp = bool(args.amp and modality != "ais" and device.type == "cuda")
    run_amp_dtype = (
        torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    )
    run_channels_last = bool(
        args.channels_last
        and modality != "ais"
        and model_name in {"resnet", "convnext"}
    )
    if run_channels_last:
        model = model.to(memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    learning_rate = args.lr_ais if modality == "ais" else args.lr_image
    backbone_learning_rate = (
        args.lr_backbone
        if modality != "ais" and args.lr_backbone is not None
        else learning_rate
    )
    run_epochs = args.ais_epochs if modality == "ais" else args.epochs
    run_patience = args.ais_patience if modality == "ais" else args.patience
    run_warmup_epochs = 0 if modality == "ais" else args.warmup_epochs
    run_label_smoothing = 0.0 if modality == "ais" else args.label_smoothing
    if modality != "ais" and args.lr_backbone is not None:
        optimizer_parameters = image_optimizer_groups(
            model,
            model_name,
            backbone_learning_rate,
            learning_rate,
        )
    else:
        optimizer_parameters = model.parameters()
    if modality != "ais" and args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, model_name, False)
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=run_amp and run_amp_dtype == torch.float16,
        init_scale=1024.0,
    )

    def learning_rate_factor(epoch_index: int) -> float:
        if run_warmup_epochs > 0 and epoch_index < run_warmup_epochs:
            return float(epoch_index + 1) / float(run_warmup_epochs)
        cosine_epochs = max(run_epochs - run_warmup_epochs, 1)
        progress = (epoch_index - run_warmup_epochs) / cosine_epochs
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_factor
    )
    weight_tensor = (
        class_weights(train_labels, len(CLASS_NAMES)).to(device)
        if args.class_weighting
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor, label_smoothing=run_label_smoothing
    )

    run_config = {
        "model": model_name,
        "architecture": architecture_name(
            model_name, modality, args.backbone_profile
        ),
        "modality": modality,
        "seed": seed,
        "train_partitions": list(TRAIN_PARTITIONS),
        "validation_partition": "target_val",
        "test_partition": "target_test" if args.evaluate_test else None,
        "checkpoint_selection": checkpoint_rule(args.checkpoint_metric),
        "metrics": ["ACC", "macro Precision", "macro Recall", "macro F1"],
        "image_input_shape": [
            3 if modality == "rgb" else 1,
            args.image_size,
            args.image_size,
        ]
        if modality != "ais"
        else None,
        "ais_semantic_fields": list(AIS_SEMANTIC_FIELDS) if modality == "ais" else None,
        "ais_numeric_components": list(AIS_VALUE_NAMES) if modality == "ais" else None,
        "ais_input": "heading_sin + heading_cos + one heading validity indicator"
        if modality == "ais"
        else None,
        "epochs": run_epochs,
        "patience": run_patience,
        "batch_size": args.batch_size,
        "learning_rate": learning_rate,
        "backbone_learning_rate": backbone_learning_rate,
        "backbone_profile": args.backbone_profile if modality != "ais" else None,
        "pretrained": bool(args.pretrained and modality != "ais"),
        "freeze_backbone_epochs": (
            args.freeze_backbone_epochs if modality != "ais" else 0
        ),
        "weight_decay": args.weight_decay,
        "label_smoothing": run_label_smoothing,
        "warmup_epochs": run_warmup_epochs,
        "automatic_mixed_precision": run_amp,
        "amp_dtype": args.amp_dtype if run_amp else None,
        "channels_last": run_channels_last,
        "image_size": args.image_size if modality != "ais" else None,
        "train_resize_size": args.train_resize_size if modality != "ais" else None,
        "random_crop": (
            args.image_augmentation and args.train_resize_size > args.image_size
            if modality != "ais"
            else False
        ),
        "class_weighting": args.class_weighting,
        "domain_class_balanced_sampling": args.domain_class_balanced_sampling,
        "image_augmentation": args.image_augmentation if modality != "ais" else False,
        "parameter_count": parameter_count,
        "started_at": utc_free_timestamp(),
    }
    atomic_write_json(run_dir / "config.json", run_config)
    logger.info(
        "start model=%s modality=%s seed=%d architecture=%s parameters=%d device=%s",
        model_name,
        modality,
        seed,
        run_config["architecture"],
        parameter_count,
        device,
    )
    logger.info(
        "protocol train=source_train+target_train val=target_val test=%s",
        "target_test after selection" if args.evaluate_test else "disabled",
    )

    best_key: Optional[Tuple[float, float]] = None
    best_epoch = 0
    best_metrics: Optional[Dict[str, object]] = None
    stale_epochs = 0
    epoch_rows: List[Dict[str, object]] = []
    checkpoint_path = run_dir / "best.pt"
    started = time.perf_counter()
    max_batches = 1 if args.smoke_test else None

    for epoch in range(1, run_epochs + 1):
        epoch_started = time.perf_counter()
        if (
            modality != "ais"
            and args.freeze_backbone_epochs > 0
            and epoch == args.freeze_backbone_epochs + 1
        ):
            set_backbone_trainable(model, model_name, True)
            logger.info("unfroze image backbone at epoch=%d", epoch)
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.grad_clip,
            scaler,
            use_amp=run_amp,
            amp_dtype=run_amp_dtype,
            channels_last=run_channels_last,
            max_batches=max_batches,
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=run_amp,
            amp_dtype=run_amp_dtype,
            channels_last=run_channels_last,
        )
        group_lrs = {
            str(group.get("group_name", "all")): float(group["lr"])
            for group in optimizer.param_groups
        }
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            **compact_metrics(val_metrics, "val_"),
            "learning_rate": group_lrs.get("head", group_lrs.get("all")),
            "backbone_learning_rate": group_lrs.get(
                "backbone", group_lrs.get("all")
            ),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        epoch_rows.append(row)
        write_csv(run_dir / "epochs.csv", epoch_rows)
        logger.info(
            "epoch=%d train_loss=%.6f train_acc=%.4f val_acc=%.4f "
            "val_precision=%.4f val_recall=%.4f val_f1=%.4f",
            epoch,
            row["train_loss"],
            row["train_acc"],
            row["val_acc"],
            row["val_precision_macro"],
            row["val_recall_macro"],
            row["val_f1_macro"],
        )
        if is_better(val_metrics, best_key, args.checkpoint_metric):
            best_key = checkpoint_key(val_metrics, args.checkpoint_metric)
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "run_config": run_config,
                },
                checkpoint_path,
            )
            logger.info(
                "new best checkpoint epoch=%d val_acc=%.4f val_f1=%.4f",
                epoch,
                val_metrics["acc"],
                val_metrics["f1_macro"],
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= run_patience:
            logger.info("early stopping at epoch=%d best_epoch=%d", epoch, best_epoch)
            break

    if best_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("Training did not produce a validation-selected checkpoint")
    checkpoint = load_own_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics: Optional[Dict[str, object]] = None
    if args.evaluate_test:
        # This is intentionally the first and only target_test evaluation in a run.
        test_dataset = SingleModalityDataset(
            manifest,
            "target_test",
            image_stats,
            modality,
            augment=False,
            image_size=args.image_size,
            train_resize_size=args.train_resize_size,
        )
        test_loader = make_loader(
            test_dataset, args.batch_size, False, args.num_workers, device, seed
        )
        test_metrics, prediction_rows = evaluate(
            model,
            test_loader,
            criterion,
            device,
            use_amp=run_amp,
            amp_dtype=run_amp_dtype,
            channels_last=run_channels_last,
            include_predictions=True,
        )
        write_csv(run_dir / "target_test_predictions.csv", prediction_rows)
        atomic_write_json(run_dir / "target_test_metrics.json", test_metrics)
        logger.info(
            "final target_test best_epoch=%d ACC=%.4f Precision=%.4f Recall=%.4f F1=%.4f",
            best_epoch,
            test_metrics["acc"],
            test_metrics["precision_macro"],
            test_metrics["recall_macro"],
            test_metrics["f1_macro"],
        )

    result: Dict[str, object] = {
        "status": "complete",
        "model": model_name,
        "architecture": run_config["architecture"],
        "modality": modality,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(epoch_rows),
        "parameter_count": parameter_count,
        "checkpoint": str(checkpoint_path),
        "validation": best_metrics,
        "target_test": test_metrics,
        "duration_seconds": time.perf_counter() - started,
        "completed_at": utc_free_timestamp(),
    }
    atomic_write_json(result_path, result)
    logger.info("complete duration_seconds=%.2f", result["duration_seconds"])
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def flat_result(result: Mapping[str, object]) -> Dict[str, object]:
    validation = result["validation"]
    target_test = result.get("target_test")
    row: Dict[str, object] = {
        "model": result["model"],
        "architecture": result["architecture"],
        "modality": result["modality"],
        "seed": result["seed"],
        "best_epoch": result["best_epoch"],
        "epochs_completed": result["epochs_completed"],
        "parameter_count": result["parameter_count"],
        **compact_metrics(validation, "val_"),
        "duration_seconds": result["duration_seconds"],
        "checkpoint": result["checkpoint"],
    }
    if target_test is not None:
        row.update(compact_metrics(target_test, "test_"))
    return row


def numeric_summary(values: Sequence[float]) -> Dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def summarize(results: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for result in results:
        grouped[(str(result["model"]), str(result["modality"]))].append(result)
    rows: List[Dict[str, object]] = []
    for (model_name, modality), group in sorted(grouped.items()):
        row: Dict[str, object] = {
            "model": model_name,
            "modality": modality,
            "runs": len(group),
            "seeds": [int(result["seed"]) for result in group],
        }
        for split, source_key in (("val", "validation"), ("test", "target_test")):
            available = [result[source_key] for result in group if result.get(source_key)]
            for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
                if available:
                    summary = numeric_summary(
                        [float(metrics[metric]) for metrics in available]
                    )
                    for statistic, value in summary.items():
                        row[f"{split}_{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def flatten_summary(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    flat_rows: List[Dict[str, object]] = []
    for row in summary_rows:
        flat: Dict[str, object] = {}
        for key, value in row.items():
            flat[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
        flat_rows.append(flat)
    return flat_rows


def protocol_audit(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "task": "VDS2Raw three-class single-modality baselines",
        "grid": {
            "models": args.models,
            "modalities": args.modalities,
            "seeds": args.seeds,
            "planned_runs": len(args.models) * len(args.modalities) * len(args.seeds),
        },
        "partitions": {
            "parameter_updates": ["source_train", "target_train"],
            "statistics_fit": ["source_train", "target_train"],
            "checkpoint_selection": "target_val",
            "final_evaluation": "target_test" if args.evaluate_test else "disabled",
        },
        "checkpoint_rule": checkpoint_rule(args.checkpoint_metric),
        "optimization": {
            "backbone_profile": args.backbone_profile,
            "pretrained": args.pretrained,
            "head_learning_rate": args.lr_image,
            "backbone_learning_rate": (
                args.lr_backbone if args.lr_backbone is not None else args.lr_image
            ),
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "domain_class_balanced_sampling": args.domain_class_balanced_sampling,
            "class_weighting": args.class_weighting,
        },
        "metrics": {
            "ACC": "overall accuracy",
            "Precision": "macro average across Cargo, Fishing, Sailing & Pleasure",
            "Recall": "macro average across Cargo, Fishing, Sailing & Pleasure",
            "F1": "macro average across Cargo, Fishing, Sailing & Pleasure",
        },
        "images": {
            "RGB": "B4/B3/B2, native 3x96x96",
            "NIR": "B8, native 1x96x96",
            "normalization_fit": ["source_train", "target_train"],
            "model_input_size": (
                [args.image_size, args.image_size]
                if any(modality != "ais" for modality in args.modalities)
                else None
            ),
            "training_augmentation": (
                (
                    f"resize {args.train_resize_size}, random "
                    f"{args.image_size}x{args.image_size} crop, horizontal flip"
                    if args.train_resize_size > args.image_size
                    else f"resize {args.image_size}, no random crop, horizontal flip"
                )
                if args.image_augmentation else "disabled"
            ),
        },
        "AIS": {
            "semantic_fields": list(AIS_SEMANTIC_FIELDS),
            "numeric_components": list(AIS_VALUE_NAMES),
            "validity_masks": ["heading_valid"],
            "input_dimension": AIS_INPUT_DIM,
            "excluded": "all nav_* values/masks and every metadata/leakage field",
        },
        "target_test_invariant": (
            "not loaded until the validation-selected checkpoint has been restored; "
            "evaluated once per completed run"
        ),
    }


def environment_report(device: torch.device) -> Dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": getattr(sys.modules.get("torchvision"), "__version__", "unknown"),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.run_name or datetime.now().strftime("single_modal_%Y%m%d_%H%M%S")
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    manifest = VDS2RawManifest(dataset_root)
    dataset_audit = manifest.audit(validate_npz=True)
    image_stats = compute_train_image_stats(manifest)
    audit = protocol_audit(args)
    atomic_write_json(run_root / "dataset_audit.json", dataset_audit)
    atomic_write_json(run_root / "train_only_image_stats.json", image_stats)
    atomic_write_json(run_root / "protocol_audit.json", audit)
    atomic_write_json(run_root / "environment.json", environment_report(device))
    atomic_write_json(
        run_root / "invocation.json",
        {
            "arguments": vars(args) | {"dataset_root": str(dataset_root), "output_root": str(output_root)},
            "started_at": utc_free_timestamp(),
        },
    )
    print(f"Dataset audit: {dataset_audit['status']} ({dataset_audit['sample_count']} samples)")
    print(f"Planned runs: {audit['grid']['planned_runs']}")
    print(f"Output: {run_root}")
    if args.audit_only:
        print("Audit-only mode complete; no model or target_test was loaded.")
        return 0

    results: List[Dict[str, object]] = []
    for model_name in args.models:
        for modality in args.modalities:
            for seed in args.seeds:
                result = run_one(
                    args,
                    manifest,
                    image_stats,
                    run_root,
                    model_name,
                    modality,
                    seed,
                    device,
                )
                results.append(result)
                write_csv(run_root / "results.csv", [flat_result(item) for item in results])
                summary_rows = summarize(results)
                atomic_write_json(run_root / "summary.json", summary_rows)
                write_csv(run_root / "summary.csv", flatten_summary(summary_rows))

    print(f"Completed runs: {len(results)}")
    print(f"Results: {run_root / 'results.csv'}")
    print(f"Summary: {run_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
