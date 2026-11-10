"""Leakage-safe grouped 5-fold image baselines for VDS2Raw.

The day3 source partition stays fixed.  All non-day3 samples are reassigned in
each fold to target_train/target_val/target_test by connected scene/MMSI groups.
Each fold trains one model seed with target-priority sampling, head warm-up,
last-stage fine-tuning, a short target-only refinement, ACC-first checkpoint
selection, and horizontal-flip test-time augmentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from VDS2Raw_JMDA.single_modal_baselines import (
    DEFAULT_SEEDS,
    MODEL_NAMES,
    SingleModalityDataset,
    architecture_name,
    atomic_write_json,
    checkpoint_key,
    classification_head,
    classification_metrics,
    compact_metrics,
    create_model,
    environment_report,
    evaluate,
    flat_result,
    flatten_summary,
    image_optimizer_groups,
    make_loader,
    parse_csv_values,
    seed_everything,
    summarize,
    utc_free_timestamp,
    write_csv,
)
from VDS2Raw_JMDA.vds2raw_dataset import (
    CLASS_NAMES,
    PARTITIONS,
    VDS2RawManifest,
    VDSRecord,
    compute_train_image_stats,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_FOLD_SEEDS = DEFAULT_SEEDS


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


class FoldManifest:
    """Manifest-compatible view with fold-specific partition assignments."""

    def __init__(
        self,
        base: VDS2RawManifest,
        assignments: Mapping[str, str],
    ):
        self.base = base
        self.dataset_root = base.dataset_root
        self.manifest_path = base.manifest_path
        self.records = [
            replace(record, partition=assignments.get(record.sample_id, record.partition))
            for record in base.records
        ]
        self.by_partition = {
            partition: [record for record in self.records if record.partition == partition]
            for partition in PARTITIONS
        }

    def resolve_npz(self, record: VDSRecord) -> Path:
        return self.base.resolve_npz(record)


def connected_group_ids(records: Sequence[VDSRecord]) -> List[str]:
    """Join samples transitively when either scene or non-empty MMSI matches."""

    union = UnionFind(len(records))
    first_seen: Dict[Tuple[str, str], int] = {}
    for index, record in enumerate(records):
        keys = [("scene", record.scene)]
        if record.mmsi:
            keys.append(("mmsi", record.mmsi))
        for key in keys:
            if key in first_seen:
                union.union(index, first_seen[key])
            else:
                first_seen[key] = index
    members: Dict[int, List[str]] = defaultdict(list)
    for index, record in enumerate(records):
        members[union.find(index)].append(record.sample_id)
    canonical = {
        root: min(sample_ids) for root, sample_ids in members.items()
    }
    return [canonical[union.find(index)] for index in range(len(records))]


def distribution_score(
    selected: Sequence[VDSRecord],
    reference: Sequence[VDSRecord],
    desired_size: float,
) -> float:
    if not selected:
        return float("inf")
    score = 4.0 * ((len(selected) - desired_size) / max(desired_size, 1.0)) ** 2
    selected_classes = Counter(record.label for record in selected)
    reference_classes = Counter(record.label for record in reference)
    for class_name in CLASS_NAMES:
        expected = desired_size * reference_classes[class_name] / len(reference)
        if selected_classes[class_name] == 0:
            score += 100.0
        score += ((selected_classes[class_name] - expected) / max(expected, 1.0)) ** 2
    selected_days = Counter(record.day for record in selected)
    reference_days = Counter(record.day for record in reference)
    for day, count in reference_days.items():
        expected = desired_size * count / len(reference)
        score += 0.20 * ((selected_days[day] - expected) / max(expected, 1.0)) ** 2
    return score


def choose_outer_folds(
    target_records: Sequence[VDSRecord],
    group_ids: Sequence[str],
    split_seed: int,
    candidates: int = 200,
) -> List[List[int]]:
    labels = np.asarray([record.label_id for record in target_records])
    indices = np.arange(len(target_records))
    best_score = float("inf")
    best: Optional[List[List[int]]] = None
    desired_size = len(target_records) / 5.0
    for offset in range(candidates):
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=split_seed + offset
        )
        folds: List[List[int]] = []
        score = 0.0
        for _, test_indices in splitter.split(indices, labels, groups=group_ids):
            chosen = test_indices.tolist()
            folds.append(chosen)
            score += distribution_score(
                [target_records[index] for index in chosen],
                target_records,
                desired_size,
            )
        if score < best_score:
            best_score = score
            best = folds
    if best is None:
        raise RuntimeError("Could not construct outer grouped folds")
    return best


def choose_validation_indices(
    remaining_indices: Sequence[int],
    target_records: Sequence[VDSRecord],
    group_ids: Sequence[str],
    split_seed: int,
    desired_size: int = 20,
    candidates: int = 120,
) -> List[int]:
    remaining = list(remaining_indices)
    records = [target_records[index] for index in remaining]
    labels = np.asarray([record.label_id for record in records])
    groups = [group_ids[index] for index in remaining]
    local_indices = np.arange(len(records))
    best_score = float("inf")
    best: Optional[List[int]] = None
    for offset in range(candidates):
        splitter = StratifiedGroupKFold(
            n_splits=6, shuffle=True, random_state=split_seed + offset
        )
        for _, val_local in splitter.split(local_indices, labels, groups=groups):
            selected = [records[index] for index in val_local]
            score = distribution_score(selected, records, desired_size)
            if score < best_score:
                best_score = score
                best = [remaining[index] for index in val_local]
    if best is None:
        raise RuntimeError("Could not construct grouped validation split")
    return best


def build_grouped_folds(
    manifest: VDS2RawManifest,
    split_seed: int = 20261110,
) -> List[Dict[str, object]]:
    source_records = [record for record in manifest.records if record.day == "day3"]
    target_records = [record for record in manifest.records if record.day != "day3"]
    if len(source_records) != 116 or len(target_records) != 155:
        raise ValueError(
            f"Expected 116 day3 source and 155 non-day3 target samples, got "
            f"{len(source_records)} and {len(target_records)}"
        )
    group_ids = connected_group_ids(target_records)
    outer_folds = choose_outer_folds(target_records, group_ids, split_seed)
    all_indices = set(range(len(target_records)))
    folds: List[Dict[str, object]] = []
    for fold_index, test_indices_list in enumerate(outer_folds, start=1):
        test_indices = set(test_indices_list)
        remaining = sorted(all_indices - test_indices)
        val_indices = set(
            choose_validation_indices(
                remaining,
                target_records,
                group_ids,
                split_seed + 10_000 * fold_index,
            )
        )
        train_indices = set(remaining) - val_indices
        assignments = {record.sample_id: "source_train" for record in source_records}
        for index, record in enumerate(target_records):
            if index in test_indices:
                assignments[record.sample_id] = "target_test"
            elif index in val_indices:
                assignments[record.sample_id] = "target_val"
            elif index in train_indices:
                assignments[record.sample_id] = "target_train"
            else:
                raise RuntimeError(f"Unassigned target sample: {record.sample_id}")
        folds.append(
            {
                "fold": fold_index,
                "assignments": assignments,
                "group_ids": {
                    record.sample_id: group_ids[index]
                    for index, record in enumerate(target_records)
                },
            }
        )
    test_coverage = Counter()
    for fold in folds:
        for sample_id, partition in fold["assignments"].items():
            if partition == "target_test":
                test_coverage[sample_id] += 1
    target_ids = {record.sample_id for record in target_records}
    if set(test_coverage) != target_ids or set(test_coverage.values()) != {1}:
        raise RuntimeError("Every target sample must be test exactly once across five folds")
    return folds


def build_sample_stratified_folds(
    manifest: VDS2RawManifest,
    split_seed: int = 20261110,
    validation_size: int = 20,
) -> List[Dict[str, object]]:
    """Build an intentionally non-grouped comparison protocol.

    This split is useful only for quantifying the effect of the protocol.  It
    does not prevent scenes (or the two repeated MMSIs) from crossing target
    partitions and must never be reported as leakage-safe grouped CV.
    """

    source_records = [record for record in manifest.records if record.day == "day3"]
    target_records = [record for record in manifest.records if record.day != "day3"]
    if len(source_records) != 116 or len(target_records) != 155:
        raise ValueError("Sample-stratified comparison expects the 116/155 corpus")
    labels = np.asarray([record.label_id for record in target_records])
    indices = np.arange(len(target_records))
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    folds: List[Dict[str, object]] = []
    for fold_index, (remaining, test_indices) in enumerate(
        outer.split(indices, labels), start=1
    ):
        validation_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validation_size,
            random_state=split_seed + 10_000 * fold_index,
        )
        train_local, val_local = next(
            validation_splitter.split(remaining, labels[remaining])
        )
        train_indices = set(remaining[train_local].tolist())
        val_indices = set(remaining[val_local].tolist())
        test_index_set = set(test_indices.tolist())
        assignments = {record.sample_id: "source_train" for record in source_records}
        for index, record in enumerate(target_records):
            if index in test_index_set:
                partition = "target_test"
            elif index in val_indices:
                partition = "target_val"
            elif index in train_indices:
                partition = "target_train"
            else:
                raise RuntimeError(f"Unassigned target sample: {record.sample_id}")
            assignments[record.sample_id] = partition
        folds.append({"fold": fold_index, "assignments": assignments})
    return folds


def partition_counts(records: Iterable[VDSRecord]) -> Dict[str, Dict[str, int]]:
    grouped: Dict[str, List[VDSRecord]] = defaultdict(list)
    for record in records:
        grouped[record.partition].append(record)
    return {
        partition: {
            "total": len(grouped[partition]),
            **{
                class_name: sum(record.label == class_name for record in grouped[partition])
                for class_name in CLASS_NAMES
            },
        }
        for partition in PARTITIONS
    }


def fold_audit(
    manifest: FoldManifest,
    fold_number: int,
    allow_scene_mmsi_overlap: bool = False,
) -> Dict[str, object]:
    overlaps: Dict[str, Dict[str, int]] = {}
    errors: List[str] = []
    warnings: List[str] = []
    for left_index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[left_index + 1 :]:
            left_records = manifest.by_partition[left]
            right_records = manifest.by_partition[right]
            sample_overlap = {r.sample_id for r in left_records} & {
                r.sample_id for r in right_records
            }
            scene_overlap = {r.scene for r in left_records if r.scene} & {
                r.scene for r in right_records if r.scene
            }
            mmsi_overlap = {r.mmsi for r in left_records if r.mmsi} & {
                r.mmsi for r in right_records if r.mmsi
            }
            key = f"{left}__{right}"
            overlaps[key] = {
                "sample": len(sample_overlap),
                "scene": len(scene_overlap),
                "mmsi": len(mmsi_overlap),
            }
            if sample_overlap:
                errors.append(f"{key}: sample={len(sample_overlap)}")
            if scene_overlap or mmsi_overlap:
                message = (
                    f"{key}: sample={len(sample_overlap)} scene={len(scene_overlap)} "
                    f"mmsi={len(mmsi_overlap)}"
                )
                if allow_scene_mmsi_overlap:
                    warnings.append(message)
                else:
                    errors.append(message)
    counts = partition_counts(manifest.records)
    for partition in PARTITIONS:
        missing = [name for name in CLASS_NAMES if counts[partition][name] == 0]
        if missing:
            errors.append(f"{partition} misses classes {missing}")
    report = {
        "status": (
            "FAIL"
            if errors
            else "WARNING_SCENE_OR_MMSI_OVERLAP"
            if warnings
            else "PASS"
        ),
        "fold": fold_number,
        "counts": counts,
        "overlaps": overlaps,
        "errors": errors,
        "warnings": warnings,
    }
    if errors:
        raise ValueError("Fold leakage audit failed: " + "; ".join(errors))
    return report


def target_priority_sampler(
    source_count: int,
    target_count: int,
    target_mass: float,
    seed: int,
) -> WeightedRandomSampler:
    if source_count < 1 or target_count < 1 or target_mass <= 0:
        raise ValueError("source_count, target_count, and target_mass must be positive")
    weights = torch.as_tensor(
        [1.0 / source_count] * source_count
        + [target_mass / target_count] * target_count,
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights,
        num_samples=source_count + target_count,
        replacement=True,
        generator=generator,
    )


def set_last_stage_trainable(model: nn.Module, model_name: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    modules: List[nn.Module] = [classification_head(model, model_name)]
    if model_name == "resnet":
        modules.append(model.layer4)
    elif model_name == "vit":
        modules.extend([model.encoder.layers[-1], model.encoder.ln])
    elif model_name == "convnext":
        modules.append(model.features[-1])
    else:
        raise ValueError(f"Unknown image model: {model_name}")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True


def freeze_batch_norm_stats(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def train_epoch_partial(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
    channels_last: bool,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    if model_name == "resnet":
        freeze_batch_norm_stats(model)
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_index, (inputs, labels, _) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            logits = model(inputs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        correct += int((logits.detach().argmax(dim=1) == labels).sum())
        total += batch_size
        if max_batches is not None and batch_index + 1 >= max_batches:
            break
    return {"loss": total_loss / total, "acc": correct / total}


@torch.inference_mode()
def evaluate_prediction_mode(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
    mode: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    if mode not in {"identity", "horizontal_flip_mean"}:
        raise ValueError(f"Unknown prediction mode: {mode}")
    model.eval()
    all_labels: List[int] = []
    all_predictions: List[int] = []
    rows: List[Dict[str, object]] = []
    total_loss = 0.0
    for inputs, labels, sample_ids in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if channels_last:
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
        ):
            probabilities = torch.softmax(model(inputs).float(), dim=1)
            if mode == "horizontal_flip_mean":
                flipped = torch.flip(inputs, dims=(-1,))
                if channels_last:
                    flipped = flipped.contiguous(memory_format=torch.channels_last)
                probabilities = 0.5 * (
                    probabilities + torch.softmax(model(flipped).float(), dim=1)
                )
        predictions = probabilities.argmax(dim=1)
        total_loss += float(
            nn.functional.nll_loss(
                torch.log(probabilities.clamp_min(1e-12)), labels, reduction="sum"
            )
        )
        labels_cpu = labels.cpu().numpy().astype(int)
        predictions_cpu = predictions.cpu().numpy().astype(int)
        probabilities_cpu = probabilities.cpu().numpy()
        all_labels.extend(labels_cpu.tolist())
        all_predictions.extend(predictions_cpu.tolist())
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
            rows.append(row)
    metrics = classification_metrics(all_labels, all_predictions)
    metrics["loss"] = total_loss / len(all_labels)
    metrics["sample_count"] = len(all_labels)
    metrics["prediction_mode"] = mode
    return metrics, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grouped 5-fold VDS2Raw image baselines")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Downloads\VDS2Raw\threeclass_ready"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_DIR / "single_modal_runs"
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--models",
        type=lambda value: parse_csv_values(value, MODEL_NAMES, "models"),
        default=list(MODEL_NAMES),
    )
    parser.add_argument(
        "--modalities",
        type=lambda value: parse_csv_values(value, ("rgb", "nir"), "modalities"),
        default=["rgb", "nir"],
    )
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--split-seed", type=int, default=20261110)
    parser.add_argument(
        "--split-mode",
        choices=("grouped", "sample-stratified"),
        default="grouped",
        help="sample-stratified is an explicitly leakage-prone protocol comparison",
    )
    parser.add_argument("--stage1-epochs", type=int, default=30)
    parser.add_argument("--stage1-patience", type=int, default=10)
    parser.add_argument("--head-only-epochs", type=int, default=5)
    parser.add_argument("--stage2-epochs", type=int, default=8)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--stage2-backbone-lr", type=float, default=5e-6)
    parser.add_argument("--stage2-head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--target-mass", type=float, default=2.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-evaluate-test", action="store_true")
    return parser


def parse_folds(value: str) -> List[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError("folds must be a comma-separated subset of 1,2,3,4,5")
    return list(dict.fromkeys(folds))


def cosine_factor(epoch_index: int, total_epochs: int) -> float:
    return 0.5 * (1.0 + math.cos(math.pi * epoch_index / max(total_epochs, 1)))


def run_fold_model(
    args: argparse.Namespace,
    manifest: FoldManifest,
    image_stats: Mapping[str, Sequence[float]],
    run_root: Path,
    fold_number: int,
    model_name: str,
    modality: str,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    run_dir = run_root / model_name / modality / f"fold_{fold_number}"
    result_path = run_dir / "result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"Skipping completed run: {model_name}/{modality}/fold_{fold_number}")
            return existing
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    source = SingleModalityDataset(
        manifest, "source_train", image_stats, modality, True, args.image_size, args.image_size
    )
    target = SingleModalityDataset(
        manifest, "target_train", image_stats, modality, True, args.image_size, args.image_size
    )
    validation = SingleModalityDataset(
        manifest, "target_val", image_stats, modality, False, args.image_size, args.image_size
    )
    stage1_dataset = ConcatDataset([source, target])
    stage1_sampler = target_priority_sampler(
        len(source), len(target), args.target_mass, seed
    )
    stage1_loader = make_loader(
        stage1_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
        seed,
        sampler=stage1_sampler,
    )
    stage2_loader = make_loader(
        target, args.batch_size, True, args.num_workers, device, seed + 1000
    )
    val_loader = make_loader(
        validation, args.batch_size, False, args.num_workers, device, seed
    )
    model = create_model(
        model_name,
        modality,
        len(CLASS_NAMES),
        image_size=args.image_size,
        backbone_profile="baseline",
        pretrained=True,
    ).to(device)
    channels_last = model_name in {"resnet", "convnext"} and device.type == "cuda"
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    use_amp = not args.no_amp and device.type == "cuda"
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classification_head(model, model_name).parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        image_optimizer_groups(model, model_name, args.backbone_lr, args.head_lr),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_factor(epoch, args.stage1_epochs),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    evaluate_test = not args.no_evaluate_test and not args.smoke_test
    run_config = {
        "protocol": "grouped 5-fold CV",
        "fold": fold_number,
        "seed": seed,
        "model": model_name,
        "architecture": architecture_name(model_name, modality, "baseline"),
        "modality": modality,
        "pretrained": True,
        "input_size": [args.image_size, args.image_size],
        "random_crop": False,
        "target_priority_mass": args.target_mass,
        "stage1": "source_train + target_train",
        "stage2": "target_train only",
        "head_only_epochs": args.head_only_epochs,
        "partial_unfreeze": "classification head + final backbone stage",
        "freeze_resnet_batch_norm_stats": model_name == "resnet",
        "checkpoint_selection": "maximum target_val ACC; macro F1 tie break",
        "test_time_augmentation": "original + horizontal flip probability mean",
        "learning_rates": {
            "stage1_backbone": args.backbone_lr,
            "stage1_head": args.head_lr,
            "stage2_backbone": args.stage2_backbone_lr,
            "stage2_head": args.stage2_head_lr,
        },
        "counts": partition_counts(manifest.records),
        "normalization_fit": ["source_train", "target_train"],
    }
    atomic_write_json(run_dir / "config.json", run_config)
    best_key: Optional[Tuple[float, float]] = None
    best_metrics: Optional[Dict[str, object]] = None
    best_epoch_label = ""
    checkpoint_path = run_dir / "best.pt"
    epoch_rows: List[Dict[str, object]] = []
    stale = 0
    started = time.perf_counter()
    max_batches = 1 if args.smoke_test else None

    def maybe_checkpoint(metrics: Mapping[str, object], epoch_label: str) -> bool:
        nonlocal best_key, best_metrics, best_epoch_label, stale
        key = checkpoint_key(metrics, "acc")
        if best_key is None or key > best_key:
            best_key = key
            best_metrics = dict(metrics)
            best_epoch_label = epoch_label
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "validation_metrics": metrics,
                    "epoch_label": epoch_label,
                    "run_config": run_config,
                },
                checkpoint_path,
            )
            return True
        stale += 1
        return False

    stage1_epochs = 1 if args.smoke_test else args.stage1_epochs
    for epoch in range(1, stage1_epochs + 1):
        if epoch == args.head_only_epochs + 1:
            set_last_stage_trainable(model, model_name)
        train_metrics = train_epoch_partial(
            model,
            model_name,
            stage1_loader,
            optimizer,
            criterion,
            device,
            args.grad_clip,
            scaler,
            use_amp,
            channels_last,
            max_batches=max_batches,
        )
        val_metrics, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            amp_dtype=torch.bfloat16,
            channels_last=channels_last,
        )
        row = {
            "phase": "stage1",
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            **compact_metrics(val_metrics, "val_"),
        }
        epoch_rows.append(row)
        write_csv(run_dir / "epochs.csv", epoch_rows)
        improved = maybe_checkpoint(val_metrics, f"stage1:{epoch}")
        print(
            f"fold={fold_number} model={model_name} modality={modality} "
            f"stage1={epoch} train_acc={train_metrics['acc']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1_macro']:.4f}" 
            + (" best" if improved else "")
        )
        scheduler.step()
        if epoch > args.head_only_epochs and stale >= args.stage1_patience:
            break

    if not args.smoke_test and args.stage2_epochs > 0:
        # Refinement must start from the validation-selected stage-1 state.
        # Otherwise an over-fitted final stage-1 epoch can silently become the
        # starting point even though it was explicitly rejected by target_val.
        stage1_checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(stage1_checkpoint["model_state"])
        set_last_stage_trainable(model, model_name)
        optimizer = torch.optim.AdamW(
            image_optimizer_groups(
                model,
                model_name,
                args.stage2_backbone_lr,
                args.stage2_head_lr,
            ),
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.stage2_epochs
        )
        for epoch in range(1, args.stage2_epochs + 1):
            train_metrics = train_epoch_partial(
                model,
                model_name,
                stage2_loader,
                optimizer,
                criterion,
                device,
                args.grad_clip,
                scaler,
                use_amp,
                channels_last,
            )
            val_metrics, _ = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
                amp_dtype=torch.bfloat16,
                channels_last=channels_last,
            )
            row = {
                "phase": "stage2",
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                **compact_metrics(val_metrics, "val_"),
            }
            epoch_rows.append(row)
            write_csv(run_dir / "epochs.csv", epoch_rows)
            improved = maybe_checkpoint(val_metrics, f"stage2:{epoch}")
            print(
                f"fold={fold_number} model={model_name} modality={modality} "
                f"stage2={epoch} train_acc={train_metrics['acc']:.4f} "
                f"val_acc={val_metrics['acc']:.4f} val_f1={val_metrics['f1_macro']:.4f}"
                + (" best" if improved else "")
            )
            scheduler.step()

    if best_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("No validation-selected checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics: Optional[Dict[str, object]] = None
    if evaluate_test:
        test_dataset = SingleModalityDataset(
            manifest,
            "target_test",
            image_stats,
            modality,
            False,
            args.image_size,
            args.image_size,
        )
        test_loader = make_loader(
            test_dataset, args.batch_size, False, args.num_workers, device, seed
        )
        val_flip_metrics, _ = evaluate_prediction_mode(
            model,
            val_loader,
            device,
            use_amp,
            channels_last,
            "horizontal_flip_mean",
        )
        test_metrics, prediction_rows = evaluate_prediction_mode(
            model,
            test_loader,
            device,
            use_amp,
            channels_last,
            "horizontal_flip_mean",
        )
        test_metrics["prediction_mode_audit"] = {
            "selection": "fixed by protocol before target_test evaluation",
            "identity": compact_metrics(best_metrics),
            "horizontal_flip_mean": compact_metrics(val_flip_metrics),
        }
        write_csv(run_dir / "target_test_predictions.csv", prediction_rows)
        atomic_write_json(run_dir / "target_test_metrics.json", test_metrics)
        print(
            f"FINAL fold={fold_number} model={model_name} modality={modality} "
            f"best={best_epoch_label} ACC={test_metrics['acc']:.4f} "
            f"P={test_metrics['precision_macro']:.4f} "
            f"R={test_metrics['recall_macro']:.4f} F1={test_metrics['f1_macro']:.4f}"
        )
    result: Dict[str, object] = {
        "status": "complete",
        "fold": fold_number,
        "seed": seed,
        "model": model_name,
        "architecture": run_config["architecture"],
        "modality": modality,
        "best_epoch": best_epoch_label,
        "epochs_completed": len(epoch_rows),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "validation": best_metrics,
        "target_test": test_metrics,
        "duration_seconds": time.perf_counter() - started,
        "completed_at": utc_free_timestamp(),
    }
    atomic_write_json(result_path, result)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_fold_manifest(path: Path, manifest: FoldManifest) -> None:
    rows = [
        {
            "sample_id": record.sample_id,
            "partition": record.partition,
            "label": record.label,
            "label_id": record.label_id,
            "npz_relpath": record.npz_relpath,
            "scene": record.scene,
            "day": record.day,
            "mmsi": record.mmsi,
        }
        for record in manifest.records
    ]
    write_csv(path, rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    selected_folds = parse_folds(args.folds)
    if args.stage1_epochs < 1 or args.stage2_epochs < 0 or args.head_only_epochs < 0:
        raise ValueError("Invalid epoch arguments")
    if args.image_size < 1 or args.image_size % 16 != 0:
        raise ValueError("image_size must be positive and divisible by ViT patch size 16")
    if args.target_mass <= 0 or args.weight_decay < 0:
        raise ValueError("target_mass must be positive and weight_decay non-negative")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_name = args.run_name or datetime.now().strftime("grouped_cv_%Y%m%d_%H%M%S")
    run_root = output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    base = VDS2RawManifest(dataset_root)
    folds = (
        build_grouped_folds(base, args.split_seed)
        if args.split_mode == "grouped"
        else build_sample_stratified_folds(base, args.split_seed)
    )
    atomic_write_json(
        run_root / "protocol.json",
        {
            "protocol": (
                "scene/MMSI-connected grouped 5-fold cross-validation"
                if args.split_mode == "grouped"
                else "sample-stratified 5-fold comparison with scene/MMSI overlap"
            ),
            "source": "all 116 day3 samples fixed as source_train",
            "target": "all 155 non-day3 samples test exactly once",
            "models": args.models,
            "modalities": args.modalities,
            "selected_folds": selected_folds,
            "fold_seeds": list(DEFAULT_FOLD_SEEDS),
            "checkpoint_selection": "target_val ACC; macro F1 tie break",
            "training": "source+target with 2:1 target mass, then target-only refinement",
            "test_time_augmentation": "original + horizontal flip probability mean",
        },
    )
    atomic_write_json(run_root / "environment.json", environment_report(device))
    results: List[Dict[str, object]] = []
    for fold_data in folds:
        fold_number = int(fold_data["fold"])
        fold_manifest = FoldManifest(base, fold_data["assignments"])
        audit = fold_audit(
            fold_manifest,
            fold_number,
            allow_scene_mmsi_overlap=args.split_mode == "sample-stratified",
        )
        fold_dir = run_root / "folds" / f"fold_{fold_number}"
        atomic_write_json(fold_dir / "audit.json", audit)
        write_fold_manifest(fold_dir / "manifest.csv", fold_manifest)
        if fold_number not in selected_folds:
            continue
        image_stats = compute_train_image_stats(fold_manifest)
        atomic_write_json(fold_dir / "train_only_image_stats.json", image_stats)
        print(
            f"Fold {fold_number} audit {audit['status']} counts="
            f"{json.dumps(audit['counts'], ensure_ascii=False)}"
        )
        if args.audit_only:
            continue
        seed = int(DEFAULT_FOLD_SEEDS[fold_number - 1])
        for model_name in args.models:
            for modality in args.modalities:
                result = run_fold_model(
                    args,
                    fold_manifest,
                    image_stats,
                    run_root,
                    fold_number,
                    model_name,
                    modality,
                    seed,
                    device,
                )
                results.append(result)
                write_csv(run_root / "results.csv", [flat_result(item) | {"fold": item["fold"]} for item in results])
                summary_rows = summarize(results)
                atomic_write_json(run_root / "summary.json", summary_rows)
                write_csv(run_root / "summary.csv", flatten_summary(summary_rows))
    if args.audit_only:
        print("Grouped 5-fold audit complete; no model or target_test was loaded.")
        return 0
    print(f"Completed runs: {len(results)}")
    print(f"Results: {run_root / 'results.csv'}")
    print(f"Summary: {run_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
