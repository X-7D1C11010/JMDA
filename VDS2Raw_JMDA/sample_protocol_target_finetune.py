"""Target-only fine-tuning for the sample-stratified VDS2Raw protocol.

The script is intended for validation-only pilots and a later locked five-fold
run.  It starts from ImageNet weights or an existing validation-selected
checkpoint, trains only on target_train, and selects checkpoints using
target_val.  target_test is not instantiated unless --evaluate-test is set.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from VDS2Raw_JMDA.grouped_cv_baselines import (
    FoldManifest,
    build_sample_stratified_folds,
    evaluate_prediction_mode,
    set_last_stage_trainable,
    train_epoch_partial,
)
from VDS2Raw_JMDA.single_modal_baselines import (
    MODEL_NAMES,
    SingleModalityDataset,
    atomic_write_json,
    checkpoint_key,
    classification_head,
    compact_metrics,
    create_model,
    image_optimizer_groups,
    make_loader,
    numeric_summary,
    parse_csv_values,
    seed_everything,
    utc_free_timestamp,
    write_csv,
)
from VDS2Raw_JMDA.vds2raw_dataset import (
    CLASS_NAMES,
    VDS2RawManifest,
    compute_train_image_stats,
)


MODALITIES = ("rgb", "nir")
FOLD_SEEDS = (42, 43, 44, 45, 46)


class ImageTransformDataset(Dataset):
    """Apply deterministic radiometric normalization and centered zoom."""

    def __init__(
        self,
        base: Dataset,
        ratio: float,
        normalization: str,
        modality: str,
        fitted_mean: Sequence[float],
        fitted_std: Sequence[float],
    ):
        if not 0 < ratio <= 1:
            raise ValueError("center zoom ratio must be in (0, 1]")
        if normalization not in {"dataset", "per-image", "imagenet-255"}:
            raise ValueError(f"unknown normalization: {normalization}")
        self.base = base
        self.ratio = float(ratio)
        self.normalization = normalization
        self.modality = modality
        self.fitted_mean = torch.as_tensor(fitted_mean, dtype=torch.float32)[:, None, None]
        self.fitted_std = torch.as_tensor(fitted_std, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, label, sample_id = self.base[index]
        if self.normalization == "per-image":
            valid = torch.any(image != 0, dim=0)
            if bool(valid.any()):
                normalized = image.clone()
                for channel in range(image.shape[0]):
                    values = image[channel][valid]
                    mean = values.mean()
                    std = values.std(unbiased=False).clamp_min(1e-6)
                    normalized[channel][valid] = (values - mean) / std
                normalized[:, ~valid] = 0
                image = normalized
        elif self.normalization == "imagenet-255":
            valid = torch.any(image != 0, dim=0)
            raw = image * self.fitted_std + self.fitted_mean
            scaled = raw / 255.0
            if self.modality == "rgb":
                mean = torch.tensor((0.485, 0.456, 0.406), dtype=image.dtype)[:, None, None]
                std = torch.tensor((0.229, 0.224, 0.225), dtype=image.dtype)[:, None, None]
            else:
                mean = torch.tensor((0.449,), dtype=image.dtype)[:, None, None]
                std = torch.tensor((0.226,), dtype=image.dtype)[:, None, None]
            image = (scaled - mean) / std
            image[:, ~valid] = 0
        if self.ratio < 1:
            height, width = image.shape[-2:]
            crop_height = max(1, int(round(height * self.ratio)))
            crop_width = max(1, int(round(width * self.ratio)))
            top = (height - crop_height) // 2
            left = (width - crop_width) // 2
            image = image[:, top : top + crop_height, left : left + crop_width]
            image = F.interpolate(
                image.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
        return image.contiguous(), label, sample_id


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Downloads\VDS2Raw\threeclass_ready"),
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "single_modal_runs",
    )
    result.add_argument("--run-name", required=True)
    result.add_argument(
        "--base-run",
        type=Path,
        default=Path(__file__).resolve().parent
        / "single_modal_runs"
        / "vds2raw_sample_stratified_fixedtta_30runs_20261110",
    )
    result.add_argument(
        "--models",
        type=lambda value: parse_csv_values(value, MODEL_NAMES, "models"),
        default=list(MODEL_NAMES),
    )
    result.add_argument(
        "--modalities",
        type=lambda value: parse_csv_values(value, MODALITIES, "modalities"),
        default=list(MODALITIES),
    )
    result.add_argument("--folds", default="1,2,3,4,5")
    result.add_argument("--split-seed", type=int, default=20261110)
    result.add_argument(
        "--split-protocol",
        choices=("target-fivefold", "all-sample-repeated-holdout"),
        default="target-fivefold",
    )
    result.add_argument(
        "--initialization",
        choices=("imagenet", "base-checkpoint"),
        default="imagenet",
    )
    result.add_argument(
        "--unfreeze-scope", choices=("last-stage", "all"), default="all"
    )
    result.add_argument("--epochs", type=int, default=25)
    result.add_argument("--head-only-epochs", type=int, default=3)
    result.add_argument("--backbone-lr", type=float, default=1e-5)
    result.add_argument("--head-lr", type=float, default=2e-4)
    result.add_argument("--weight-decay", type=float, default=1e-2)
    result.add_argument("--label-smoothing", type=float, default=0.0)
    result.add_argument("--center-zoom-ratio", type=float, default=1.0)
    result.add_argument(
        "--input-normalization",
        choices=("dataset", "per-image", "imagenet-255"),
        default="dataset",
    )
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--grad-clip", type=float, default=5.0)
    result.add_argument(
        "--prediction-mode",
        choices=("identity", "horizontal_flip_mean"),
        default="identity",
    )
    result.add_argument(
        "--evaluate-test", action=argparse.BooleanOptionalAction, default=False
    )
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def parse_folds(value: str) -> List[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError("folds must be a comma-separated subset of 1,2,3,4,5")
    return list(dict.fromkeys(folds))


def build_all_sample_repeated_holdouts(
    base: VDS2RawManifest,
) -> List[Dict[str, object]]:
    """Build five 70/15/15 stratified splits over all 271 samples.

    These are repeated holdouts rather than disjoint CV folds.  The split is
    sample-level and therefore deliberately does not enforce scene/MMSI
    separation.
    """

    indices = list(range(len(base.records)))
    labels = [record.label_id for record in base.records]
    output: List[Dict[str, object]] = []
    for run_index, seed in enumerate(FOLD_SEEDS, start=1):
        train_val, test = train_test_split(
            indices,
            test_size=0.15,
            random_state=seed,
            stratify=labels,
        )
        train_val_labels = [labels[index] for index in train_val]
        train, validation = train_test_split(
            train_val,
            test_size=0.15 / 0.85,
            random_state=seed + 1000,
            stratify=train_val_labels,
        )
        assignments = {
            base.records[index].sample_id: "target_train" for index in train
        }
        assignments.update(
            {base.records[index].sample_id: "target_val" for index in validation}
        )
        assignments.update(
            {base.records[index].sample_id: "target_test" for index in test}
        )
        if len(assignments) != len(base.records):
            raise RuntimeError("Repeated holdout did not assign every sample")
        output.append({"fold": run_index, "seed": seed, "assignments": assignments})
    return output


def summarize(results: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for result in results:
        grouped[(result["model"], result["modality"])].append(result)
    rows: List[Dict[str, object]] = []
    for (model_name, modality), group in sorted(grouped.items()):
        row: Dict[str, object] = {
            "model": model_name,
            "modality": modality,
            "runs": len(group),
        }
        for scope, key in (("val", "validation"), ("test", "target_test")):
            available = [item[key] for item in group if item.get(key)]
            if not available:
                continue
            for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
                values = [float(metrics[metric]) for metrics in available]
                for statistic, value in numeric_summary(values).items():
                    row[f"{scope}_{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def main() -> int:
    args = parser().parse_args()
    folds = parse_folds(args.folds)
    if args.epochs < 1 or not 0 <= args.head_only_epochs < args.epochs:
        raise ValueError("epochs must be positive and head-only-epochs must be smaller")
    if args.backbone_lr <= 0 or args.head_lr <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer arguments")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("label-smoothing must be in [0, 1)")
    if not 0 < args.center_zoom_ratio <= 1:
        raise ValueError("center-zoom-ratio must be in (0, 1]")

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    output_root = args.output_root.expanduser().resolve() / args.run_name
    output_root.mkdir(parents=True, exist_ok=True)
    base_run = args.base_run.expanduser().resolve()
    base = VDS2RawManifest(args.dataset_root.expanduser().resolve())
    fold_specs = (
        build_sample_stratified_folds(base, args.split_seed)
        if args.split_protocol == "target-fivefold"
        else build_all_sample_repeated_holdouts(base)
    )
    results: List[Dict[str, object]] = []

    protocol = {
        "split": (
            "target-only sample-stratified five-fold with known scene/MMSI overlap"
            if args.split_protocol == "target-fivefold"
            else "all-271-sample stratified repeated 70/15/15 holdout with known scene/MMSI overlap"
        ),
        "training_partition": "target_train only",
        "selection_partition": "target_val only",
        "test_enabled": args.evaluate_test,
        "initialization": args.initialization,
        "unfreeze_scope": args.unfreeze_scope,
        "epochs": args.epochs,
        "head_only_epochs": args.head_only_epochs,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "center_zoom_ratio": args.center_zoom_ratio,
        "input_normalization": args.input_normalization,
        "prediction_mode": args.prediction_mode,
    }
    atomic_write_json(output_root / "protocol.json", protocol)

    for fold_data in fold_specs:
        fold = int(fold_data["fold"])
        if fold not in folds:
            continue
        seed = int(fold_data.get("seed", FOLD_SEEDS[fold - 1]))
        manifest = FoldManifest(base, fold_data["assignments"])
        image_stats = compute_train_image_stats(manifest)
        for model_name in args.models:
            for modality in args.modalities:
                run_dir = output_root / model_name / modality / f"fold_{fold}"
                result_path = run_dir / "result.json"
                if result_path.is_file():
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "complete":
                        results.append(existing)
                        print(f"Skipping completed {model_name}/{modality}/fold_{fold}")
                        continue

                seed_everything(seed)
                started = time.perf_counter()
                model = create_model(
                    model_name,
                    modality,
                    len(CLASS_NAMES),
                    image_size=224,
                    backbone_profile="baseline",
                    pretrained=args.initialization == "imagenet",
                ).to(device)
                initial_checkpoint: Optional[str] = None
                if args.initialization == "base-checkpoint":
                    checkpoint_path = (
                        base_run / model_name / modality / f"fold_{fold}" / "best.pt"
                    )
                    checkpoint = torch.load(
                        checkpoint_path, map_location=device, weights_only=False
                    )
                    model.load_state_dict(checkpoint["model_state"])
                    initial_checkpoint = str(checkpoint_path)

                channels_last = model_name in {"resnet", "convnext"} and device.type == "cuda"
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)
                for parameter in model.parameters():
                    parameter.requires_grad = False
                for parameter in classification_head(model, model_name).parameters():
                    parameter.requires_grad = True

                training_base = SingleModalityDataset(
                    manifest, "target_train", image_stats, modality, True, 224, 224
                )
                validation_base = SingleModalityDataset(
                    manifest, "target_val", image_stats, modality, False, 224, 224
                )
                training = ImageTransformDataset(
                    training_base,
                    args.center_zoom_ratio,
                    args.input_normalization,
                    modality,
                    image_stats["mean"][:3] if modality == "rgb" else image_stats["mean"][3:],
                    image_stats["std"][:3] if modality == "rgb" else image_stats["std"][3:],
                )
                validation = ImageTransformDataset(
                    validation_base,
                    args.center_zoom_ratio,
                    args.input_normalization,
                    modality,
                    image_stats["mean"][:3] if modality == "rgb" else image_stats["mean"][3:],
                    image_stats["std"][:3] if modality == "rgb" else image_stats["std"][3:],
                )
                train_loader = make_loader(
                    training, args.batch_size, True, args.num_workers, device, seed
                )
                val_loader = make_loader(
                    validation, args.batch_size, False, args.num_workers, device, seed
                )

                optimizer = torch.optim.AdamW(
                    image_optimizer_groups(
                        model, model_name, args.backbone_lr, args.head_lr
                    ),
                    weight_decay=args.weight_decay,
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda epoch: 0.5
                    * (1.0 + math.cos(math.pi * epoch / args.epochs)),
                )
                criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                best_key = None
                best_metrics = None
                best_epoch = 0
                checkpoint_path = run_dir / "best.pt"
                epoch_rows: List[Dict[str, object]] = []

                # Epoch zero is a valid candidate, especially when continuing
                # from a validation-selected base checkpoint.  Keeping it in
                # the comparison prevents target-only adaptation from silently
                # replacing a better starting point.
                initial_val_metrics, _ = evaluate_prediction_mode(
                    model,
                    val_loader,
                    device,
                    use_amp,
                    channels_last,
                    args.prediction_mode,
                )
                best_key = checkpoint_key(initial_val_metrics, "acc")
                best_metrics = dict(initial_val_metrics)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "validation_metrics": initial_val_metrics,
                        "epoch": 0,
                        "protocol": protocol,
                    },
                    checkpoint_path,
                )
                epoch_rows.append(
                    {
                        "epoch": 0,
                        "train_loss": "",
                        "train_acc": "",
                        **compact_metrics(initial_val_metrics, "val_"),
                    }
                )
                write_csv(run_dir / "epochs.csv", epoch_rows)
                print(
                    f"fold={fold} model={model_name} modality={modality} "
                    f"epoch=0 val={initial_val_metrics['acc']:.4f} "
                    f"f1={initial_val_metrics['f1_macro']:.4f} best"
                )

                for epoch in range(1, args.epochs + 1):
                    if epoch == args.head_only_epochs + 1:
                        if args.unfreeze_scope == "all":
                            for parameter in model.parameters():
                                parameter.requires_grad = True
                        else:
                            set_last_stage_trainable(model, model_name)
                    train_metrics = train_epoch_partial(
                        model,
                        model_name,
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        args.grad_clip,
                        scaler,
                        use_amp,
                        channels_last,
                    )
                    val_metrics, _ = evaluate_prediction_mode(
                        model,
                        val_loader,
                        device,
                        use_amp,
                        channels_last,
                        args.prediction_mode,
                    )
                    key = checkpoint_key(val_metrics, "acc")
                    improved = best_key is None or key > best_key
                    if improved:
                        best_key = key
                        best_metrics = dict(val_metrics)
                        best_epoch = epoch
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "model_state": model.state_dict(),
                                "validation_metrics": val_metrics,
                                "epoch": epoch,
                                "protocol": protocol,
                            },
                            checkpoint_path,
                        )
                    epoch_rows.append(
                        {
                            "epoch": epoch,
                            "train_loss": train_metrics["loss"],
                            "train_acc": train_metrics["acc"],
                            **compact_metrics(val_metrics, "val_"),
                        }
                    )
                    write_csv(run_dir / "epochs.csv", epoch_rows)
                    print(
                        f"fold={fold} model={model_name} modality={modality} "
                        f"epoch={epoch} train={train_metrics['acc']:.4f} "
                        f"val={val_metrics['acc']:.4f} f1={val_metrics['f1_macro']:.4f}"
                        + (" best" if improved else "")
                    )
                    scheduler.step()

                if best_metrics is None:
                    raise RuntimeError("No checkpoint was selected")
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                test_metrics = None
                if args.evaluate_test:
                    test_base = SingleModalityDataset(
                        manifest, "target_test", image_stats, modality, False, 224, 224
                    )
                    test = ImageTransformDataset(
                        test_base,
                        args.center_zoom_ratio,
                        args.input_normalization,
                        modality,
                        image_stats["mean"][:3] if modality == "rgb" else image_stats["mean"][3:],
                        image_stats["std"][:3] if modality == "rgb" else image_stats["std"][3:],
                    )
                    test_loader = make_loader(
                        test,
                        args.batch_size,
                        False,
                        args.num_workers,
                        device,
                        seed,
                    )
                    test_metrics, prediction_rows = evaluate_prediction_mode(
                        model,
                        test_loader,
                        device,
                        use_amp,
                        channels_last,
                        args.prediction_mode,
                    )
                    write_csv(run_dir / "target_test_predictions.csv", prediction_rows)

                result: Dict[str, object] = {
                    "status": "complete",
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "modality": modality,
                    "initial_checkpoint": initial_checkpoint,
                    "best_epoch": best_epoch,
                    "validation": best_metrics,
                    "target_test": test_metrics,
                    "duration_seconds": time.perf_counter() - started,
                    "completed_at": utc_free_timestamp(),
                }
                atomic_write_json(result_path, result)
                results.append(result)
                del model, optimizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_rows = summarize(results)
    atomic_write_json(output_root / "results.json", results)
    atomic_write_json(output_root / "summary.json", summary_rows)
    write_csv(output_root / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
