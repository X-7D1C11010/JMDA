"""Fixed target_train+target_val head refit for sample-stratified CV."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from VDS2Raw_JMDA.grouped_cv_baselines import (
    FoldManifest,
    build_sample_stratified_folds,
    evaluate_prediction_mode,
    train_epoch_partial,
)
from VDS2Raw_JMDA.single_modal_baselines import (
    MODEL_NAMES,
    SingleModalityDataset,
    atomic_write_json,
    classification_head,
    compact_metrics,
    create_model,
    image_optimizer_groups,
    make_loader,
    numeric_summary,
    seed_everything,
    write_csv,
)
from VDS2Raw_JMDA.vds2raw_dataset import (
    CLASS_NAMES,
    VDS2RawManifest,
    compute_train_image_stats,
)


MODALITIES = ("rgb", "nir")
FOLD_SEEDS = (42, 43, 44, 45, 46)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Downloads\VDS2Raw\threeclass_ready"),
    )
    result.add_argument(
        "--base-run",
        type=Path,
        default=Path(__file__).resolve().parent
        / "single_modal_runs"
        / "vds2raw_sample_stratified_fixedtta_30runs_20261110",
    )
    result.add_argument("--split-seed", type=int, default=20261110)
    result.add_argument("--refit-epochs", type=int, default=3)
    result.add_argument("--head-lr", type=float, default=5e-5)
    result.add_argument("--weight-decay", type=float, default=1e-2)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def expanded_training_manifest(
    base: VDS2RawManifest, fold_manifest: FoldManifest
) -> FoldManifest:
    assignments = {
        record.sample_id: (
            "target_train" if record.partition == "target_val" else record.partition
        )
        for record in fold_manifest.records
    }
    return FoldManifest(base, assignments)


def main() -> int:
    args = parser().parse_args()
    if args.refit_epochs < 1 or args.head_lr <= 0:
        raise ValueError("refit_epochs and head_lr must be positive")
    device = torch.device(args.device)
    base_run = args.base_run.resolve()
    output_root = base_run / f"refit_head{args.refit_epochs}_lr{args.head_lr:g}"
    base = VDS2RawManifest(args.dataset_root.resolve())
    folds = build_sample_stratified_folds(base, args.split_seed)
    results: List[Dict[str, object]] = []
    for fold_data in folds:
        fold = int(fold_data["fold"])
        seed = FOLD_SEEDS[fold - 1]
        fold_manifest = FoldManifest(base, fold_data["assignments"])
        image_stats = compute_train_image_stats(fold_manifest)
        refit_manifest = expanded_training_manifest(base, fold_manifest)
        if len(refit_manifest.by_partition["target_train"]) != 124:
            raise RuntimeError("Refit must contain 124 non-test target samples")
        for model_name in MODEL_NAMES:
            for modality in MODALITIES:
                seed_everything(seed)
                run_dir = output_root / model_name / modality / f"fold_{fold}"
                result_path = run_dir / "result.json"
                model = create_model(
                    model_name,
                    modality,
                    len(CLASS_NAMES),
                    image_size=224,
                    backbone_profile="baseline",
                    pretrained=True,
                ).to(device)
                checkpoint_path = (
                    base_run / model_name / modality / f"fold_{fold}" / "best.pt"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                channels_last = model_name in {"resnet", "convnext"} and device.type == "cuda"
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)
                for parameter in model.parameters():
                    parameter.requires_grad = False
                for parameter in classification_head(model, model_name).parameters():
                    parameter.requires_grad = True
                optimizer = torch.optim.AdamW(
                    image_optimizer_groups(model, model_name, 0.0, args.head_lr),
                    weight_decay=args.weight_decay,
                )
                criterion = nn.CrossEntropyLoss()
                scaler = torch.amp.GradScaler("cuda", enabled=False)
                training = SingleModalityDataset(
                    refit_manifest,
                    "target_train",
                    image_stats,
                    modality,
                    True,
                    224,
                    224,
                )
                train_loader = make_loader(
                    training,
                    args.batch_size,
                    True,
                    args.num_workers,
                    device,
                    seed,
                )
                epoch_rows: List[Dict[str, object]] = []
                for epoch in range(1, args.refit_epochs + 1):
                    metrics = train_epoch_partial(
                        model,
                        model_name,
                        train_loader,
                        optimizer,
                        criterion,
                        device,
                        5.0,
                        scaler,
                        device.type == "cuda",
                        channels_last,
                    )
                    epoch_rows.append({"epoch": epoch, **metrics})

                # target_test is instantiated only after the fixed refit ends.
                test_dataset = SingleModalityDataset(
                    fold_manifest,
                    "target_test",
                    image_stats,
                    modality,
                    False,
                    224,
                    224,
                )
                test_loader = make_loader(
                    test_dataset,
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
                    device.type == "cuda",
                    channels_last,
                    "horizontal_flip_mean",
                )
                result: Dict[str, object] = {
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "modality": modality,
                    "initial_checkpoint": str(checkpoint_path),
                    "refit_partitions": ["target_train", "target_val"],
                    "refit_sample_count": len(training),
                    "refit_epochs": args.refit_epochs,
                    "head_lr": args.head_lr,
                    "target_test": test_metrics,
                }
                atomic_write_json(result_path, result)
                write_csv(run_dir / "refit_epochs.csv", epoch_rows)
                write_csv(run_dir / "target_test_predictions.csv", prediction_rows)
                results.append(result)
                print(
                    f"fold={fold} model={model_name} modality={modality} "
                    f"ACC={test_metrics['acc']:.4f} F1={test_metrics['f1_macro']:.4f}"
                )
                del model, optimizer
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_rows: List[Dict[str, object]] = []
    grouped: Dict[tuple, List[Mapping[str, object]]] = defaultdict(list)
    for result in results:
        grouped[(result["model"], result["modality"])].append(result)
    for (model_name, modality), group in sorted(grouped.items()):
        row: Dict[str, object] = {
            "model": model_name,
            "modality": modality,
            "runs": len(group),
        }
        for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
            values = [float(result["target_test"][metric]) for result in group]
            for statistic, value in numeric_summary(values).items():
                row[f"test_{metric}_{statistic}"] = value
        summary_rows.append(row)
    atomic_write_json(output_root / "results.json", results)
    atomic_write_json(output_root / "summary.json", summary_rows)
    write_csv(output_root / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
