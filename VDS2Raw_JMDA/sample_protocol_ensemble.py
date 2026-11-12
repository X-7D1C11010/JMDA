"""Validation-selected same-modality ensemble for sample-stratified CV.

The three pretrained backbones are combined only within RGB or only within
NIR.  Per-fold weights are selected on target_val before that fold's saved
target_test probabilities are read.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from VDS2Raw_JMDA.grouped_cv_baselines import (
    FoldManifest,
    build_sample_stratified_folds,
    evaluate_prediction_mode,
)
from VDS2Raw_JMDA.single_modal_baselines import (
    SingleModalityDataset,
    atomic_write_json,
    classification_metrics,
    compact_metrics,
    create_model,
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


MODELS = ("resnet", "vit", "convnext")
MODALITIES = ("rgb", "nir")
PROBABILITY_COLUMNS = tuple(
    f"prob_{name.replace(' & ', '_').replace(' ', '_')}" for name in CLASS_NAMES
)


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
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def probability_table(rows: Sequence[Mapping[str, object]]) -> Tuple[List[str], np.ndarray, np.ndarray]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[column]) for column in PROBABILITY_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    return sample_ids, labels, probabilities


def candidate_weights() -> List[Tuple[float, float, float]]:
    candidates = [
        tuple(value / 10.0 for value in values)
        for values in itertools.product(range(11), repeat=3)
        if sum(values) == 10
    ]
    candidates.append((1.0 / 3.0,) * 3)
    return candidates


def select_weights(
    labels: np.ndarray, probability_stack: np.ndarray
) -> Tuple[Tuple[float, float, float], Dict[str, object]]:
    best_key = None
    best_weights = None
    best_metrics = None
    equal = np.asarray((1.0 / 3.0,) * 3)
    for weights in candidate_weights():
        probabilities = np.average(
            probability_stack, axis=0, weights=np.asarray(weights)
        )
        metrics = classification_metrics(labels, probabilities.argmax(axis=1))
        distance = float(np.square(np.asarray(weights) - equal).sum())
        key = float(metrics["acc"]), float(metrics["f1_macro"]), -distance
        if best_key is None or key > best_key:
            best_key = key
            best_weights = weights
            best_metrics = metrics
    if best_weights is None or best_metrics is None:
        raise RuntimeError("No ensemble candidate was selected")
    return best_weights, best_metrics


def read_test_probabilities(
    base_run: Path, fold: int, modality: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    tables = []
    for model_name in MODELS:
        path = (
            base_run
            / model_name
            / modality
            / f"fold_{fold}"
            / "target_test_predictions.csv"
        )
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            tables.append(list(csv.DictReader(handle)))
    ids, labels, first = probability_table(tables[0])
    stack = [first]
    for rows in tables[1:]:
        other_ids, other_labels, probabilities = probability_table(rows)
        if other_ids != ids or not np.array_equal(other_labels, labels):
            raise RuntimeError("Backbone test predictions are not sample-aligned")
        stack.append(probabilities)
    return ids, labels, np.stack(stack, axis=0)


def main() -> int:
    args = parser().parse_args()
    device = torch.device(args.device)
    base_run = args.base_run.resolve()
    output_root = base_run / "same_modality_ensemble"
    base = VDS2RawManifest(args.dataset_root.resolve())
    folds = build_sample_stratified_folds(base, args.split_seed)
    fold_results: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for fold_data in folds:
        fold = int(fold_data["fold"])
        seed_everything(41 + fold)
        manifest = FoldManifest(base, fold_data["assignments"])
        image_stats = compute_train_image_stats(manifest)
        for modality in MODALITIES:
            validation_rows_by_model = []
            for model_name in MODELS:
                model = create_model(
                    model_name,
                    modality,
                    len(CLASS_NAMES),
                    image_size=224,
                    backbone_profile="baseline",
                    pretrained=True,
                ).to(device)
                channels_last = model_name in {"resnet", "convnext"} and device.type == "cuda"
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)
                checkpoint_path = (
                    base_run / model_name / modality / f"fold_{fold}" / "best.pt"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                validation = SingleModalityDataset(
                    manifest, "target_val", image_stats, modality, False, 224, 224
                )
                loader = make_loader(
                    validation,
                    args.batch_size,
                    False,
                    args.num_workers,
                    device,
                    41 + fold,
                )
                _, rows = evaluate_prediction_mode(
                    model,
                    loader,
                    device,
                    device.type == "cuda",
                    channels_last,
                    "horizontal_flip_mean",
                )
                validation_rows_by_model.append(rows)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            val_ids, val_labels, first_val = probability_table(
                validation_rows_by_model[0]
            )
            val_stack = [first_val]
            for rows in validation_rows_by_model[1:]:
                ids, labels, probabilities = probability_table(rows)
                if ids != val_ids or not np.array_equal(labels, val_labels):
                    raise RuntimeError("Backbone validation predictions are not aligned")
                val_stack.append(probabilities)
            weights, validation_metrics = select_weights(
                val_labels, np.stack(val_stack, axis=0)
            )

            # Target-test files are opened only after target_val fixes weights.
            test_ids, test_labels, test_stack = read_test_probabilities(
                base_run, fold, modality
            )
            test_probabilities = np.average(
                test_stack, axis=0, weights=np.asarray(weights)
            )
            test_predictions = test_probabilities.argmax(axis=1)
            test_metrics = classification_metrics(test_labels, test_predictions)
            fold_result: Dict[str, object] = {
                "fold": fold,
                "modality": modality,
                "models": list(MODELS),
                "weights": dict(zip(MODELS, weights)),
                "selection": "target_val ACC; macro-F1 tie break; equal-weight proximity tie break",
                "validation": validation_metrics,
                "target_test": test_metrics,
            }
            fold_results.append(fold_result)
            for index, sample_id in enumerate(test_ids):
                row: Dict[str, object] = {
                    "fold": fold,
                    "modality": modality,
                    "sample_id": sample_id,
                    "label": int(test_labels[index]),
                    "prediction": int(test_predictions[index]),
                }
                for class_index, column in enumerate(PROBABILITY_COLUMNS):
                    row[column] = float(test_probabilities[index, class_index])
                prediction_rows.append(row)
            print(
                f"fold={fold} modality={modality} weights={weights} "
                f"val_acc={validation_metrics['acc']:.4f} "
                f"test_acc={test_metrics['acc']:.4f} test_f1={test_metrics['f1_macro']:.4f}"
            )

    summary_rows: List[Dict[str, object]] = []
    for modality in MODALITIES:
        group = [row for row in fold_results if row["modality"] == modality]
        summary: Dict[str, object] = {"modality": modality, "runs": len(group)}
        for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
            values = [float(row["target_test"][metric]) for row in group]
            for statistic, value in numeric_summary(values).items():
                summary[f"test_{metric}_{statistic}"] = value
        summary_rows.append(summary)
    atomic_write_json(output_root / "fold_results.json", fold_results)
    atomic_write_json(output_root / "summary.json", summary_rows)
    write_csv(output_root / "summary.csv", summary_rows)
    write_csv(output_root / "target_test_predictions.csv", prediction_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
