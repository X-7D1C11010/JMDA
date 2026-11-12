"""Validation-only class-bias calibration for sample-stratified checkpoints."""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
    MODEL_NAMES,
    SingleModalityDataset,
    atomic_write_json,
    classification_metrics,
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


MODALITIES = ("rgb", "nir")
FOLD_SEEDS = (42, 43, 44, 45, 46)
PROBABILITY_COLUMNS = (
    "prob_Cargo",
    "prob_Fishing",
    "prob_Sailing_Pleasure",
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
    result.add_argument("--output-name", default="validation_calibrated_evaluation")
    result.add_argument("--split-seed", type=int, default=20261110)
    result.add_argument("--max-absolute-bias", type=float, default=0.6)
    result.add_argument("--bias-step", type=float, default=0.1)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def arrays(rows: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probabilities = np.asarray(
        [[float(row[column]) for column in PROBABILITY_COLUMNS] for row in rows],
        dtype=np.float64,
    )
    return labels, probabilities


def calibrated_predictions(probabilities: np.ndarray, bias: Sequence[float]) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-12, 1.0))
    return (logits + np.asarray(bias, dtype=np.float64)).argmax(axis=1)


def fit_bias(
    rows_by_mode: Mapping[str, Sequence[Mapping[str, object]]],
    maximum: float,
    step: float,
) -> Tuple[str, List[float], Dict[str, object]]:
    if maximum < 0 or step <= 0:
        raise ValueError("bias bounds must be non-negative and step must be positive")
    count = int(round(maximum / step))
    values = [index * step for index in range(-count, count + 1)]
    best_key = None
    best = None
    for mode, rows in rows_by_mode.items():
        labels, probabilities = arrays(rows)
        # A common logit offset has no effect, so Cargo is the zero reference.
        for fishing, sailing in itertools.product(values, repeat=2):
            bias = [0.0, fishing, sailing]
            predictions = calibrated_predictions(probabilities, bias)
            metrics = classification_metrics(labels, predictions)
            norm = math.sqrt(fishing * fishing + sailing * sailing)
            key = (
                float(metrics["acc"]),
                float(metrics["f1_macro"]),
                -norm,
                mode == "identity",
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (mode, bias, metrics)
    if best is None:
        raise RuntimeError("No calibration candidate was selected")
    return best


def apply_calibration(
    rows: Sequence[Mapping[str, object]], bias: Sequence[float]
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    labels, probabilities = arrays(rows)
    predictions = calibrated_predictions(probabilities, bias)
    metrics = classification_metrics(labels, predictions)
    output: List[Dict[str, object]] = []
    for row, prediction in zip(rows, predictions.tolist()):
        output.append({**dict(row), "prediction": int(prediction)})
    return metrics, output


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
        for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
            values = [float(result["target_test"][metric]) for result in group]
            for statistic, value in numeric_summary(values).items():
                row[f"test_{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def main() -> int:
    args = parser().parse_args()
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    base_run = args.base_run.expanduser().resolve()
    output_root = base_run / args.output_name
    output_root.mkdir(parents=True, exist_ok=True)
    base = VDS2RawManifest(args.dataset_root.expanduser().resolve())
    folds = build_sample_stratified_folds(base, args.split_seed)
    results: List[Dict[str, object]] = []

    for fold_data in folds:
        fold = int(fold_data["fold"])
        seed = FOLD_SEEDS[fold - 1]
        manifest = FoldManifest(base, fold_data["assignments"])
        image_stats = compute_train_image_stats(manifest)
        for model_name in MODEL_NAMES:
            for modality in MODALITIES:
                seed_everything(seed)
                run_dir = output_root / model_name / modality / f"fold_{fold}"
                result_path = run_dir / "result.json"
                if result_path.is_file():
                    existing = json.loads(result_path.read_text(encoding="utf-8"))
                    if existing.get("status") == "complete":
                        results.append(existing)
                        print(f"Skipping completed {model_name}/{modality}/fold_{fold}")
                        continue

                checkpoint_path = (
                    base_run / model_name / modality / f"fold_{fold}" / "best.pt"
                )
                model = create_model(
                    model_name,
                    modality,
                    len(CLASS_NAMES),
                    image_size=224,
                    backbone_profile="baseline",
                    pretrained=False,
                ).to(device)
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                channels_last = model_name in {"resnet", "convnext"} and device.type == "cuda"
                if channels_last:
                    model = model.to(memory_format=torch.channels_last)

                validation = SingleModalityDataset(
                    manifest, "target_val", image_stats, modality, False, 224, 224
                )
                val_loader = make_loader(
                    validation,
                    args.batch_size,
                    False,
                    args.num_workers,
                    device,
                    seed,
                )
                validation_rows: Dict[str, List[Dict[str, object]]] = {}
                for mode in ("identity", "horizontal_flip_mean"):
                    _, rows = evaluate_prediction_mode(
                        model, val_loader, device, use_amp, channels_last, mode
                    )
                    validation_rows[mode] = rows
                mode, bias, calibrated_validation = fit_bias(
                    validation_rows, args.max_absolute_bias, args.bias_step
                )

                # target_test is instantiated only after mode and class bias
                # have been selected entirely on target_val.
                test = SingleModalityDataset(
                    manifest, "target_test", image_stats, modality, False, 224, 224
                )
                test_loader = make_loader(
                    test,
                    args.batch_size,
                    False,
                    args.num_workers,
                    device,
                    seed,
                )
                _, raw_test_rows = evaluate_prediction_mode(
                    model, test_loader, device, use_amp, channels_last, mode
                )
                test_metrics, calibrated_rows = apply_calibration(raw_test_rows, bias)
                result: Dict[str, object] = {
                    "status": "complete",
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "modality": modality,
                    "checkpoint": str(checkpoint_path),
                    "selection": "prediction mode and class bias selected on target_val only",
                    "prediction_mode": mode,
                    "class_log_probability_bias": bias,
                    "validation": calibrated_validation,
                    "target_test": test_metrics,
                }
                atomic_write_json(result_path, result)
                write_csv(run_dir / "target_test_predictions.csv", calibrated_rows)
                results.append(result)
                print(
                    f"fold={fold} model={model_name} modality={modality} mode={mode} "
                    f"bias={bias} val={calibrated_validation['acc']:.4f} "
                    f"test={test_metrics['acc']:.4f} f1={test_metrics['f1_macro']:.4f}"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary_rows = summarize(results)
    atomic_write_json(output_root / "results.json", results)
    atomic_write_json(output_root / "summary.json", summary_rows)
    write_csv(output_root / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
