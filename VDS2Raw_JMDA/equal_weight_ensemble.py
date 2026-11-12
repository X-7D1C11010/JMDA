"""Equal-weight same-modality ensemble over saved five-fold predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from VDS2Raw_JMDA.single_modal_baselines import (
    atomic_write_json,
    classification_metrics,
    numeric_summary,
    write_csv,
)
from VDS2Raw_JMDA.vds2raw_dataset import CLASS_NAMES


MODELS = ("resnet", "vit", "convnext")
MODALITIES = ("rgb", "nir")
PROBABILITY_COLUMNS = tuple(
    f"prob_{name.replace(' & ', '_').replace(' ', '_')}" for name in CLASS_NAMES
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--prediction-root", type=Path, required=True)
    result.add_argument("--output-name", default="equal_weight_same_modality_ensemble")
    return result


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aligned_probabilities(
    prediction_root: Path, fold: int, modality: str
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    tables = [
        load_rows(
            prediction_root
            / model_name
            / modality
            / f"fold_{fold}"
            / "target_test_predictions.csv"
        )
        for model_name in MODELS
    ]
    sample_ids = [row["sample_id"] for row in tables[0]]
    labels = np.asarray([int(row["label"]) for row in tables[0]], dtype=np.int64)
    probability_stack = []
    for rows in tables:
        if [row["sample_id"] for row in rows] != sample_ids:
            raise RuntimeError("Backbone predictions are not sample-aligned")
        other_labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        if not np.array_equal(other_labels, labels):
            raise RuntimeError("Backbone labels are not aligned")
        probability_stack.append(
            np.asarray(
                [
                    [float(row[column]) for column in PROBABILITY_COLUMNS]
                    for row in rows
                ],
                dtype=np.float64,
            )
        )
    return sample_ids, labels, np.stack(probability_stack, axis=0)


def main() -> int:
    args = parser().parse_args()
    prediction_root = args.prediction_root.resolve()
    output_root = prediction_root / args.output_name
    fold_results: List[Dict[str, object]] = []
    prediction_rows: List[Dict[str, object]] = []
    for fold in range(1, 6):
        for modality in MODALITIES:
            sample_ids, labels, probability_stack = aligned_probabilities(
                prediction_root, fold, modality
            )
            probabilities = probability_stack.mean(axis=0)
            predictions = probabilities.argmax(axis=1)
            metrics = classification_metrics(labels, predictions)
            fold_results.append(
                {
                    "fold": fold,
                    "modality": modality,
                    "models": list(MODELS),
                    "weights": {model_name: 1.0 / 3.0 for model_name in MODELS},
                    "target_test": metrics,
                }
            )
            for index, sample_id in enumerate(sample_ids):
                row: Dict[str, object] = {
                    "fold": fold,
                    "modality": modality,
                    "sample_id": sample_id,
                    "label": int(labels[index]),
                    "prediction": int(predictions[index]),
                }
                for class_index, column in enumerate(PROBABILITY_COLUMNS):
                    row[column] = float(probabilities[index, class_index])
                prediction_rows.append(row)
            print(
                f"fold={fold} modality={modality} ACC={metrics['acc']:.4f} "
                f"P={metrics['precision_macro']:.4f} "
                f"R={metrics['recall_macro']:.4f} F1={metrics['f1_macro']:.4f}"
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
