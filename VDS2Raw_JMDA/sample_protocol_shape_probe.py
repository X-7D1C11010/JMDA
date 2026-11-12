"""RGB/NIR-only HOG and morphology probe for sample-stratified VDS2Raw."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from VDS2Raw_JMDA.grouped_cv_baselines import FoldManifest, build_sample_stratified_folds
from VDS2Raw_JMDA.single_modal_baselines import (
    atomic_write_json,
    classification_metrics,
    numeric_summary,
    write_csv,
)
from VDS2Raw_JMDA.vds2raw_dataset import VDS2RawManifest


MODALITIES = ("rgb", "nir")
HOG = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)


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
        default=Path(__file__).resolve().parent
        / "single_modal_runs"
        / "sample_stratified_shape_probe_20261110",
    )
    result.add_argument("--split-seed", type=int, default=20261110)
    return result


def robust_channel(channel: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = channel[valid]
    low, high = np.percentile(values, (2.0, 98.0))
    scaled = np.clip((channel - low) / max(high - low, 1e-6), 0.0, 1.0)
    scaled[~valid] = 0.0
    return scaled.astype(np.float32)


def morphology(gray: np.ndarray) -> np.ndarray:
    output: List[float] = []
    height, width = gray.shape
    yy, xx = np.mgrid[:height, :width]
    center_distance = np.sqrt(
        ((yy - (height - 1) / 2) / height) ** 2
        + ((xx - (width - 1) / 2) / width) ** 2
    )
    output.extend(
        [
            float(gray.mean()),
            float(gray.std()),
            float(gray.max()),
            *[float(value) for value in np.percentile(gray, (5, 25, 50, 75, 90, 95, 98, 99))],
        ]
    )
    for percentile in (85, 90, 93, 95, 97):
        mask = (gray >= np.percentile(gray, percentile)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            output.extend([0.0] * 9)
            continue
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[component, cv2.CC_STAT_AREA])
        component_mask = labels == component
        component_y, component_x = np.nonzero(component_mask)
        covariance = np.cov(np.stack([component_x, component_y]), bias=True)
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        major = float(np.sqrt(max(eigenvalues[0], 0.0)))
        minor = float(np.sqrt(max(eigenvalues[-1], 0.0)))
        output.extend(
            [
                area / (height * width),
                float(stats[component, cv2.CC_STAT_WIDTH]) / width,
                float(stats[component, cv2.CC_STAT_HEIGHT]) / height,
                major / max(height, width),
                minor / max(height, width),
                major / max(minor, 1e-6),
                float(centroids[component, 0] / width),
                float(centroids[component, 1] / height),
                float(center_distance[component_mask].mean()),
            ]
        )
    return np.asarray(output, dtype=np.float32)


def image_features(payload: Mapping[str, np.ndarray], modality: str) -> np.ndarray:
    image = payload[modality].astype(np.float32)
    valid = np.any(image != 0, axis=0)
    channels = np.stack([robust_channel(channel, valid) for channel in image])
    if modality == "rgb":
        gray = 0.299 * channels[0] + 0.587 * channels[1] + 0.114 * channels[2]
    else:
        gray = channels[0]
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog = HOG.compute((resized * 255).astype(np.uint8)).reshape(-1)
    coarse = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).reshape(-1)
    return np.concatenate([hog.astype(np.float32), coarse.astype(np.float32), morphology(gray)])


def partition_arrays(
    manifest: FoldManifest, partition: str, modality: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    features: List[np.ndarray] = []
    labels: List[int] = []
    sample_ids: List[str] = []
    for record in manifest.by_partition[partition]:
        with np.load(manifest.resolve_npz(record), allow_pickle=False) as payload:
            features.append(image_features(payload, modality))
        labels.append(record.label_id)
        sample_ids.append(record.sample_id)
    return np.stack(features), np.asarray(labels, dtype=np.int64), sample_ids


def candidates() -> List[Dict[str, object]]:
    return [
        {"scope": scope, "kernel": kernel, "class_weight": class_weight, "C": c}
        for scope in ("target", "source_target")
        for kernel in ("linear", "rbf")
        for class_weight in (None, "balanced")
        for c in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]


def fit(spec: Mapping[str, object], arrays: Mapping[str, Tuple[np.ndarray, np.ndarray, List[str]]]):
    if spec["scope"] == "target":
        features, labels, _ = arrays["target_train"]
    else:
        features = np.concatenate(
            [arrays["source_train"][0], arrays["target_train"][0]], axis=0
        )
        labels = np.concatenate(
            [arrays["source_train"][1], arrays["target_train"][1]], axis=0
        )
    classifier = make_pipeline(
        StandardScaler(),
        SVC(
            C=float(spec["C"]),
            kernel=str(spec["kernel"]),
            gamma="scale",
            class_weight=spec["class_weight"],
            probability=True,
            random_state=0,
        ),
    )
    classifier.fit(features, labels)
    return classifier


def main() -> int:
    args = parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base = VDS2RawManifest(args.dataset_root.expanduser().resolve())
    folds = build_sample_stratified_folds(base, args.split_seed)
    results: List[Dict[str, object]] = []
    for fold_data in folds:
        fold = int(fold_data["fold"])
        manifest = FoldManifest(base, fold_data["assignments"])
        for modality in MODALITIES:
            arrays_by_partition = {
                partition: partition_arrays(manifest, partition, modality)
                for partition in ("source_train", "target_train", "target_val")
            }
            val_features, val_labels, _ = arrays_by_partition["target_val"]
            best_key = None
            best_spec = None
            validation_rows: List[Dict[str, object]] = []
            for spec in candidates():
                classifier = fit(spec, arrays_by_partition)
                predictions = classifier.predict(val_features)
                metrics = classification_metrics(val_labels, predictions)
                row = {**spec, **{f"val_{key}": metrics[key] for key in ("acc", "precision_macro", "recall_macro", "f1_macro")}}
                validation_rows.append(row)
                key = (float(metrics["acc"]), float(metrics["f1_macro"]))
                if best_key is None or key > best_key:
                    best_key = key
                    best_spec = dict(spec)
            if best_spec is None:
                raise RuntimeError("No shape-probe candidate selected")
            classifier = fit(best_spec, arrays_by_partition)
            test_features, test_labels, test_ids = partition_arrays(
                manifest, "target_test", modality
            )
            probabilities = classifier.predict_proba(test_features)
            class_order = classifier.named_steps["svc"].classes_.astype(int)
            ordered = np.zeros((len(test_labels), 3), dtype=np.float64)
            ordered[:, class_order] = probabilities
            predictions = ordered.argmax(axis=1)
            metrics = classification_metrics(test_labels, predictions)
            run_dir = output_root / modality / f"fold_{fold}"
            rows = [
                {
                    "sample_id": sample_id,
                    "label": int(label),
                    "prediction": int(prediction),
                    "prob_Cargo": float(probability[0]),
                    "prob_Fishing": float(probability[1]),
                    "prob_Sailing_Pleasure": float(probability[2]),
                }
                for sample_id, label, prediction, probability in zip(
                    test_ids, test_labels, predictions, ordered
                )
            ]
            result: Dict[str, object] = {
                "fold": fold,
                "modality": modality,
                "selected": best_spec,
                "validation_key": best_key,
                "target_test": metrics,
            }
            atomic_write_json(run_dir / "result.json", result)
            write_csv(run_dir / "candidate_validation_metrics.csv", validation_rows)
            write_csv(run_dir / "target_test_predictions.csv", rows)
            results.append(result)
            print(
                f"fold={fold} modality={modality} selected={best_spec} "
                f"ACC={metrics['acc']:.4f} F1={metrics['f1_macro']:.4f}"
            )

    summary: List[Dict[str, object]] = []
    grouped: Dict[str, List[Mapping[str, object]]] = defaultdict(list)
    for result in results:
        grouped[str(result["modality"])].append(result)
    for modality, group in sorted(grouped.items()):
        row: Dict[str, object] = {"modality": modality, "runs": len(group)}
        for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
            values = [float(item["target_test"][metric]) for item in group]
            for statistic, value in numeric_summary(values).items():
                row[f"test_{metric}_{statistic}"] = value
        summary.append(row)
    atomic_write_json(output_root / "results.json", results)
    write_csv(output_root / "summary.csv", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
