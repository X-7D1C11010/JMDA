"""SVM probes on fine-tuned backbone embeddings plus RGB/NIR shape features."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from VDS2Raw_JMDA.grouped_cv_baselines import FoldManifest, build_sample_stratified_folds
from VDS2Raw_JMDA.grouped_cv_feature_probe import (
    candidate_specs,
    extract_features,
    fit_candidate,
)
from VDS2Raw_JMDA.sample_protocol_shape_probe import partition_arrays as shape_arrays
from VDS2Raw_JMDA.single_modal_baselines import (
    MODEL_NAMES,
    SingleModalityDataset,
    atomic_write_json,
    classification_metrics,
    create_model,
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
        "--checkpoint-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "single_modal_runs"
        / "vds2raw_sample_stratified_fixedtta_30runs_20261110",
    )
    result.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent
        / "single_modal_runs"
        / "sample_stratified_deep_shape_probe_20261110",
    )
    result.add_argument("--split-seed", type=int, default=20261110)
    result.add_argument("--batch-size", type=int, default=32)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


def combined_arrays(
    model,
    model_name: str,
    manifest: FoldManifest,
    image_stats: Mapping[str, Sequence[float]],
    modality: str,
    partition: str,
    device: torch.device,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dataset = SingleModalityDataset(
        manifest, partition, image_stats, modality, False, 224, 224
    )
    deep, labels, sample_ids = extract_features(
        model,
        model_name,
        dataset,
        device,
        args.batch_size,
        args.num_workers,
        seed,
        device.type == "cuda",
    )
    shape, shape_labels, shape_ids = shape_arrays(manifest, partition, modality)
    if sample_ids != shape_ids or not np.array_equal(labels, shape_labels):
        raise RuntimeError("Deep and shape feature rows are not aligned")
    # Normalize block magnitudes before the pipeline's feature-wise scaling so
    # neither the backbone embedding nor HOG block dominates numerically.
    deep = deep / np.maximum(np.linalg.norm(deep, axis=1, keepdims=True), 1e-8)
    shape = shape / np.maximum(np.linalg.norm(shape, axis=1, keepdims=True), 1e-8)
    return np.concatenate([deep, shape], axis=1), labels, sample_ids


def main() -> int:
    args = parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
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
                model = create_model(
                    model_name,
                    modality,
                    len(CLASS_NAMES),
                    image_size=224,
                    backbone_profile="baseline",
                    pretrained=False,
                ).to(device)
                checkpoint_path = (
                    checkpoint_root / model_name / modality / f"fold_{fold}" / "best.pt"
                )
                checkpoint = torch.load(
                    checkpoint_path, map_location=device, weights_only=False
                )
                model.load_state_dict(checkpoint["model_state"])
                if model_name in {"resnet", "convnext"} and device.type == "cuda":
                    model = model.to(memory_format=torch.channels_last)

                arrays = {
                    partition: combined_arrays(
                        model,
                        model_name,
                        manifest,
                        image_stats,
                        modality,
                        partition,
                        device,
                        args,
                        seed,
                    )
                    for partition in ("source_train", "target_train", "target_val")
                }
                val_features, val_labels, _ = arrays["target_val"]
                best_key = None
                best_spec = None
                candidate_rows: List[Dict[str, object]] = []
                for spec in candidate_specs():
                    classifier = fit_candidate(spec, arrays)
                    predictions = classifier.predict(val_features)
                    metrics = classification_metrics(val_labels, predictions)
                    row = {
                        **spec,
                        **{
                            f"val_{key}": metrics[key]
                            for key in ("acc", "precision_macro", "recall_macro", "f1_macro")
                        },
                    }
                    candidate_rows.append(row)
                    key = (float(metrics["acc"]), float(metrics["f1_macro"]))
                    if best_key is None or key > best_key:
                        best_key = key
                        best_spec = dict(spec)
                if best_spec is None:
                    raise RuntimeError("No deep-shape probe was selected")
                classifier = fit_candidate(best_spec, arrays)
                test_features, test_labels, test_ids = combined_arrays(
                    model,
                    model_name,
                    manifest,
                    image_stats,
                    modality,
                    "target_test",
                    device,
                    args,
                    seed,
                )
                predictions = classifier.predict(test_features)
                metrics = classification_metrics(test_labels, predictions)
                run_dir = output_root / model_name / modality / f"fold_{fold}"
                prediction_rows = [
                    {
                        "sample_id": sample_id,
                        "label": int(label),
                        "prediction": int(prediction),
                    }
                    for sample_id, label, prediction in zip(
                        test_ids, test_labels, predictions
                    )
                ]
                result: Dict[str, object] = {
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "modality": modality,
                    "checkpoint": str(checkpoint_path),
                    "selected": best_spec,
                    "validation_key": best_key,
                    "target_test": metrics,
                }
                atomic_write_json(run_dir / "result.json", result)
                write_csv(run_dir / "candidate_validation_metrics.csv", candidate_rows)
                write_csv(run_dir / "target_test_predictions.csv", prediction_rows)
                results.append(result)
                print(
                    f"fold={fold} model={model_name} modality={modality} "
                    f"selected={best_spec} ACC={metrics['acc']:.4f} "
                    f"F1={metrics['f1_macro']:.4f}"
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    summary: List[Dict[str, object]] = []
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
            values = [float(item["target_test"][metric]) for item in group]
            for statistic, value in numeric_summary(values).items():
                row[f"test_{metric}_{statistic}"] = value
        summary.append(row)
    atomic_write_json(output_root / "results.json", results)
    write_csv(output_root / "summary.csv", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
