"""Frozen ImageNet feature probes for the grouped VDS2Raw protocol.

This is a small-data fallback to last-stage neural fine-tuning.  Candidate
SVM heads are selected exclusively on target_val; target_test is loaded only
after the winning candidate has been fixed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from VDS2Raw_JMDA.grouped_cv_baselines import (
    FoldManifest,
    build_grouped_folds,
    build_sample_stratified_folds,
)
from VDS2Raw_JMDA.single_modal_baselines import (
    MODEL_NAMES,
    SingleModalityDataset,
    atomic_write_json,
    classification_head,
    classification_metrics,
    create_model,
    make_loader,
    parse_csv_values,
    seed_everything,
    write_csv,
)
from VDS2Raw_JMDA.vds2raw_dataset import (
    CLASS_NAMES,
    VDS2RawManifest,
    compute_train_image_stats,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Grouped frozen-feature SVM probes")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"D:\Downloads\VDS2Raw\threeclass_ready"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=PACKAGE_DIR / "single_modal_runs"
    )
    parser.add_argument("--run-name", default="grouped_cv_feature_probe")
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
    parser.add_argument("--folds", default="1")
    parser.add_argument("--split-seed", type=int, default=20261110)
    parser.add_argument(
        "--split-mode",
        choices=("grouped", "sample-stratified"),
        default="grouped",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    return parser


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    model_name: str,
    dataset: SingleModalityDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    use_amp: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    captured: List[torch.Tensor] = []

    def capture_inputs(_module: nn.Module, inputs: Tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach().float().flatten(1).cpu())

    handle = classification_head(model, model_name).register_forward_pre_hook(capture_inputs)
    loader = make_loader(
        dataset, batch_size, False, num_workers, device, seed
    )
    labels: List[int] = []
    sample_ids: List[str] = []
    model.eval()
    try:
        for inputs, batch_labels, batch_sample_ids in loader:
            inputs = inputs.to(device, non_blocking=True)
            if model_name in {"resnet", "convnext"} and device.type == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                model(inputs)
            labels.extend(batch_labels.numpy().astype(int).tolist())
            sample_ids.extend(list(batch_sample_ids))
    finally:
        handle.remove()
    features = torch.cat(captured, dim=0).numpy()
    return features, np.asarray(labels, dtype=np.int64), sample_ids


def candidate_specs() -> List[Dict[str, object]]:
    return [
        {
            "train_scope": train_scope,
            "kernel": kernel,
            "class_weight": class_weight,
            "C": c_value,
        }
        for train_scope in ("target_only", "source_and_target")
        for kernel in ("linear", "rbf")
        for class_weight in (None, "balanced")
        for c_value in (0.01, 0.1, 1.0, 10.0, 100.0)
    ]


def fit_candidate(
    spec: Mapping[str, object],
    arrays: Mapping[str, Tuple[np.ndarray, np.ndarray, List[str]]],
) -> object:
    if spec["train_scope"] == "target_only":
        train_features, train_labels, _ = arrays["target_train"]
    else:
        train_features = np.concatenate(
            [arrays["source_train"][0], arrays["target_train"][0]], axis=0
        )
        train_labels = np.concatenate(
            [arrays["source_train"][1], arrays["target_train"][1]], axis=0
        )
    classifier = make_pipeline(
        StandardScaler(),
        SVC(
            C=float(spec["C"]),
            kernel=str(spec["kernel"]),
            gamma="scale",
            class_weight=spec["class_weight"],
        ),
    )
    classifier.fit(train_features, train_labels)
    return classifier


def run_probe(
    manifest: FoldManifest,
    image_stats: Mapping[str, Sequence[float]],
    model_name: str,
    modality: str,
    fold_number: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, object]:
    seed = 41 + fold_number
    seed_everything(seed)
    device = torch.device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"
    model = create_model(
        model_name,
        modality,
        len(CLASS_NAMES),
        image_size=args.image_size,
        backbone_profile="baseline",
        pretrained=True,
    ).to(device)
    if model_name in {"resnet", "convnext"} and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    arrays: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for partition in ("source_train", "target_train", "target_val"):
        dataset = SingleModalityDataset(
            manifest,
            partition,
            image_stats,
            modality,
            False,
            args.image_size,
            args.image_size,
        )
        arrays[partition] = extract_features(
            model,
            model_name,
            dataset,
            device,
            args.batch_size,
            args.num_workers,
            seed,
            use_amp,
        )

    validation_features, validation_labels, _ = arrays["target_val"]
    candidates: List[Dict[str, object]] = []
    best_key: Optional[Tuple[float, float]] = None
    best_spec: Optional[Dict[str, object]] = None
    for spec in candidate_specs():
        classifier = fit_candidate(spec, arrays)
        predictions = classifier.predict(validation_features)
        metrics = classification_metrics(validation_labels, predictions)
        row = {**spec, **{f"val_{key}": metrics[key] for key in ("acc", "precision_macro", "recall_macro", "f1_macro")}}
        candidates.append(row)
        key = float(metrics["acc"]), float(metrics["f1_macro"])
        if best_key is None or key > best_key:
            best_key = key
            best_spec = dict(spec)
    if best_spec is None:
        raise RuntimeError("No SVM candidate was selected")
    selected = fit_candidate(best_spec, arrays)
    # The test partition is instantiated and loaded only after the winning
    # candidate has been fixed by target_val.
    test_dataset = SingleModalityDataset(
        manifest,
        "target_test",
        image_stats,
        modality,
        False,
        args.image_size,
        args.image_size,
    )
    arrays["target_test"] = extract_features(
        model,
        model_name,
        test_dataset,
        device,
        args.batch_size,
        args.num_workers,
        seed,
        use_amp,
    )
    test_features, test_labels, test_sample_ids = arrays["target_test"]
    test_predictions = selected.predict(test_features)
    test_metrics = classification_metrics(test_labels, test_predictions)
    best_row = max(
        candidates,
        key=lambda row: (float(row["val_acc"]), float(row["val_f1_macro"])),
    )
    prediction_rows = [
        {
            "sample_id": sample_id,
            "label": int(label),
            "prediction": int(prediction),
        }
        for sample_id, label, prediction in zip(
            test_sample_ids, test_labels, test_predictions
        )
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_validation_metrics.csv", candidates)
    write_csv(output_dir / "target_test_predictions.csv", prediction_rows)
    result: Dict[str, object] = {
        "fold": fold_number,
        "seed": seed,
        "model": model_name,
        "modality": modality,
        "feature_extractor": "frozen ImageNet pretrained backbone",
        "selection": "target_val ACC, macro-F1 tie break",
        "selected_candidate": best_spec,
        "validation": {
            key.removeprefix("val_"): value
            for key, value in best_row.items()
            if key.startswith("val_")
        },
        "target_test": test_metrics,
    }
    atomic_write_json(output_dir / "result.json", result)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = build_parser().parse_args()
    selected_folds = [int(value.strip()) for value in args.folds.split(",")]
    base = VDS2RawManifest(args.dataset_root.resolve())
    folds = (
        build_grouped_folds(base, args.split_seed)
        if args.split_mode == "grouped"
        else build_sample_stratified_folds(base, args.split_seed)
    )
    run_root = args.output_root.resolve() / args.run_name
    results: List[Dict[str, object]] = []
    for fold_data in folds:
        fold_number = int(fold_data["fold"])
        if fold_number not in selected_folds:
            continue
        manifest = FoldManifest(base, fold_data["assignments"])
        image_stats = compute_train_image_stats(manifest)
        for model_name in args.models:
            for modality in args.modalities:
                result = run_probe(
                    manifest,
                    image_stats,
                    model_name,
                    modality,
                    fold_number,
                    args,
                    run_root / model_name / modality / f"fold_{fold_number}",
                )
                results.append(result)
                metrics = result["target_test"]
                print(
                    f"FINAL fold={fold_number} model={model_name} modality={modality} "
                    f"candidate={json.dumps(result['selected_candidate'])} "
                    f"ACC={metrics['acc']:.4f} P={metrics['precision_macro']:.4f} "
                    f"R={metrics['recall_macro']:.4f} F1={metrics['f1_macro']:.4f}"
                )
    atomic_write_json(run_root / "results.json", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
