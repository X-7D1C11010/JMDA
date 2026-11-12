"""Re-evaluate sample-stratified checkpoints with a validation-approved TTA mode.

This script does not retrain a model.  It loads the validation-selected
checkpoint from each fold, compares identity and horizontal-flip inference on
target_val, and only then evaluates the requested (already fixed) prediction
mode on target_test.  The test partition is never used to select the mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    result.add_argument("--output-name", default="identity_checkpoint_evaluation")
    result.add_argument("--split-seed", type=int, default=20261110)
    result.add_argument(
        "--prediction-mode",
        choices=("identity", "horizontal_flip_mean"),
        default="identity",
    )
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--num-workers", type=int, default=0)
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return result


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
            "prediction_mode": group[0]["prediction_mode"],
        }
        for metric in ("acc", "precision_macro", "recall_macro", "f1_macro"):
            values = [float(result["target_test"][metric]) for result in group]
            for statistic, value in numeric_summary(values).items():
                row[f"test_{metric}_{statistic}"] = value
        rows.append(row)
    return rows


def main() -> int:
    args = parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    base_run = args.base_run.expanduser().resolve()
    output_root = base_run / args.output_name
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    base = VDS2RawManifest(dataset_root)
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
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(checkpoint_path)
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
                val_identity, _ = evaluate_prediction_mode(
                    model, val_loader, device, use_amp, channels_last, "identity"
                )
                val_flip, _ = evaluate_prediction_mode(
                    model,
                    val_loader,
                    device,
                    use_amp,
                    channels_last,
                    "horizontal_flip_mean",
                )
                selected_validation = (
                    val_identity
                    if args.prediction_mode == "identity"
                    else val_flip
                )
                alternative_validation = (
                    val_flip
                    if args.prediction_mode == "identity"
                    else val_identity
                )
                if float(selected_validation["acc"]) < float(alternative_validation["acc"]):
                    raise RuntimeError(
                        f"Requested {args.prediction_mode} is worse on target_val for "
                        f"{model_name}/{modality}/fold_{fold}: "
                        f"selected={selected_validation['acc']}, "
                        f"alternative={alternative_validation['acc']}"
                    )

                # Instantiate target_test only after the inference mode has
                # passed the validation-only selection check above.
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
                test_metrics, prediction_rows = evaluate_prediction_mode(
                    model,
                    test_loader,
                    device,
                    use_amp,
                    channels_last,
                    args.prediction_mode,
                )
                result: Dict[str, object] = {
                    "status": "complete",
                    "fold": fold,
                    "seed": seed,
                    "model": model_name,
                    "modality": modality,
                    "checkpoint": str(checkpoint_path),
                    "prediction_mode": args.prediction_mode,
                    "selection": "target_val only; target_test not used for mode selection",
                    "validation_identity": compact_metrics(val_identity),
                    "validation_horizontal_flip_mean": compact_metrics(val_flip),
                    "target_test": test_metrics,
                }
                atomic_write_json(result_path, result)
                write_csv(run_dir / "target_test_predictions.csv", prediction_rows)
                results.append(result)
                print(
                    f"fold={fold} model={model_name} modality={modality} "
                    f"mode={args.prediction_mode} ACC={test_metrics['acc']:.4f} "
                    f"F1={test_metrics['f1_macro']:.4f}"
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
