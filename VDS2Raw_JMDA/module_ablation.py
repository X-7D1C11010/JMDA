"""Dedicated JMDA-Net training and ablation entry point for VDS2Raw v2."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .engine import ABLATION_SPECS, run_experiment
    from .vds2raw_dataset import (
        VDS2RawManifest,
        compute_train_image_stats,
        write_json,
    )
except ImportError:
    from engine import ABLATION_SPECS, run_experiment  # type: ignore
    from vds2raw_dataset import (  # type: ignore
        VDS2RawManifest,
        compute_train_image_stats,
        write_json,
    )


def parse_seeds(value: str) -> List[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    if len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seed values must be unique")
    return seeds


def summarize_numeric_values(values: Sequence[float]) -> Dict[str, object]:
    numeric = [float(value) for value in values]
    if not numeric:
        raise ValueError("Cannot summarize an empty value list")
    mean = sum(numeric) / len(numeric)
    population_variance = sum((value - mean) ** 2 for value in numeric) / len(numeric)
    sample_variance = (
        sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
        if len(numeric) > 1
        else None
    )
    return {
        "values": numeric,
        "count": len(numeric),
        "minimum": min(numeric),
        "maximum": max(numeric),
        "mean": mean,
        "variance_population": population_variance,
        "std_population": population_variance**0.5,
        "variance_sample": sample_variance,
        "std_sample": sample_variance**0.5 if sample_variance is not None else None,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_run_row(run: Mapping[str, object]) -> Dict[str, object]:
    best = run["best_val_metrics"]
    maxima = run["best_val_metrics_by_name"]
    test = run["target_test_metrics"]
    return {
        "ablation_name": run["ablation_name"],
        "seed": run["seed"],
        "best_epoch": run["best_epoch"],
        "val_accuracy": best["accuracy"],
        "val_precision_macro": best["precision_macro"],
        "val_recall_macro": best["recall_macro"],
        "val_f1_macro": best["f1_macro"],
        "max_val_accuracy": maxima["accuracy"]["value"],
        "max_val_accuracy_epoch": maxima["accuracy"]["epoch"],
        "max_val_precision_macro": maxima["precision_macro"]["value"],
        "max_val_precision_macro_epoch": maxima["precision_macro"]["epoch"],
        "max_val_recall_macro": maxima["recall_macro"]["value"],
        "max_val_recall_macro_epoch": maxima["recall_macro"]["epoch"],
        "max_val_f1_macro": maxima["f1_macro"]["value"],
        "max_val_f1_macro_epoch": maxima["f1_macro"]["epoch"],
        "test_accuracy": test["accuracy"] if test is not None else None,
        "test_precision_macro": test["precision_macro"] if test is not None else None,
        "test_recall_macro": test["recall_macro"] if test is not None else None,
        "test_f1_macro": test["f1_macro"] if test is not None else None,
        "run_dir": run["run_dir"],
    }


def _complexity_run_row(run: Mapping[str, object]) -> Dict[str, object]:
    return {
        "ablation_name": run["ablation_name"],
        "seed": run["seed"],
        **dict(run["complexity"]),
        "run_dir": run["run_dir"],
    }


def _aggregate_runs(runs: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    fields = {}
    for split, key in (("best_val", "best_val_metrics"), ("target_test", "target_test_metrics")):
        available = [run[key] for run in runs if run[key] is not None]
        if not available:
            continue
        fields[split] = {
            metric: summarize_numeric_values([item[metric] for item in available])
            for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        }
    fields["independent_val_max"] = {
        metric: summarize_numeric_values(
            [run["best_val_metrics_by_name"][metric]["value"] for run in runs]
        )
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    }

    complexity_fields = (
        "total_parameters",
        "trainable_parameters",
        "inference_parameters",
        "inference_gflops_per_sample",
        "training_forward_gflops_per_pair",
        "training_loop_seconds",
        "total_run_seconds",
        "mean_epoch_seconds",
        "inference_latency_ms_per_sample",
        "inference_throughput_samples_per_second",
        "training_peak_allocated_bytes",
        "training_peak_reserved_bytes",
    )
    complexity = {}
    for field in complexity_fields:
        values = [run["complexity"].get(field) for run in runs]
        numeric = [float(value) for value in values if value is not None]
        if numeric:
            complexity[field] = summarize_numeric_values(numeric)

    # The user requested the highest target-test result among the independent
    # validation-selected seed checkpoints.  Save it as an explicitly
    # diagnostic oracle only: it must not replace the mean/variance report or
    # be used for model/hyperparameter selection.
    test_runs = [run for run in runs if run["target_test_metrics"] is not None]
    if test_runs:
        oracle_run = max(
            test_runs,
            key=lambda run: (
                float(run["target_test_metrics"]["accuracy"]),
                float(run["target_test_metrics"]["f1_macro"]),
                -int(run["seed"]),
            ),
        )
        diagnostic_test_oracle = {
            "warning": (
                "Highest target_test result across seed checkpoints is an oracle "
                "diagnostic, not an unbiased final metric and not a selection rule."
            ),
            "selection_pool": (
                "one target_val-selected eligible checkpoint per independent seed"
            ),
            "seed": oracle_run["seed"],
            "best_epoch": oracle_run["best_epoch"],
            "metrics": oracle_run["target_test_metrics"],
            "run_dir": oracle_run["run_dir"],
        }
    else:
        diagnostic_test_oracle = None
    return {
        "metrics": fields,
        "complexity": complexity,
        "diagnostic_target_test_oracle_best_checkpoint": diagnostic_test_oracle,
    }


def write_suite_reports(
    suite_dir: Path,
    summaries: Sequence[Mapping[str, object]],
    ablations: Sequence[str],
) -> None:
    aggregate = {}
    aggregate_metric_rows = []
    aggregate_complexity_rows = []
    for ablation_name in ablations:
        selected = [run for run in summaries if run["ablation_name"] == ablation_name]
        if not selected:
            continue
        ablation_aggregate = _aggregate_runs(selected)
        aggregate[ablation_name] = ablation_aggregate
        ablation_dir = suite_dir / ablation_name
        write_json(
            ablation_dir / "summary.json",
            {"runs": selected, "aggregate": ablation_aggregate},
        )
        oracle = ablation_aggregate[
            "diagnostic_target_test_oracle_best_checkpoint"
        ]
        if oracle is not None:
            write_json(
                ablation_dir / "diagnostic_target_test_oracle_best_checkpoint.json",
                oracle,
            )
        _write_csv(
            ablation_dir / "metrics_by_seed.csv",
            [_metric_run_row(run) for run in selected],
        )
        _write_csv(
            ablation_dir / "complexity_by_seed.csv",
            [_complexity_run_row(run) for run in selected],
        )
        for split, metrics in ablation_aggregate["metrics"].items():
            for metric, statistics in metrics.items():
                aggregate_metric_rows.append(
                    {
                        "ablation_name": ablation_name,
                        "split": split,
                        "metric": metric,
                        **{key: value for key, value in statistics.items() if key != "values"},
                    }
                )
        for metric, statistics in ablation_aggregate["complexity"].items():
            aggregate_complexity_rows.append(
                {
                    "ablation_name": ablation_name,
                    "metric": metric,
                    **{key: value for key, value in statistics.items() if key != "values"},
                }
            )

    write_json(suite_dir / "summary.json", {"runs": list(summaries), "aggregate": aggregate})
    _write_csv(suite_dir / "metrics_by_run.csv", [_metric_run_row(run) for run in summaries])
    _write_csv(
        suite_dir / "complexity_by_run.csv",
        [_complexity_run_row(run) for run in summaries],
    )
    _write_csv(suite_dir / "aggregate_metrics.csv", aggregate_metric_rows)
    _write_csv(suite_dir / "aggregate_complexity.csv", aggregate_complexity_rows)


def build_parser() -> argparse.ArgumentParser:
    default_dataset_root = os.environ.get(
        "VDS2RAW_ROOT",
        r"D:\Downloads\VDS2Raw\threeclass_ready",
    )
    parser = argparse.ArgumentParser(
        description="VDS2Raw v2 three-modal supervised domain adaptation with JMDA-Net"
    )
    parser.add_argument("--dataset_root", default=default_dataset_root)
    parser.add_argument("--output_root", default=str(PACKAGE_DIR / "runs"))
    parser.add_argument(
        "--ablation_mode",
        choices=["all", *ABLATION_SPECS.keys()],
        default="full",
        help="all runs the full model and all three Tensor/OT ablations",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--inference_warmup_runs",
        type=int,
        default=10,
        help="model-only warm-up forwards before inference timing",
    )
    parser.add_argument(
        "--inference_benchmark_runs",
        type=int,
        default=50,
        help="model-only target forwards used for latency/throughput timing",
    )
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument(
        "--nir_projection_dim",
        type=int,
        default=32,
        help=(
            "NIR Tensor projection width; repeated single-band B8 has lower "
            "empirical rank than RGB"
        ),
    )
    parser.add_argument(
        "--ais_projection_dim",
        type=int,
        default=32,
        help=(
            "AIS Tensor projection width; 32 stays below the observed "
            "37--41 dimensional AIS effective rank"
        ),
    )
    parser.add_argument("--svd_max_sweeps", type=int, default=10)
    parser.add_argument("--svd_tolerance", type=float, default=1e-3)
    parser.add_argument("--svd_stat_batch_size", type=int, default=128)
    parser.add_argument("--lr_rgb", type=float, default=5e-6)
    parser.add_argument("--lr_other", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--tensor_loss_weight", type=float, default=0.12)
    parser.add_argument("--adv_loss_weight", type=float, default=0.08)
    parser.add_argument(
        "--discriminator_update_interval",
        type=int,
        default=2,
        help="update the domain discriminator every N training batches",
    )
    parser.add_argument(
        "--transport_mode",
        choices=["row_softmax", "sinkhorn"],
        default="sinkhorn",
    )
    parser.add_argument("--ot_epsilon", type=float, default=0.1)
    parser.add_argument("--ot_sinkhorn_iterations", type=int, default=50)
    parser.add_argument("--ot_correction_scale", type=float, default=0.1)
    parser.add_argument("--ot_correction_reg_weight", type=float, default=1e-3)
    parser.add_argument("--ot_warmup_epochs", type=int, default=10)
    parser.add_argument("--ot_ramp_epochs", type=int, default=5)
    parser.add_argument(
        "--class_conditional_ot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use locally cached ImageNet ResNet18 weights",
    )
    parser.add_argument(
        "--pretrained_weights",
        default=None,
        help="offline path to the official resnet18-f37072fd.pth file",
    )
    parser.add_argument(
        "--freeze_rgb_backbone",
        "--freeze_rgb_early",
        dest="freeze_rgb_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze the complete ResNet18 feature stack and train only its projection",
    )
    parser.add_argument(
        "--balance_classes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="repeat each target class to the largest target-class count per epoch",
    )
    parser.add_argument(
        "--image_augmentation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "historical resize-256/random-crop-224/horizontal-flip transform; "
            "AIS heading_sin is transformed with the flip"
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="0 uses the complete target_train partition; positive values are smoke-test only",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="one epoch/one batch with OT active; still uses complete train-only SVD statistics",
    )
    parser.add_argument(
        "--evaluate_test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="evaluate target_test once after validation-selected checkpoint loading",
    )
    parser.add_argument("--audit_only", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args) -> None:
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "projection_dim",
        "nir_projection_dim",
        "ais_projection_dim",
        "svd_max_sweeps",
        "svd_stat_batch_size",
        "ot_sinkhorn_iterations",
        "ot_ramp_epochs",
        "discriminator_update_interval",
        "inference_benchmark_runs",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be at least 1")
    if (
        args.num_workers < 0
        or args.max_train_batches < 0
        or args.ot_warmup_epochs < 0
        or args.inference_warmup_runs < 0
    ):
        parser.error(
            "num_workers, max_train_batches, ot_warmup_epochs, and "
            "inference_warmup_runs cannot be negative"
        )
    for field in (
        "lr_rgb",
        "lr_other",
        "weight_decay",
        "tensor_loss_weight",
        "adv_loss_weight",
        "ot_epsilon",
        "ot_correction_scale",
        "ot_correction_reg_weight",
    ):
        if getattr(args, field) < 0:
            parser.error(f"--{field} cannot be negative")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label_smoothing must be in [0, 1)")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    if args.smoke_test:
        args.epochs = 1
        args.max_train_batches = 1
        args.ot_warmup_epochs = 0
        args.ot_ramp_epochs = 1
        args.evaluate_test = False
        args.inference_warmup_runs = min(args.inference_warmup_runs, 2)
        args.inference_benchmark_runs = min(args.inference_benchmark_runs, 3)

    manifest = VDS2RawManifest(args.dataset_root)
    audit_report = manifest.audit(validate_npz=True)
    image_stats = compute_train_image_stats(manifest)
    print(json.dumps(audit_report, ensure_ascii=False, indent=2))
    print(json.dumps(image_stats, ensure_ascii=False, indent=2))
    if args.audit_only:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suite_dir = Path(args.output_root).expanduser().resolve() / f"vds2raw_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=False)
    write_json(suite_dir / "dataset_audit.json", audit_report)
    write_json(suite_dir / "image_stats.json", image_stats)
    write_json(
        suite_dir / "invocation.json",
        json.loads(json.dumps(vars(args), default=str)),
    )

    ablations = (
        ["full", "no_tensor_no_ot", "no_tensor_with_ot", "with_tensor_no_ot"]
        if args.ablation_mode == "all"
        else [args.ablation_mode]
    )
    summaries = []
    for ablation_name in ablations:
        for seed in args.seeds:
            summaries.append(
                run_experiment(
                    args=args,
                    manifest=manifest,
                    image_stats=image_stats,
                    audit_report=audit_report,
                    ablation_name=ablation_name,
                    seed=seed,
                    suite_dir=suite_dir,
                )
            )
            # Persist suite-level raw tables after every independent seed so a
            # later interruption cannot erase completed experiment summaries.
            write_suite_reports(suite_dir, summaries, ablations)

    write_suite_reports(suite_dir, summaries, ablations)

    print(f"Training complete. Results: {suite_dir}")


if __name__ == "__main__":
    main()
