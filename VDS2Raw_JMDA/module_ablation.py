"""Dedicated JMDA-Net training and ablation entry point for VDS2Raw v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List


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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--svd_max_sweeps", type=int, default=10)
    parser.add_argument("--svd_tolerance", type=float, default=1e-3)
    parser.add_argument("--svd_stat_batch_size", type=int, default=32)
    parser.add_argument("--lr_rgb", type=float, default=1e-5)
    parser.add_argument("--lr_other", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--tensor_loss_weight", type=float, default=0.12)
    parser.add_argument("--adv_loss_weight", type=float, default=0.08)
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
        "--freeze_rgb_early",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze ResNet18 stem and layer1",
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
        "svd_max_sweeps",
        "svd_stat_batch_size",
        "ot_sinkhorn_iterations",
        "ot_ramp_epochs",
    )
    for field in positive_integer_fields:
        if getattr(args, field) < 1:
            parser.error(f"--{field} must be at least 1")
    if args.num_workers < 0 or args.max_train_batches < 0 or args.ot_warmup_epochs < 0:
        parser.error("num_workers, max_train_batches, and ot_warmup_epochs cannot be negative")
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
            write_json(suite_dir / "summary.json", {"runs": summaries})

    aggregate = {}
    for ablation_name in ablations:
        selected = [run for run in summaries if run["ablation_name"] == ablation_name]
        fields = {
            "best_val_accuracy": [run["best_val_metrics"]["accuracy"] for run in selected],
            "best_val_f1_macro": [run["best_val_metrics"]["f1_macro"] for run in selected],
        }
        test_runs = [run for run in selected if run["target_test_metrics"] is not None]
        if test_runs:
            fields.update(
                {
                    "target_test_accuracy": [
                        run["target_test_metrics"]["accuracy"] for run in test_runs
                    ],
                    "target_test_f1_macro": [
                        run["target_test_metrics"]["f1_macro"] for run in test_runs
                    ],
                }
            )
        aggregate[ablation_name] = {
            field: {
                "values": values,
                "mean": sum(values) / len(values),
                "std_population": (
                    sum((value - sum(values) / len(values)) ** 2 for value in values)
                    / len(values)
                )
                ** 0.5,
            }
            for field, values in fields.items()
        }
    write_json(suite_dir / "summary.json", {"runs": summaries, "aggregate": aggregate})

    print(f"Training complete. Results: {suite_dir}")


if __name__ == "__main__":
    main()
