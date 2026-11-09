"""Run supervised VDS2Raw RGB, NIR, and heading-only AIS experiments.

Each modality/seed is launched as an independent ``module_ablation.py``
process.  This guarantees that GPU admission is checked again before every
training run and leaves the original trainer outputs intact.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
MODULE_SCRIPT = PACKAGE_DIR / "module_ablation.py"

try:
    from .run_hparam_sensitivity import (
        _atomic_write_csv,
        _atomic_write_json,
        _fingerprint,
        _utc_timestamp,
        wait_for_eligible_gpu,
    )
except ImportError:
    from run_hparam_sensitivity import (  # type: ignore
        _atomic_write_csv,
        _atomic_write_json,
        _fingerprint,
        _utc_timestamp,
        wait_for_eligible_gpu,
    )


DEFAULT_DATASET_ROOT = "/home/lixiang/lx/VDS2Raw/threeclass_ready"
DEFAULT_OUTPUT_ROOT = "/home/lixiang/lx/JMDA/VDS2Raw_JMDA/runs/modality_experiments"
MODALITIES = ("rgb", "nir", "ais")
ABLATION_BY_MODALITY = {
    "rgb": "single_rgb",
    "nir": "single_nir",
    "ais": "single_ais",
}

PER_SEED_FIELDS = (
    "task_index",
    "task_id",
    "status",
    "modality",
    "input_definition",
    "feature_dimension",
    "seed",
    "selected_epoch",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
    "gpu_physical_index",
    "trainer_suite_dir",
    "trainer_run_dir",
    "error",
)

METRIC_FIELDS = (
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "test_accuracy",
    "test_precision_macro",
    "test_recall_macro",
    "test_f1_macro",
)


def parse_seeds(value: str) -> List[int]:
    try:
        seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from exc
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serial supervised VDS2Raw single-modality experiments: "
            "RGB, NIR, and JSON heading-only AIS"
        )
    )
    parser.add_argument(
        "--dataset-root",
        "--dataset_root",
        dest="dataset_root",
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-root",
        "--output_root",
        dest="output_root",
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--pretrained-weights",
        "--pretrained_weights",
        dest="pretrained_weights",
        default=None,
        help="Offline ResNet18 weights used by the RGB experiment",
    )
    parser.add_argument(
        "--seeds",
        type=parse_seeds,
        default=parse_seeds("42,43,44,45,46"),
        help="Comma-separated independent seeds; each is a separate GPU-admitted task",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--evaluate-test",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate target_test once from each target_val-selected checkpoint",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu-min-free-gb", type=float, default=15.0)
    parser.add_argument("--gpu-max-utilization", type=float, default=15.0)
    parser.add_argument("--gpu-poll-seconds", type=float, default=60.0)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.epochs < 15:
        parser.error("--epochs must be at least 15 for the inherited validation window")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and num workers cannot be negative")
    if args.gpu_min_free_gb < 0:
        parser.error("--gpu-min-free-gb cannot be negative")
    if not 0 < args.gpu_max_utilization <= 100:
        parser.error("--gpu-max-utilization must be in (0, 100]")
    if args.gpu_poll_seconds <= 0:
        parser.error("--gpu-poll-seconds must be positive")


def input_definition(modality: str) -> str:
    return {
        "rgb": "Sentinel-2 B4/B3/B2 visible RGB",
        "nir": "Sentinel-2 B8 near-infrared (not thermal infrared)",
        "ais": (
            "JSON heading only: heading_sin, heading_cos and one validity flag; "
            "not RF/IQ; excludes length, width, nav status, ship type and MMSI"
        ),
    }[modality]


def build_tasks(seeds: Sequence[int]) -> List[Dict[str, object]]:
    tasks = []
    index = 1
    for modality in MODALITIES:
        for seed in seeds:
            tasks.append(
                {
                    "task_index": index,
                    "task_id": f"{index:03d}_{modality}_seed_{seed}",
                    "modality": modality,
                    "ablation_mode": ABLATION_BY_MODALITY[modality],
                    "seed": int(seed),
                    "input_definition": input_definition(modality),
                    "feature_dimension": 512,
                }
            )
            index += 1
    if len(tasks) != len(MODALITIES) * len(seeds):
        raise AssertionError("Modality task plan size mismatch")
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise AssertionError("Modality task identifiers are not unique")
    return tasks


def complete_training_parameters(
    task: Mapping[str, object], args: argparse.Namespace
) -> Dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root),
        "training_protocol": "supervised",
        "ablation_mode": task["ablation_mode"],
        "modality": task["modality"],
        "input_definition": task["input_definition"],
        "feature_dimension": 512,
        "seed": task["seed"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": "cuda",
        "transport_mode": "sinkhorn",
        "ot_epsilon": 0.10,
        "ot_sinkhorn_iterations": 50,
        "ot_correction_scale": 0.1,
        "ot_correction_reg_weight": 1e-3,
        "adv_loss_weight": 0.08,
        "lr_rgb": 5e-6,
        "lr_other": 1e-4,
        "lr_discriminator": 5e-5,
        "weight_decay": 5e-4,
        "discriminator_update_interval": 2,
        "freeze_rgb_backbone": True,
        "balance_classes": True,
        "image_augmentation": True,
        "ot_warmup_epochs": 10,
        "ot_ramp_epochs": 5,
        "checkpoint_selection": "target_val; eligible epochs >= 15",
        "evaluate_test": args.evaluate_test,
        "pretrained_weights": args.pretrained_weights,
    }


def build_command(
    task: Mapping[str, object],
    args: argparse.Namespace,
    trainer_output_root: Path,
) -> List[str]:
    command = [
        str(args.python_executable),
        "-u",
        "-B",
        str(MODULE_SCRIPT),
        "--dataset_root",
        str(args.dataset_root),
        "--output_root",
        str(trainer_output_root),
        "--training_protocol",
        "supervised",
        "--ablation_mode",
        str(task["ablation_mode"]),
        "--seeds",
        str(task["seed"]),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        "cuda",
        "--transport_mode",
        "sinkhorn",
        "--projection_dim",
        "64",
        "--nir_projection_dim",
        "32",
        "--ais_projection_dim",
        "32",
        "--svd_max_sweeps",
        "10",
        "--svd_tolerance",
        "1e-3",
        "--svd_stat_batch_size",
        "128",
        "--lr_rgb",
        "5e-6",
        "--lr_other",
        "1e-4",
        "--lr_discriminator",
        "5e-5",
        "--weight_decay",
        "5e-4",
        "--tensor_loss_weight",
        "0.12",
        "--adv_loss_weight",
        "0.08",
        "--discriminator_update_interval",
        "2",
        "--freeze_rgb_backbone",
        "--balance_classes",
        "--image_augmentation",
        "--ot_epsilon",
        "0.1",
        "--ot_sinkhorn_iterations",
        "50",
        "--ot_correction_scale",
        "0.1",
        "--ot_correction_reg_weight",
        "1e-3",
        "--ot_warmup_epochs",
        "10",
        "--ot_ramp_epochs",
        "5",
        "--evaluate_test" if args.evaluate_test else "--no-evaluate_test",
    ]
    if args.pretrained_weights:
        command.extend(["--pretrained_weights", str(args.pretrained_weights)])
    return command


def base_row(task: Mapping[str, object], status: str) -> Dict[str, object]:
    return {
        "task_index": task["task_index"],
        "task_id": task["task_id"],
        "status": status,
        "modality": task["modality"],
        "input_definition": task["input_definition"],
        "feature_dimension": task["feature_dimension"],
        "seed": task["seed"],
        "selected_epoch": "",
        **{field: "" for field in METRIC_FIELDS},
        "gpu_physical_index": "",
        "trainer_suite_dir": "",
        "trainer_run_dir": "",
        "error": "",
    }


def read_metrics(
    suite_dir: Path, task: Mapping[str, object]
) -> Tuple[Dict[str, object], Dict[str, str]]:
    metrics_path = suite_dir / "metrics_by_run.csv"
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one metrics row, found {len(rows)}")
    raw = rows[0]
    expected = (
        "supervised",
        str(task["ablation_mode"]),
        str(task["seed"]),
    )
    actual = (raw.get("training_protocol"), raw.get("ablation_name"), raw.get("seed"))
    if actual != expected:
        raise RuntimeError(f"Metrics identity mismatch: expected={expected}, actual={actual}")
    required = (
        "best_epoch",
        "val_accuracy",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "run_dir",
    )
    missing = [name for name in required if not str(raw.get(name, "")).strip()]
    if missing:
        raise RuntimeError(f"Missing selected-checkpoint fields: {missing}")
    row = base_row(task, "completed")
    row.update(
        {
            "selected_epoch": int(raw["best_epoch"]),
            "val_accuracy": float(raw["val_accuracy"]),
            "val_precision_macro": float(raw["val_precision_macro"]),
            "val_recall_macro": float(raw["val_recall_macro"]),
            "val_f1_macro": float(raw["val_f1_macro"]),
            "trainer_suite_dir": str(suite_dir),
            "trainer_run_dir": raw["run_dir"],
        }
    )
    for field in (
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
    ):
        value = str(raw.get(field, "")).strip()
        if value:
            row[field] = float(value)
    return row, raw


def find_suite(
    trainer_output_root: Path, task: Mapping[str, object]
) -> Tuple[Path, Dict[str, object], Dict[str, str]]:
    errors = []
    candidates = sorted(
        trainer_output_root.glob("vds2raw_*/metrics_by_run.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            row, raw = read_metrics(path.parent, task)
            return path.parent, row, raw
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    raise RuntimeError(
        "No complete matching metrics_by_run.csv. " + "; ".join(errors)
    )


def load_json(path: Path) -> Optional[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def resume_row(
    task_dir: Path,
    task: Mapping[str, object],
    fingerprint: str,
) -> Optional[Dict[str, object]]:
    status = load_json(task_dir / "status.json")
    if (
        status is None
        or status.get("status") != "completed"
        or status.get("fingerprint") != fingerprint
    ):
        return None
    try:
        row, _ = read_metrics(Path(str(status["trainer_suite_dir"])), task)
        row["gpu_physical_index"] = status.get("gpu_physical_index", "")
        return row
    except Exception:
        return None


def next_attempt(logs_dir: Path) -> int:
    numbers = []
    for path in logs_dir.glob("train_attempt_*.log"):
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            pass
    return max(numbers, default=0) + 1


def run_child(command: Sequence[str], log_path: Path, environment: Mapping[str, str]) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        handle.write(f"started_at={_utc_timestamp()}\n")
        handle.write(f"command={shlex.join(command)}\n\n")
        process = subprocess.Popen(
            list(command),
            cwd=str(REPO_ROOT),
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def write_tables(
    output_root: Path,
    rows_by_task: Mapping[str, Mapping[str, object]],
    tasks: Sequence[Mapping[str, object]],
) -> None:
    rows = [
        rows_by_task[str(task["task_id"])]
        for task in tasks
        if str(task["task_id"]) in rows_by_task
    ]
    _atomic_write_csv(output_root / "per_seed.csv", rows, PER_SEED_FIELDS)

    summary_rows = []
    for modality in MODALITIES:
        selected = [row for row in rows if row["modality"] == modality]
        completed = [row for row in selected if row["status"] == "completed"]
        summary: Dict[str, object] = {
            "modality": modality,
            "input_definition": input_definition(modality),
            "feature_dimension": 512,
            "requested_runs": sum(task["modality"] == modality for task in tasks),
            "completed_runs": len(completed),
            "failed_runs": len(selected) - len(completed),
            "seeds_completed": ",".join(str(row["seed"]) for row in completed),
        }
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in completed if row[metric] != ""]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else ""
            summary[f"{metric}_std_population"] = (
                statistics.pstdev(values) if values else ""
            )
            summary[f"{metric}_std_sample"] = (
                statistics.stdev(values) if len(values) > 1 else ""
            )
        summary_rows.append(summary)
    _atomic_write_csv(
        output_root / "summary.csv",
        summary_rows,
        tuple(summary_rows[0].keys()),
    )


def execute_task(
    task: Mapping[str, object],
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, object]:
    task_dir = output_root / "tasks" / str(task["task_id"])
    trainer_output_root = task_dir / "trainer_results"
    logs_dir = task_dir / "logs"
    trainer_output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    parameters = complete_training_parameters(task, args)
    fingerprint = _fingerprint(parameters)
    command = build_command(task, args, trainer_output_root)
    _atomic_write_json(
        task_dir / "config.json",
        {
            **dict(task),
            "fingerprint": fingerprint,
            "complete_training_parameters": parameters,
            "command_argv": command,
            "command_shell": shlex.join(command),
        },
    )
    (task_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")

    existing = resume_row(task_dir, task, fingerprint)
    if existing is not None:
        print(f"{_utc_timestamp()} | {task['task_id']}: complete; skipping", flush=True)
        return existing

    gpu = wait_for_eligible_gpu(args, output_root, str(task["task_id"]))
    attempt = next_attempt(logs_dir)
    attempt_log = logs_dir / f"train_attempt_{attempt:03d}.log"
    running = {
        "status": "running",
        "fingerprint": fingerprint,
        "attempt": attempt,
        "started_at": _utc_timestamp(),
        "gpu_physical_index": gpu.physical_index,
        "gpu_uuid": gpu.uuid,
        "gpu_free_mib_at_start": gpu.free_mib,
        "gpu_utilization_percent_at_start": gpu.utilization_percent,
        "cuda_visible_devices_for_child": gpu.visibility_token,
        "log_path": str(attempt_log),
    }
    _atomic_write_json(task_dir / "status.json", running)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.visibility_token
    try:
        return_code = run_child(command, attempt_log, environment)
    except KeyboardInterrupt:
        shutil.copyfile(attempt_log, task_dir / "train.log")
        _atomic_write_json(
            task_dir / "status.json",
            {
                **running,
                "status": "interrupted",
                "finished_at": _utc_timestamp(),
                "error": "Interrupted by user",
            },
        )
        raise
    except Exception as exc:
        return_code = None
        launch_error = f"{type(exc).__name__}: {exc}"
    else:
        launch_error = ""
    if attempt_log.exists():
        shutil.copyfile(attempt_log, task_dir / "train.log")

    row = base_row(task, "failed")
    row["gpu_physical_index"] = gpu.physical_index
    if return_code == 0 and not launch_error:
        try:
            suite_dir, row, raw = find_suite(trainer_output_root, task)
            row["gpu_physical_index"] = gpu.physical_index
            _atomic_write_json(
                task_dir / "status.json",
                {
                    **running,
                    "status": "completed",
                    "finished_at": _utc_timestamp(),
                    "return_code": 0,
                    "trainer_suite_dir": str(suite_dir),
                    "trainer_run_dir": raw["run_dir"],
                    "result": row,
                },
            )
            return row
        except Exception as exc:
            error = f"Result extraction failed: {type(exc).__name__}: {exc}"
    elif launch_error:
        error = f"Training launch failed: {launch_error}"
    else:
        error = f"module_ablation.py exited with return code {return_code}"
    row["error"] = error
    _atomic_write_json(
        task_dir / "status.json",
        {
            **running,
            "status": "failed",
            "finished_at": _utc_timestamp(),
            "return_code": return_code,
            "error": error,
        },
    )
    return row


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    tasks = build_tasks(args.seeds)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {
        "plan_version": 1,
        "created_at": _utc_timestamp(),
        "task_count": len(tasks),
        "execution": "serial; GPU admission checked before every modality/seed task",
        "protocol": {
            "training": "supervised",
            "domain_adaptation": "Sinkhorn OT; no Tensor in single-modality runs",
            "feature_dimension": 512,
            "checkpoint_selection": "target_val, accuracy then macro-F1, epoch >= 15",
            "target_test": (
                "evaluated once after loading the validation-selected checkpoint"
                if args.evaluate_test
                else "disabled"
            ),
        },
        "gpu_admission": {
            "minimum_free_gib": args.gpu_min_free_gb,
            "maximum_utilization_percent_exclusive": args.gpu_max_utilization,
            "poll_seconds": args.gpu_poll_seconds,
            "external_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "tasks": tasks,
    }
    _atomic_write_json(output_root / "plan.json", plan)

    if args.dry_run:
        for task in tasks:
            command = build_command(
                task,
                args,
                output_root / "tasks" / str(task["task_id"]) / "trainer_results",
            )
            print(
                f"[{task['task_index']:02d}/{len(tasks):02d}] "
                f"{task['task_id']}: {shlex.join(command)}"
            )
        print(
            f"DRY RUN PASS: {len(tasks)} independent tasks, modalities={MODALITIES}, "
            "GPU admission before every task; AIS input is JSON heading only."
        )
        return 0

    if not Path(args.dataset_root).expanduser().is_dir():
        parser.error(f"Dataset root not found: {args.dataset_root}")
    if args.pretrained_weights and not Path(args.pretrained_weights).expanduser().is_file():
        parser.error(f"Pretrained weights not found: {args.pretrained_weights}")

    rows_by_task: Dict[str, Mapping[str, object]] = {}
    try:
        for task in tasks:
            row = execute_task(task, args, output_root)
            rows_by_task[str(task["task_id"])] = row
            write_tables(output_root, rows_by_task, tasks)
    except KeyboardInterrupt:
        write_tables(output_root, rows_by_task, tasks)
        return 130
    failures = [row for row in rows_by_task.values() if row["status"] != "completed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
