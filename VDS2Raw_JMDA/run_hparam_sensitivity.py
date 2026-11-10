"""Serial one-factor hyperparameter sensitivity runner for VDS2Raw JMDA-Net.

This wrapper deliberately delegates every training run to ``module_ablation.py``.
It only owns the scan plan, GPU admission control, resumability, and result
collection.  Target-test evaluation is disabled for every configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
MODULE_SCRIPT = PACKAGE_DIR / "module_ablation.py"

DEFAULT_DATASET_ROOT = "/home/lixiang/lx/VDS2Raw/threeclass_ready"
DEFAULT_OUTPUT_ROOT = (
    "/home/lixiang/lx/JMDA/VDS2Raw_JMDA/runs/hparam_sensitivity_refined"
)

SCAN_VALUES: Mapping[str, Sequence[object]] = {
    "tensor_loss_weight": (0.08, 0.10, 0.12, 0.14, 0.16),
    "adv_loss_weight": (0.04, 0.06, 0.08, 0.10, 0.12),
    "ot_correction_reg_weight": (1e-4, 5e-4, 1e-3, 5e-3, 1e-2),
    "projection_dim": (16, 32, 48, 64, 80),
}

BASELINE_VALUES: Mapping[str, object] = {
    "tensor_loss_weight": 0.12,
    "adv_loss_weight": 0.08,
    "ot_correction_reg_weight": 1e-3,
    "projection_dim": 64,
    "ot_epsilon": 0.10,
}

FIXED_TRAINING_PARAMETERS: Mapping[str, object] = {
    "training_protocol": "supervised",
    "ablation_mode": "full",
    "seeds": "42",
    "epochs": 100,
    "batch_size": 32,
    "num_workers": 4,
    "device": "cuda",
    "transport_mode": "sinkhorn",
    "nir_projection_dim": 32,
    "ais_projection_dim": 32,
    "svd_max_sweeps": 10,
    "svd_tolerance": 1e-3,
    "svd_stat_batch_size": 128,
    "lr_rgb": 5e-6,
    "lr_other": 1e-4,
    "lr_discriminator": 5e-5,
    "weight_decay": 5e-4,
    "discriminator_update_interval": 2,
    "freeze_rgb_backbone": True,
    "balance_classes": True,
    "image_augmentation": True,
    "ot_sinkhorn_iterations": 50,
    "ot_correction_scale": 0.1,
    "ot_warmup_epochs": 10,
    "ot_ramp_epochs": 5,
    "evaluate_test": False,
}

RESULT_FIELDS = (
    "config_index",
    "config_id",
    "status",
    "is_baseline",
    "scan_parameter",
    "scan_value",
    "seed",
    "tensor_loss_weight",
    "adv_loss_weight",
    "ot_correction_reg_weight",
    "ot_epsilon",
    "rgb_projection_dim",
    "nir_projection_dim",
    "ais_projection_dim",
    "selected_epoch",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "last10_accuracy_mean",
    "last10_accuracy_std",
    "last10_epoch_count",
    "gpu_physical_index",
    "trainer_suite_dir",
    "trainer_run_dir",
    "error",
)


@dataclass(frozen=True)
class ScanConfiguration:
    index: int
    config_id: str
    is_baseline: bool
    scan_parameter: str
    scan_value: object
    values: Mapping[str, object]

    def serializable(self) -> Dict[str, object]:
        return {
            "config_index": self.index,
            "config_id": self.config_id,
            "is_baseline": self.is_baseline,
            "scan_parameter": self.scan_parameter,
            "scan_value": self.scan_value,
            "scan_values": dict(self.values),
            "actual_projection_dimensions": {
                "rgb": int(self.values["projection_dim"]),
                "nir": int(FIXED_TRAINING_PARAMETERS["nir_projection_dim"]),
                "ais": int(FIXED_TRAINING_PARAMETERS["ais_projection_dim"]),
            },
        }


@dataclass(frozen=True)
class GPUState:
    physical_index: str
    uuid: str
    free_mib: int
    utilization_percent: float
    visibility_token: str


def _value_slug(value: object) -> str:
    text = format(value, ".12g") if isinstance(value, float) else str(value)
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def build_scan_plan() -> List[ScanConfiguration]:
    plan: List[ScanConfiguration] = [
        ScanConfiguration(
            index=1,
            config_id="001_baseline",
            is_baseline=True,
            scan_parameter="baseline_shared",
            scan_value="",
            values=dict(BASELINE_VALUES),
        )
    ]
    index = 2
    for parameter, values in SCAN_VALUES.items():
        baseline = BASELINE_VALUES[parameter]
        for value in values:
            if value == baseline:
                continue
            plan.append(
                ScanConfiguration(
                    index=index,
                    config_id=(
                        f"{index:03d}_{parameter}_{_value_slug(value)}"
                    ),
                    is_baseline=False,
                    scan_parameter=parameter,
                    scan_value=value,
                    values={**BASELINE_VALUES, parameter: value},
                )
            )
            index += 1
    validate_scan_plan(plan)
    return plan


def validate_scan_plan(plan: Sequence[ScanConfiguration]) -> None:
    if len(plan) != 17:
        raise AssertionError(f"Expected 17 unique configurations, found {len(plan)}")
    if len({item.config_id for item in plan}) != len(plan):
        raise AssertionError("Configuration identifiers are not unique")
    baselines = [item for item in plan if item.is_baseline]
    if len(baselines) != 1 or dict(baselines[0].values) != dict(BASELINE_VALUES):
        raise AssertionError("The shared baseline must occur exactly once")

    expected_nonbaseline_counts = {
        parameter: len(values) - 1 for parameter, values in SCAN_VALUES.items()
    }
    actual_counts = {parameter: 0 for parameter in SCAN_VALUES}
    for item in plan:
        if item.is_baseline:
            continue
        differences = [
            parameter
            for parameter, baseline in BASELINE_VALUES.items()
            if item.values[parameter] != baseline
        ]
        if differences != [item.scan_parameter]:
            raise AssertionError(
                f"{item.config_id} does not change exactly its declared parameter: "
                f"differences={differences}"
            )
        if item.scan_value not in SCAN_VALUES[item.scan_parameter]:
            raise AssertionError(f"Unexpected scan value in {item.config_id}")
        actual_counts[item.scan_parameter] += 1
    if actual_counts != expected_nonbaseline_counts:
        raise AssertionError(
            f"Incorrect per-parameter task counts: {actual_counts}; "
            f"expected {expected_nonbaseline_counts}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 17-configuration supervised VDS2Raw one-factor "
            "hyperparameter sensitivity suite serially"
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
        help="Offline resnet18-f37072fd.pth path forwarded to module_ablation.py",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for child training processes",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpu-min-free-gb",
        type=float,
        default=15.0,
        help="Minimum free GPU memory in GiB before a run may start",
    )
    parser.add_argument(
        "--gpu-max-utilization",
        type=float,
        default=15.0,
        help="Required utilization is strictly lower than this percentage",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=float,
        default=60.0,
        help="Seconds between GPU availability checks",
    )
    return parser


def validate_wrapper_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.gpu_min_free_gb < 0:
        parser.error("--gpu-min-free-gb cannot be negative")
    if not 0 < args.gpu_max_utilization <= 100:
        parser.error("--gpu-max-utilization must be in (0, 100]")
    if args.gpu_poll_seconds <= 0:
        parser.error("--gpu-poll-seconds must be positive")
    if args.pretrained_weights is not None and not str(args.pretrained_weights).strip():
        parser.error("--pretrained-weights cannot be empty")


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_write_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _utc_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_parameter_payload(
    config: ScanConfiguration, args: argparse.Namespace
) -> Dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root),
        "pretrained_weights": (
            str(args.pretrained_weights) if args.pretrained_weights is not None else None
        ),
        **dict(FIXED_TRAINING_PARAMETERS),
        **dict(config.values),
    }


def build_training_command(
    config: ScanConfiguration,
    args: argparse.Namespace,
    trainer_output_root: Path,
) -> List[str]:
    values = config.values
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
        "full",
        "--seeds",
        "42",
        "--epochs",
        "100",
        "--batch_size",
        "32",
        "--num_workers",
        "4",
        "--device",
        "cuda",
        "--transport_mode",
        "sinkhorn",
        "--projection_dim",
        str(values["projection_dim"]),
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
        str(values["tensor_loss_weight"]),
        "--adv_loss_weight",
        str(values["adv_loss_weight"]),
        "--discriminator_update_interval",
        "2",
        "--freeze_rgb_backbone",
        "--balance_classes",
        "--image_augmentation",
        "--ot_epsilon",
        str(values["ot_epsilon"]),
        "--ot_sinkhorn_iterations",
        "50",
        "--ot_correction_scale",
        "0.1",
        "--ot_correction_reg_weight",
        str(values["ot_correction_reg_weight"]),
        "--ot_warmup_epochs",
        "10",
        "--ot_ramp_epochs",
        "5",
        "--no-evaluate_test",
    ]
    if args.pretrained_weights is not None:
        command.extend(["--pretrained_weights", str(args.pretrained_weights)])
    return command


def _plan_document(
    plan: Sequence[ScanConfiguration], args: argparse.Namespace
) -> Dict[str, object]:
    return {
        "plan_version": 1,
        "created_at": _utc_timestamp(),
        "configuration_count": len(plan),
        "execution": "serial; seed 42 once per configuration",
        "selection_and_reporting": {
            "checkpoint_split": "target_val",
            "checkpoint_rule": "inherited from module_ablation.py",
            "target_test_evaluation": False,
            "cross_seed_statistics": False,
        },
        "baseline_reused_for": list(SCAN_VALUES.keys()),
        "scan_values": {key: list(value) for key, value in SCAN_VALUES.items()},
        "baseline_values": dict(BASELINE_VALUES),
        "fixed_training_parameters": dict(FIXED_TRAINING_PARAMETERS),
        "paths": {
            "dataset_root": str(args.dataset_root),
            "output_root": str(Path(args.output_root).expanduser()),
            "pretrained_weights": args.pretrained_weights,
            "module_script": str(MODULE_SCRIPT),
            "python_executable": str(args.python_executable),
        },
        "gpu_admission": {
            "minimum_free_gib": args.gpu_min_free_gb,
            "maximum_utilization_percent_exclusive": args.gpu_max_utilization,
            "poll_seconds": args.gpu_poll_seconds,
            "external_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "configurations": [item.serializable() for item in plan],
    }


def _plan_csv_rows(plan: Sequence[ScanConfiguration]) -> List[Dict[str, object]]:
    rows = []
    for item in plan:
        rows.append(
            {
                "config_index": item.index,
                "config_id": item.config_id,
                "is_baseline": item.is_baseline,
                "scan_parameter": item.scan_parameter,
                "scan_value": item.scan_value,
                "tensor_loss_weight": item.values["tensor_loss_weight"],
                "adv_loss_weight": item.values["adv_loss_weight"],
                "ot_correction_reg_weight": item.values[
                    "ot_correction_reg_weight"
                ],
                "ot_epsilon": item.values["ot_epsilon"],
                "rgb_projection_dim": item.values["projection_dim"],
                "nir_projection_dim": FIXED_TRAINING_PARAMETERS[
                    "nir_projection_dim"
                ],
                "ais_projection_dim": FIXED_TRAINING_PARAMETERS[
                    "ais_projection_dim"
                ],
            }
        )
    return rows


def _append_scheduler_log(output_root: Path, message: str) -> None:
    timestamped = f"{_utc_timestamp()} | {message}"
    print(timestamped, flush=True)
    with (output_root / "scheduler.log").open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")


def _visible_gpu_tokens() -> Optional[List[str]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens or tokens == ["-1"]:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES exposes no GPU; expose at least one server GPU"
        )
    return tokens


def query_gpu_states() -> List[GPUState]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi query failed: {detail}") from exc

    available: List[Tuple[str, str, int, float]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise RuntimeError(f"Unexpected nvidia-smi output line: {line!r}")
        try:
            available.append((fields[0], fields[1], int(fields[2]), float(fields[3])))
        except ValueError as exc:
            raise RuntimeError(f"Invalid nvidia-smi numeric field: {line!r}") from exc

    restrictions = _visible_gpu_tokens()
    states: List[GPUState] = []
    for physical_index, uuid, free_mib, utilization in available:
        if restrictions is None:
            visibility_token = physical_index
        else:
            matching = [
                token
                for token in restrictions
                if token == physical_index
                or token == uuid
                or (token.startswith("GPU-") and uuid.startswith(token))
            ]
            if not matching:
                continue
            visibility_token = matching[0]
        states.append(
            GPUState(
                physical_index=physical_index,
                uuid=uuid,
                free_mib=free_mib,
                utilization_percent=utilization,
                visibility_token=visibility_token,
            )
        )
    if restrictions is not None and not states:
        raise RuntimeError(
            "None of CUDA_VISIBLE_DEVICES="
            f"{','.join(restrictions)} matched an nvidia-smi GPU index or UUID"
        )
    if not states:
        raise RuntimeError("nvidia-smi reported no GPUs")
    return states


def wait_for_eligible_gpu(
    args: argparse.Namespace, output_root: Path, config_id: str
) -> GPUState:
    minimum_free_mib = int(args.gpu_min_free_gb * 1024)
    while True:
        states = query_gpu_states()
        eligible = [
            state
            for state in states
            if state.free_mib >= minimum_free_mib
            and state.utilization_percent < args.gpu_max_utilization
        ]
        if eligible:
            selected = max(
                eligible,
                key=lambda state: (
                    state.free_mib,
                    -state.utilization_percent,
                    -int(state.physical_index),
                ),
            )
            _append_scheduler_log(
                output_root,
                f"{config_id}: selected physical GPU {selected.physical_index} "
                f"(free={selected.free_mib} MiB, "
                f"utilization={selected.utilization_percent:g}%, "
                f"visibility_token={selected.visibility_token})",
            )
            return selected

        state_text = "; ".join(
            f"GPU {state.physical_index}: free={state.free_mib} MiB, "
            f"utilization={state.utilization_percent:g}%"
            for state in states
        )
        _append_scheduler_log(
            output_root,
            f"{config_id}: waiting for free>={minimum_free_mib} MiB and "
            f"utilization<{args.gpu_max_utilization:g}%; {state_text}",
        )
        time.sleep(args.gpu_poll_seconds)


def _read_single_metrics_row(metrics_path: Path) -> Dict[str, str]:
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row in {metrics_path}, found {len(rows)}"
        )
    row = rows[0]
    if row.get("training_protocol") != "supervised":
        raise RuntimeError("metrics_by_run.csv is not a supervised run")
    if row.get("ablation_name") != "full" or row.get("seed") != "42":
        raise RuntimeError("metrics_by_run.csv is not full/seed-42")
    required = (
        "best_epoch",
        "val_accuracy",
        "val_precision_macro",
        "val_recall_macro",
        "val_f1_macro",
        "run_dir",
    )
    missing = [field for field in required if not str(row.get(field, "")).strip()]
    if missing:
        raise RuntimeError(f"Missing selected-validation fields: {missing}")
    return row


def _find_completed_suite(trainer_output_root: Path) -> Tuple[Path, Dict[str, str]]:
    candidates = sorted(
        trainer_output_root.glob("vds2raw_*/metrics_by_run.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    errors = []
    for metrics_path in candidates:
        try:
            return metrics_path.parent, _read_single_metrics_row(metrics_path)
        except Exception as exc:  # preserve diagnostics and inspect older attempts
            errors.append(f"{metrics_path}: {exc}")
    suffix = f" Invalid candidates: {'; '.join(errors)}" if errors else ""
    raise RuntimeError(
        f"No complete metrics_by_run.csv found under {trainer_output_root}.{suffix}"
    )


def _base_result_row(config: ScanConfiguration, status: str) -> Dict[str, object]:
    return {
        "config_index": config.index,
        "config_id": config.config_id,
        "status": status,
        "is_baseline": config.is_baseline,
        "scan_parameter": config.scan_parameter,
        "scan_value": config.scan_value,
        "seed": 42,
        "tensor_loss_weight": config.values["tensor_loss_weight"],
        "adv_loss_weight": config.values["adv_loss_weight"],
        "ot_correction_reg_weight": config.values["ot_correction_reg_weight"],
        "ot_epsilon": config.values["ot_epsilon"],
        "rgb_projection_dim": config.values["projection_dim"],
        "nir_projection_dim": FIXED_TRAINING_PARAMETERS["nir_projection_dim"],
        "ais_projection_dim": FIXED_TRAINING_PARAMETERS["ais_projection_dim"],
        "selected_epoch": "",
        "val_accuracy": "",
        "val_precision_macro": "",
        "val_recall_macro": "",
        "val_f1_macro": "",
        "last10_accuracy_mean": "",
        "last10_accuracy_std": "",
        "last10_epoch_count": "",
        "gpu_physical_index": "",
        "trainer_suite_dir": "",
        "trainer_run_dir": "",
        "error": "",
    }


def _result_from_metrics(
    config: ScanConfiguration,
    suite_dir: Path,
    metrics: Mapping[str, str],
    gpu_index: object = "",
) -> Dict[str, object]:
    last_window = _read_last_window_accuracy(suite_dir, window=10)
    row = _base_result_row(config, "completed")
    row.update(
        {
            "selected_epoch": int(metrics["best_epoch"]),
            "val_accuracy": float(metrics["val_accuracy"]),
            "val_precision_macro": float(metrics["val_precision_macro"]),
            "val_recall_macro": float(metrics["val_recall_macro"]),
            "val_f1_macro": float(metrics["val_f1_macro"]),
            **last_window,
            "gpu_physical_index": gpu_index,
            "trainer_suite_dir": str(suite_dir),
            "trainer_run_dir": metrics["run_dir"],
        }
    )
    return row


def _read_last_window_accuracy(
    suite_dir: Path, window: int = 10
) -> Dict[str, object]:
    metrics_path = suite_dir / "full" / "seed_42" / "metrics.csv"
    if not metrics_path.is_file():
        raise RuntimeError(f"Per-epoch metrics are missing: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    usable = []
    for row in rows:
        try:
            usable.append((int(row["epoch"]), float(row["val_accuracy"])))
        except (KeyError, TypeError, ValueError):
            continue
    usable.sort(key=lambda item: item[0])
    selected = usable[-window:]
    if len(selected) != window:
        raise RuntimeError(
            f"Expected {window} final epoch ACC values in {metrics_path}, "
            f"found {len(selected)}"
        )
    values = [value for _, value in selected]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "last10_accuracy_mean": mean,
        "last10_accuracy_std": variance ** 0.5,
        "last10_epoch_count": len(values),
    }


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _completed_resume_row(
    config: ScanConfiguration,
    task_dir: Path,
    expected_fingerprint: str,
) -> Optional[Dict[str, object]]:
    status = _load_json(task_dir / "status.json")
    if (
        status is None
        or status.get("status") != "completed"
        or status.get("fingerprint") != expected_fingerprint
    ):
        return None
    try:
        suite_dir = Path(str(status["trainer_suite_dir"]))
        metrics = _read_single_metrics_row(suite_dir / "metrics_by_run.csv")
        return _result_from_metrics(
            config,
            suite_dir,
            metrics,
            status.get("gpu_physical_index", ""),
        )
    except Exception:
        return None


def _next_attempt_number(logs_dir: Path) -> int:
    existing = list(logs_dir.glob("train_attempt_*.log")) if logs_dir.exists() else []
    numbers = []
    for path in existing:
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(numbers, default=0) + 1


def _append_ot_residuals(log_path: Path, trainer_output_root: Path) -> None:
    metric_files = sorted(
        trainer_output_root.glob("vds2raw_*/full/seed_42/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n=== OT marginal residuals from trainer metrics.csv ===\n")
        if not metric_files:
            log_handle.write("No per-epoch metrics.csv was produced.\n")
            return
        metrics_path = metric_files[-1]
        log_handle.write(f"source={metrics_path}\n")
        try:
            with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                log_handle.write(
                    f"epoch={row.get('epoch', '')} "
                    f"ot_marginal_residual={row.get('ot_marginal_residual', '')}\n"
                )
        except Exception as exc:
            # Log enrichment must never hide the original trainer outcome.
            log_handle.write(
                f"Unable to read OT residual history: {type(exc).__name__}: {exc}\n"
            )


def _run_child(command: Sequence[str], log_path: Path, environment: Mapping[str, str]) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"started_at={_utc_timestamp()}\n")
        log_handle.write(f"command={shlex.join(command)}\n\n")
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
                log_handle.write(line)
            return process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def _write_aggregate_tables(
    output_root: Path,
    rows_by_id: Mapping[str, Mapping[str, object]],
    plan: Sequence[ScanConfiguration],
) -> None:
    rows = [rows_by_id[item.config_id] for item in plan if item.config_id in rows_by_id]
    _atomic_write_csv(output_root / "per_seed.csv", rows, RESULT_FIELDS)

    # Keep 17 actual-run rows in per_seed.csv.  For direct sensitivity plotting,
    # summary.csv reuses the single baseline result once in each of the four
    # parameter groups (20 plotting points, still only 17 trained configs).
    summary_rows: List[Mapping[str, object]] = []
    for row in rows:
        if not bool(row["is_baseline"]):
            summary_rows.append(row)
            continue
        for parameter in SCAN_VALUES:
            reused = dict(row)
            reused["scan_parameter"] = parameter
            reused["scan_value"] = BASELINE_VALUES[parameter]
            summary_rows.append(reused)
    _atomic_write_csv(output_root / "summary.csv", summary_rows, RESULT_FIELDS)


def execute_configuration(
    config: ScanConfiguration,
    args: argparse.Namespace,
    output_root: Path,
) -> Dict[str, object]:
    task_dir = output_root / "configs" / config.config_id
    trainer_output_root = task_dir / "trainer_results"
    logs_dir = task_dir / "logs"
    task_dir.mkdir(parents=True, exist_ok=True)
    trainer_output_root.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    parameter_payload = _training_parameter_payload(config, args)
    fingerprint = _fingerprint(parameter_payload)
    command = build_training_command(config, args, trainer_output_root)
    _atomic_write_json(
        task_dir / "config.json",
        {
            **config.serializable(),
            "fingerprint": fingerprint,
            "complete_training_parameters": parameter_payload,
            "fixed_training_parameters": dict(FIXED_TRAINING_PARAMETERS),
            "command_argv": command,
            "command_shell": shlex.join(command),
        },
    )
    (task_dir / "command.txt").write_text(
        shlex.join(command) + "\n", encoding="utf-8"
    )

    resumed = _completed_resume_row(config, task_dir, fingerprint)
    if resumed is not None:
        _append_scheduler_log(output_root, f"{config.config_id}: complete; skipping")
        return resumed

    gpu = wait_for_eligible_gpu(args, output_root, config.config_id)
    attempt = _next_attempt_number(logs_dir)
    attempt_log = logs_dir / f"train_attempt_{attempt:03d}.log"
    started_at = _utc_timestamp()
    running_status = {
        "status": "running",
        "fingerprint": fingerprint,
        "attempt": attempt,
        "started_at": started_at,
        "gpu_physical_index": gpu.physical_index,
        "gpu_uuid": gpu.uuid,
        "gpu_free_mib_at_start": gpu.free_mib,
        "gpu_utilization_percent_at_start": gpu.utilization_percent,
        "cuda_visible_devices_for_child": gpu.visibility_token,
        "log_path": str(attempt_log),
    }
    _atomic_write_json(task_dir / "status.json", running_status)

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu.visibility_token
    return_code: Optional[int] = None
    error = ""
    try:
        return_code = _run_child(command, attempt_log, environment)
    except KeyboardInterrupt:
        error = "Interrupted by user"
        _append_ot_residuals(attempt_log, trainer_output_root)
        shutil.copyfile(attempt_log, task_dir / "train.log")
        _atomic_write_json(
            task_dir / "status.json",
            {
                **running_status,
                "status": "interrupted",
                "finished_at": _utc_timestamp(),
                "error": error,
            },
        )
        raise
    except Exception as exc:
        error = f"Training launch failed: {type(exc).__name__}: {exc}"

    _append_ot_residuals(attempt_log, trainer_output_root)
    shutil.copyfile(attempt_log, task_dir / "train.log")

    row = _base_result_row(config, "failed")
    row["gpu_physical_index"] = gpu.physical_index
    if return_code == 0 and not error:
        try:
            suite_dir, metrics = _find_completed_suite(trainer_output_root)
            row = _result_from_metrics(
                config, suite_dir, metrics, gpu.physical_index
            )
            _atomic_write_json(
                task_dir / "status.json",
                {
                    **running_status,
                    "status": "completed",
                    "finished_at": _utc_timestamp(),
                    "return_code": return_code,
                    "trainer_suite_dir": str(suite_dir),
                    "trainer_run_dir": metrics["run_dir"],
                    "result": row,
                },
            )
            _append_scheduler_log(output_root, f"{config.config_id}: completed")
            return row
        except Exception as exc:
            error = f"Result extraction failed: {type(exc).__name__}: {exc}"
    elif not error:
        error = f"module_ablation.py exited with return code {return_code}"

    row["error"] = error
    _atomic_write_json(
        task_dir / "status.json",
        {
            **running_status,
            "status": "failed",
            "finished_at": _utc_timestamp(),
            "return_code": return_code,
            "error": error,
        },
    )
    _append_scheduler_log(output_root, f"{config.config_id}: FAILED: {error}")
    return row


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_wrapper_args(parser, args)
    plan = build_scan_plan()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    _atomic_write_json(output_root / "plan.json", _plan_document(plan, args))
    plan_rows = _plan_csv_rows(plan)
    _atomic_write_csv(
        output_root / "plan.csv", plan_rows, tuple(plan_rows[0].keys())
    )

    if args.dry_run:
        for config in plan:
            task_dir = output_root / "configs" / config.config_id
            trainer_output_root = task_dir / "trainer_results"
            command = build_training_command(config, args, trainer_output_root)
            print(
                f"[{config.index:02d}/{len(plan):02d}] {config.config_id}: "
                f"{shlex.join(command)}"
            )
        print(
            "DRY RUN PASS: one shared baseline, 17 unique configurations, "
            "and exactly one changed scan parameter per non-baseline configuration."
        )
        print(f"Plan saved to: {output_root}")
        return 0

    if not MODULE_SCRIPT.is_file():
        parser.error(f"Training entry point not found: {MODULE_SCRIPT}")
    if not Path(args.dataset_root).expanduser().is_dir():
        parser.error(f"Dataset root not found: {args.dataset_root}")
    if args.pretrained_weights is not None and not Path(
        args.pretrained_weights
    ).expanduser().is_file():
        parser.error(f"Pretrained weights not found: {args.pretrained_weights}")

    rows_by_id: Dict[str, Mapping[str, object]] = {}
    _append_scheduler_log(
        output_root,
        "starting supervised sensitivity suite: 17 serial configurations, seed=42, "
        "selection=target_val, target_test=disabled",
    )
    try:
        for config in plan:
            row = execute_configuration(config, args, output_root)
            rows_by_id[config.config_id] = row
            _write_aggregate_tables(output_root, rows_by_id, plan)
    except KeyboardInterrupt:
        _write_aggregate_tables(output_root, rows_by_id, plan)
        _append_scheduler_log(output_root, "suite interrupted by user")
        return 130

    failed = [row for row in rows_by_id.values() if row["status"] != "completed"]
    _append_scheduler_log(
        output_root,
        f"suite finished: completed={len(rows_by_id) - len(failed)}, "
        f"failed={len(failed)}",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
