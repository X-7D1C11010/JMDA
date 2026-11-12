#!/usr/bin/env python3
"""Serial one-factor hyperparameter sensitivity runner for JMDA-Net.

This wrapper intentionally launches ``module_ablation.py`` as a subprocess so
the existing dataset construction, supervised training, validation selection,
and metric reporting logic remain unchanged.  It adds only experiment-plan
validation, GPU admission control, per-task artifacts, resume support, and CSV
aggregation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


BASELINE: Dict[str, object] = {
    "tensor_loss_weight": 0.12,
    "adv_loss_weight": 0.08,
    "domain_loss_weight": 1.0,
    "ot_correction_reg_weight": 1e-3,
    "projection_dim": 64,
    "ot_epsilon": 0.10,
}

REFINED_SCAN_VALUES: Mapping[str, Sequence[object]] = {
    "tensor_loss_weight": (0.08, 0.10, 0.12, 0.14, 0.16),
    "adv_loss_weight": (0.04, 0.06, 0.08, 0.10, 0.12),
    "ot_correction_reg_weight": (1e-4, 5e-4, 1e-3, 5e-3, 1e-2),
    "projection_dim": (16, 32, 48, 64, 80),
}

LAMBDA1_SCAN_VALUES: Mapping[str, Sequence[object]] = {
    "domain_loss_weight": (0.06, 0.08, 0.10, 0.12, 0.14, 1.0),
}

SCAN_PROFILES: Mapping[str, Mapping[str, Sequence[object]]] = {
    "refined": REFINED_SCAN_VALUES,
    "lambda1": LAMBDA1_SCAN_VALUES,
}

SCAN_VALUES: Mapping[str, Sequence[object]] = REFINED_SCAN_VALUES
SCAN_PARAMETERS: Tuple[str, ...] = tuple(SCAN_VALUES)

# Every trainer-facing method hyperparameter is passed explicitly.  A scan
# profile controls only which one-factor configurations are created.
TRAINING_HYPERPARAMETERS: Tuple[str, ...] = (
    "tensor_loss_weight",
    "adv_loss_weight",
    "domain_loss_weight",
    "ot_correction_reg_weight",
    "projection_dim",
    "ot_epsilon",
)

FIXED_TRAINING_ARGUMENTS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("--ablation_mode", "full"),
    ("--use_target_labels", None),
    ("--no_auto_ablation_hparams", None),
    ("--transport_mode", "sinkhorn"),
    ("--batch_size", "32"),
    ("--epochs", "100"),
    ("--num_iterations", "1"),
    ("--lr_feature", "5e-6"),
    ("--lr_other", "2e-4"),
    ("--weight_decay", "5e-4"),
    ("--report_strategy", "best"),
    ("--svd_max_sweeps", "10"),
    ("--svd_tolerance", "1e-3"),
    ("--svd_stat_batch_size", "128"),
    ("--ot_sinkhorn_iterations", "50"),
    ("--ot_correction_scale", "0.1"),
    ("--ot_warmup_epochs", "10"),
    ("--ot_ramp_epochs", "5"),
)

METRIC_FIELDS: Tuple[str, ...] = (
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
)

LAST_WINDOW_FIELDS: Tuple[str, ...] = (
    "last10_accuracy_mean",
    "last10_accuracy_std",
    "last10_epoch_count",
)

CSV_FIELDS: Tuple[str, ...] = (
    "configuration",
    "scan_parameter",
    "scan_value",
    "target_weather",
    "target_root",
    "seed",
    "domain_loss_weight",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "last10_accuracy_mean",
    "last10_accuracy_std",
    "last10_epoch_count",
    "status",
    "return_code",
    "gpu_id",
    "task_directory",
    "updated_at",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def value_token(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    value_text = format(float(value), ".10g")
    return value_text.replace("-", "m").replace("+", "").replace(".", "p")


def display_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    return format(float(value), ".10g")


def slugify(text: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", text.strip())
    safe = safe.strip("_")
    return safe or "target"


def values_equal(left: object, right: object) -> bool:
    if isinstance(left, int) and isinstance(right, int):
        return left == right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def activate_scan_profile(profile: str) -> None:
    global SCAN_VALUES, SCAN_PARAMETERS
    SCAN_VALUES = SCAN_PROFILES[profile]
    SCAN_PARAMETERS = tuple(SCAN_VALUES)


def build_configurations() -> List[Dict[str, object]]:
    configurations: List[Dict[str, object]] = [
        {
            "name": "baseline",
            "scan_parameter": "baseline",
            "scan_value": "baseline",
            "parameters": dict(BASELINE),
        }
    ]

    for parameter in SCAN_PARAMETERS:
        for value in SCAN_VALUES[parameter]:
            if values_equal(value, BASELINE[parameter]):
                continue
            parameters = dict(BASELINE)
            parameters[parameter] = value
            configurations.append(
                {
                    "name": f"{parameter}_{value_token(value)}",
                    "scan_parameter": parameter,
                    "scan_value": display_value(value),
                    "parameters": parameters,
                }
            )
    return configurations


def validate_configurations(configurations: Sequence[Mapping[str, object]]) -> None:
    expected_count = 1 + sum(
        1
        for parameter in SCAN_PARAMETERS
        for value in SCAN_VALUES[parameter]
        if not values_equal(value, BASELINE[parameter])
    )
    if len(configurations) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} unique configurations, "
            f"found {len(configurations)}."
        )

    names = [str(config["name"]) for config in configurations]
    if len(set(names)) != len(names):
        raise RuntimeError("Configuration names are not unique.")
    if names.count("baseline") != 1:
        raise RuntimeError("The shared baseline must appear exactly once.")

    signatures = set()
    for config in configurations:
        parameters = config["parameters"]
        if not isinstance(parameters, Mapping):
            raise RuntimeError(f"Invalid parameter map in {config['name']}.")
        signature = canonical_json(parameters)
        if signature in signatures:
            raise RuntimeError(f"Duplicate parameter configuration: {config['name']}.")
        signatures.add(signature)

        differences = [
            parameter
            for parameter in SCAN_PARAMETERS
            if not values_equal(parameters[parameter], BASELINE[parameter])
        ]
        if config["name"] == "baseline":
            if differences:
                raise RuntimeError("The baseline differs from its declared baseline values.")
            continue
        if len(differences) != 1:
            raise RuntimeError(
                f"{config['name']} changes {len(differences)} parameters: {differences}."
            )
        if differences[0] != config["scan_parameter"]:
            raise RuntimeError(
                f"{config['name']} metadata does not match changed parameter {differences[0]}."
            )


def is_weather_domain(path: Path) -> bool:
    return path.is_dir() and all((path / phase).is_dir() for phase in ("train", "val"))


def split_target_specs(values: Sequence[str]) -> List[str]:
    specs: List[str] = []
    for value in values:
        specs.extend(part.strip() for part in value.split(",") if part.strip())
    if not specs:
        raise ValueError("At least one target weather/path must be specified.")
    return specs


def discover_targets(source_root: Path, target_specs: Sequence[str], dry_run: bool) -> List[Path]:
    data_root = source_root.parent
    source_name = source_root.name
    targets: List[Path] = []

    for spec in split_target_specs(target_specs):
        if spec.lower() == "all":
            if not data_root.is_dir():
                raise ValueError(
                    f"Cannot discover --target-root all because {data_root} does not exist."
                )
            discovered = [
                child
                for child in sorted(data_root.iterdir(), key=lambda item: item.name)
                if child.name != source_name and is_weather_domain(child)
            ]
            if not discovered:
                raise ValueError(f"No weather domains found under {data_root}.")
            targets.extend(discovered)
            continue

        candidate = Path(spec).expanduser()
        if not candidate.is_absolute():
            candidate = data_root / candidate
        if not dry_run and not is_weather_domain(candidate):
            raise ValueError(
                f"Target weather directory must contain train/ and val/: {candidate}"
            )
        targets.append(candidate)

    unique_targets: List[Path] = []
    seen = set()
    for target in targets:
        normalized = str(target.resolve(strict=False))
        if normalized not in seen:
            seen.add(normalized)
            unique_targets.append(target)
    return unique_targets


def fixed_training_cli() -> List[str]:
    result: List[str] = []
    for option, value in FIXED_TRAINING_ARGUMENTS:
        result.append(option)
        if value is not None:
            result.append(value)
    return result


def build_training_command(
    python_executable: str,
    module_path: Path,
    source_root: Path,
    target_root: Path,
    parameters: Mapping[str, object],
) -> List[str]:
    command = [python_executable, "-u", str(module_path)]
    command.extend(fixed_training_cli())
    command.extend(("--source_root", str(source_root), "--target_root", str(target_root)))
    for parameter in TRAINING_HYPERPARAMETERS:
        command.extend((f"--{parameter}", display_value(parameters[parameter])))
    return command


def command_text(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def task_signature(
    config: Mapping[str, object], source_root: Path, target_root: Path, command: Sequence[str]
) -> str:
    payload = {
        "configuration": config["name"],
        "source_root": str(source_root.resolve(strict=False)),
        "target_root": str(target_root.resolve(strict=False)),
        "command": list(command),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def build_plan(
    configurations: Sequence[Mapping[str, object]],
    targets: Sequence[Path],
    source_root: Path,
    output_dir: Path,
    module_path: Path,
    python_executable: str,
) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for config in configurations:
        for target in targets:
            target_weather = target.name
            task_dir = output_dir / str(config["name"]) / slugify(target_weather)
            command = build_training_command(
                python_executable,
                module_path,
                source_root,
                target,
                config["parameters"],
            )
            plan.append(
                {
                    "configuration": config["name"],
                    "scan_parameter": config["scan_parameter"],
                    "scan_value": config["scan_value"],
                    "parameters": dict(config["parameters"]),
                    "target_weather": target_weather,
                    "target_root": str(target),
                    "source_root": str(source_root),
                    "seed": 42,
                    "task_directory": str(task_dir),
                    "command": command,
                    "signature": task_signature(config, source_root, target, command),
                }
            )
    return plan


def parse_gpu_ids(value: str) -> List[int]:
    try:
        gpu_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("GPU IDs must be comma-separated integers.") from exc
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise argparse.ArgumentTypeError("Provide one or more unique GPU IDs.")
    return gpu_ids


def query_gpus(nvidia_smi: str) -> List[Dict[str, int]]:
    command = [
        nvidia_smi,
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"nvidia-smi failed: {message}")

    gpus: List[Dict[str, int]] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpu_id, memory_free_mib, utilization_percent = map(int, parts)
        except ValueError:
            continue
        gpus.append(
            {
                "gpu_id": gpu_id,
                "memory_free_mib": memory_free_mib,
                "utilization_percent": utilization_percent,
            }
        )
    if not gpus:
        raise RuntimeError("nvidia-smi returned no parseable GPU rows.")
    return gpus


def choose_gpu(
    gpu_rows: Iterable[Mapping[str, int]],
    allowed_gpu_ids: Sequence[int],
    minimum_free_mib: int,
    maximum_utilization_percent: int,
) -> Optional[Dict[str, int]]:
    allowed = set(allowed_gpu_ids)
    eligible = [
        dict(row)
        for row in gpu_rows
        if row["gpu_id"] in allowed
        and row["memory_free_mib"] >= minimum_free_mib
        and row["utilization_percent"] < maximum_utilization_percent
    ]
    if not eligible:
        return None
    eligible.sort(
        key=lambda row: (
            -row["memory_free_mib"],
            row["utilization_percent"],
            row["gpu_id"],
        )
    )
    return eligible[0]


def wait_for_gpu(args: argparse.Namespace, runner_log: Path) -> Dict[str, int]:
    while True:
        try:
            rows = query_gpus(args.nvidia_smi)
            selected = choose_gpu(
                rows,
                args.gpu_ids,
                args.min_free_memory_mib,
                args.max_gpu_utilization,
            )
            snapshot = "; ".join(
                f"GPU {row['gpu_id']}: free={row['memory_free_mib']}MiB, "
                f"util={row['utilization_percent']}%"
                for row in rows
                if row["gpu_id"] in set(args.gpu_ids)
            )
            if selected is not None:
                log_runner(
                    runner_log,
                    f"GPU admission passed; selected GPU {selected['gpu_id']} "
                    f"(free={selected['memory_free_mib']}MiB, "
                    f"util={selected['utilization_percent']}%). Snapshot: {snapshot}",
                )
                return selected
            log_runner(
                runner_log,
                "No GPU currently satisfies "
                f"free>={args.min_free_memory_mib}MiB and "
                f"util<{args.max_gpu_utilization}%. Snapshot: {snapshot}. "
                f"Retrying in {args.gpu_poll_seconds}s.",
            )
        except (OSError, RuntimeError) as exc:
            log_runner(
                runner_log,
                f"GPU query failed ({exc}); retrying in {args.gpu_poll_seconds}s.",
            )
        time.sleep(args.gpu_poll_seconds)


def log_runner(path: Path, message: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def parse_report_metrics(log_path: Path) -> Optional[Dict[str, float]]:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    block_matches = list(
        re.finditer(
            r"迭代\s+1\s+报告指标:\s*\n"
            r"\s*Accuracy:\s*(?P<accuracy>[0-9]+(?:\.[0-9]+)?)\s*\n"
            r"\s*Precision \(Macro\):\s*(?P<precision_macro>[0-9]+(?:\.[0-9]+)?)\s*\n"
            r"\s*Recall \(Macro\):\s*(?P<recall_macro>[0-9]+(?:\.[0-9]+)?)\s*\n"
            r"\s*F1 \(Macro\):\s*(?P<f1_macro>[0-9]+(?:\.[0-9]+)?)",
            text,
        )
    )
    if not block_matches:
        return None
    return {field: float(block_matches[-1].group(field)) for field in METRIC_FIELDS}


def parse_last_window_accuracy(
    log_path: Path, window: int = 10
) -> Optional[Dict[str, float]]:
    """Return the mean/std ACC over the final ``window`` recorded epochs."""
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    epoch_values: Dict[int, float] = {}
    for match in re.finditer(
        r"Epoch \[(?P<epoch>\d+)/(?:\d+)\].*?Val Acc:\s*"
        r"(?P<accuracy>[0-9]+(?:\.[0-9]+)?)",
        text,
    ):
        epoch_values[int(match.group("epoch"))] = float(match.group("accuracy"))
    if len(epoch_values) < window:
        return None
    selected = [value for _, value in sorted(epoch_values.items())[-window:]]
    mean = sum(selected) / len(selected)
    variance = sum((value - mean) ** 2 for value in selected) / len(selected)
    return {
        "last10_accuracy_mean": mean,
        "last10_accuracy_std": math.sqrt(variance),
        "last10_epoch_count": float(len(selected)),
    }


def successful_status(task: Mapping[str, object], task_dir: Path) -> bool:
    status_path = task_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = status.get("metrics")
    return (
        status.get("status") == "success"
        and status.get("task_signature") == task["signature"]
        and (task_dir / "train.log").is_file()
        and (task_dir / "command.txt").is_file()
        and (task_dir / "parameters.json").is_file()
        and isinstance(metrics, Mapping)
        and all(isinstance(metrics.get(field), (int, float)) for field in METRIC_FIELDS)
        and all(
            isinstance(metrics.get(field), (int, float))
            for field in LAST_WINDOW_FIELDS
        )
    )


def archive_prior_attempt(task_dir: Path) -> Optional[Path]:
    """Preserve an incomplete/failed attempt before retrying the same task."""
    artifact_names = ("train.log", "command.txt", "parameters.json", "status.json")
    existing = [task_dir / name for name in artifact_names if (task_dir / name).exists()]
    if not existing:
        return None

    attempts_root = task_dir / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = attempts_root / f"attempt_{timestamp}"
    suffix = 1
    while archive_dir.exists():
        archive_dir = attempts_root / f"attempt_{timestamp}_{suffix:02d}"
        suffix += 1
    archive_dir.mkdir(parents=True)
    for artifact in existing:
        shutil.move(str(artifact), str(archive_dir / artifact.name))
    return archive_dir


def csv_row_from_status(status: Mapping[str, object]) -> Dict[str, object]:
    metrics = status.get("metrics") if isinstance(status.get("metrics"), Mapping) else {}
    return {
        "configuration": status.get("configuration", ""),
        "scan_parameter": status.get("scan_parameter", ""),
        "scan_value": status.get("scan_value", ""),
        "target_weather": status.get("target_weather", ""),
        "target_root": status.get("target_root", ""),
        "seed": status.get("seed", 42),
        "domain_loss_weight": status.get("domain_loss_weight", ""),
        "accuracy": metrics.get("accuracy", ""),
        "precision_macro": metrics.get("precision_macro", ""),
        "recall_macro": metrics.get("recall_macro", ""),
        "f1_macro": metrics.get("f1_macro", ""),
        "last10_accuracy_mean": metrics.get("last10_accuracy_mean", ""),
        "last10_accuracy_std": metrics.get("last10_accuracy_std", ""),
        "last10_epoch_count": metrics.get("last10_epoch_count", ""),
        "status": status.get("status", ""),
        "return_code": status.get("return_code", ""),
        "gpu_id": status.get("gpu_id", ""),
        "task_directory": status.get("task_directory", ""),
        "updated_at": status.get("updated_at", ""),
    }


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    os.replace(temporary, path)


def update_csv_outputs(plan: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    for task in plan:
        status_path = Path(str(task["task_directory"])) / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if status.get("task_signature") != task["signature"]:
            continue
        rows.append(csv_row_from_status(status))

    rows.sort(key=lambda row: (str(row["configuration"]), str(row["target_weather"])))
    # summary.csv reuses the single baseline once in every active parameter
    # group without retraining it.
    atomic_write_csv(output_dir / "per_seed.csv", rows)
    summary_rows: List[Dict[str, object]] = []
    for row in rows:
        if row["configuration"] != "baseline":
            summary_rows.append(row)
            continue
        for parameter in SCAN_PARAMETERS:
            reused = dict(row)
            reused["scan_parameter"] = parameter
            reused["scan_value"] = display_value(BASELINE[parameter])
            summary_rows.append(reused)
    atomic_write_csv(output_dir / "summary.csv", summary_rows)


def base_status(task: Mapping[str, object]) -> Dict[str, object]:
    return {
        "configuration": task["configuration"],
        "scan_parameter": task["scan_parameter"],
        "scan_value": task["scan_value"],
        "target_weather": task["target_weather"],
        "target_root": task["target_root"],
        "source_root": task["source_root"],
        "seed": task["seed"],
        "domain_loss_weight": task["parameters"]["domain_loss_weight"],
        "task_directory": task["task_directory"],
        "task_signature": task["signature"],
        "status": "pending",
        "return_code": None,
        "gpu_id": None,
        "metrics": None,
        "updated_at": utc_now(),
    }


def run_task(
    task: Mapping[str, object],
    args: argparse.Namespace,
    plan: Sequence[Mapping[str, object]],
    output_dir: Path,
    runner_log: Path,
) -> bool:
    task_dir = Path(str(task["task_directory"]))
    if successful_status(task, task_dir):
        log_runner(
            runner_log,
            f"SKIP complete task {task['configuration']} x {task['target_weather']}.",
        )
        return True

    status_path = task_dir / "status.json"
    if status_path.exists():
        try:
            prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prior_status = {}
        prior_signature = prior_status.get("task_signature")
        if prior_signature and prior_signature != task["signature"]:
            raise RuntimeError(
                f"Task directory contains a different experiment signature: {task_dir}. "
                "Use a different --output-dir to preserve the existing artifacts."
            )

    archived_attempt = archive_prior_attempt(task_dir)
    if archived_attempt is not None:
        log_runner(
            runner_log,
            f"Archived incomplete prior attempt to {archived_attempt} before retry.",
        )

    task_dir.mkdir(parents=True, exist_ok=True)
    command = list(task["command"])
    atomic_write_text(task_dir / "command.txt", command_text(command) + "\n")
    atomic_write_json(
        task_dir / "parameters.json",
        {
            "configuration": task["configuration"],
            "scan_parameter": task["scan_parameter"],
            "scan_value": task["scan_value"],
            "seed": task["seed"],
            "source_root": task["source_root"],
            "target_root": task["target_root"],
            "scanned_parameters": task["parameters"],
            "fixed_training_arguments": [
                {"option": option, "value": value}
                for option, value in FIXED_TRAINING_ARGUMENTS
            ],
            "runner_arguments": {
                "python": args.python,
                "module_path": str(args.module_path),
                "gpu_ids": args.gpu_ids,
                "minimum_free_memory_mib": args.min_free_memory_mib,
                "utilization_must_be_below_percent": args.max_gpu_utilization,
                "gpu_poll_seconds": args.gpu_poll_seconds,
                "nvidia_smi": args.nvidia_smi,
            },
            "command": command,
            "task_signature": task["signature"],
        },
    )

    status = base_status(task)
    status["status"] = "waiting_for_gpu"
    status["updated_at"] = utc_now()
    atomic_write_json(status_path, status)
    update_csv_outputs(plan, output_dir)

    selected_gpu = wait_for_gpu(args, runner_log)
    gpu_id = selected_gpu["gpu_id"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    status.update(
        {
            "status": "running",
            "gpu_id": gpu_id,
            "gpu_admission": selected_gpu,
            "started_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    atomic_write_json(status_path, status)
    update_csv_outputs(plan, output_dir)
    log_runner(
        runner_log,
        f"START {task['configuration']} x {task['target_weather']} on GPU {gpu_id}: "
        f"{command_text(command)}",
    )

    train_log = task_dir / "train.log"
    return_code: Optional[int] = None
    interrupted = False
    with train_log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=str(args.project_root),
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        status["pid"] = process.pid
        status["updated_at"] = utc_now()
        atomic_write_json(status_path, status)
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            interrupted = True
            process.terminate()
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()

    metrics = parse_report_metrics(train_log)
    last_window_metrics = parse_last_window_accuracy(train_log)
    if metrics is not None and last_window_metrics is not None:
        metrics.update(last_window_metrics)
    if interrupted:
        final_status = "interrupted"
        failure_reason = "Runner interrupted by user."
    elif return_code != 0:
        final_status = "failed"
        failure_reason = f"Training process exited with return code {return_code}."
    elif metrics is None or last_window_metrics is None:
        final_status = "metrics_missing"
        failure_reason = (
            "Training exited successfully but required report or final-window metrics "
            "were not found."
        )
    else:
        final_status = "success"
        failure_reason = None

    status.update(
        {
            "status": final_status,
            "return_code": return_code,
            "metrics": metrics,
            "failure_reason": failure_reason,
            "finished_at": utc_now(),
            "updated_at": utc_now(),
        }
    )
    atomic_write_json(status_path, status)
    update_csv_outputs(plan, output_dir)
    log_runner(
        runner_log,
        f"END {task['configuration']} x {task['target_weather']}: "
        f"status={final_status}, return_code={return_code}, metrics={metrics}.",
    )
    if interrupted:
        raise KeyboardInterrupt
    return final_status == "success"


def print_dry_run(
    configurations: Sequence[Mapping[str, object]],
    targets: Sequence[Path],
    plan: Sequence[Mapping[str, object]],
) -> None:
    print("DRY-RUN validation passed")
    print(f"Unique configurations: {len(configurations)}")
    print("Shared baseline occurrences: 1")
    print(f"Target weather domains: {len(targets)}")
    print(f"Serialized tasks (configuration x target): {len(plan)}")
    print("Every non-baseline configuration changes exactly one scanned parameter.")
    print()
    for index, task in enumerate(plan, start=1):
        print(
            f"[{index:03d}/{len(plan):03d}] config={task['configuration']} "
            f"scan={task['scan_parameter']} value={task['scan_value']} "
            f"target={task['target_weather']} seed={task['seed']}"
        )
        print(f"  output={task['task_directory']}")
        print(f"  command={command_text(task['command'])}")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    parser = argparse.ArgumentParser(
        description=(
            "Run a supervised JMDA-Net one-factor "
            "hyperparameter sensitivity plan serially."
        )
    )
    parser.add_argument(
        "--scan-profile",
        choices=tuple(SCAN_PROFILES),
        default="refined",
        help="refined scans four historical parameters; lambda1 scans only domain_loss_weight",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/mnt/data/lixiang/data/Data/晴天"),
        help="source weather directory (default: /mnt/data/lixiang/data/Data/晴天)",
    )
    parser.add_argument(
        "--target-root",
        nargs="+",
        default=["逆光"],
        help=(
            "target weather name/path; accepts multiple values, comma-separated values, "
            "or 'all' to discover source-root siblings containing train/ and val/"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "Ablation" / "hparam_sensitivity_results_refined",
        help="persistent result root used for artifacts and resume",
    )
    parser.add_argument(
        "--python",
        default=os.environ.get("PYTHON", sys.executable),
        help="Python executable used to launch Ablation/module_ablation.py",
    )
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=parse_gpu_ids("0,1"),
        help="physical GPU IDs considered by admission control (default: 0,1)",
    )
    parser.add_argument(
        "--min-free-memory-mib",
        type=int,
        default=15 * 1024,
        help="minimum free memory required before starting a task (default: 15360 MiB)",
    )
    parser.add_argument(
        "--max-gpu-utilization",
        type=int,
        default=15,
        help="GPU utilization must be strictly below this percentage (default: 15)",
    )
    parser.add_argument(
        "--gpu-poll-seconds",
        type=int,
        default=60,
        help="seconds between GPU admission checks (default: 60)",
    )
    parser.add_argument(
        "--nvidia-smi",
        default="nvidia-smi",
        help="nvidia-smi executable path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the complete plan without querying GPUs or training",
    )
    args = parser.parse_args()
    args.project_root = project_root
    args.module_path = project_root / "Ablation" / "module_ablation.py"
    if args.min_free_memory_mib < 1:
        parser.error("--min-free-memory-mib must be positive")
    if args.max_gpu_utilization < 1 or args.max_gpu_utilization > 100:
        parser.error("--max-gpu-utilization must be in [1, 100]")
    if args.gpu_poll_seconds < 1:
        parser.error("--gpu-poll-seconds must be positive")
    return args


def main() -> int:
    args = parse_args()
    activate_scan_profile(args.scan_profile)
    configurations = build_configurations()
    validate_configurations(configurations)

    source_root = args.source_root.expanduser()
    if not args.dry_run and not is_weather_domain(source_root):
        raise ValueError(
            f"Source weather directory must contain train/ and val/: {source_root}"
        )
    if not args.module_path.is_file():
        raise FileNotFoundError(f"Training entry point not found: {args.module_path}")

    targets = discover_targets(source_root, args.target_root, args.dry_run)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    plan = build_plan(
        configurations,
        targets,
        source_root,
        output_dir,
        args.module_path,
        args.python,
    )

    if args.dry_run:
        print_dry_run(configurations, targets, plan)
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    runner_log = output_dir / "runner.log"
    atomic_write_json(
        output_dir / "plan.json",
        {
            "created_at": utc_now(),
            "scan_profile": args.scan_profile,
            "scan_values": {key: list(values) for key, values in SCAN_VALUES.items()},
            "configuration_count": len(configurations),
            "target_count": len(targets),
            "task_count": len(plan),
            "gpu_admission": {
                "gpu_ids": args.gpu_ids,
                "minimum_free_memory_mib": args.min_free_memory_mib,
                "utilization_must_be_below_percent": args.max_gpu_utilization,
                "poll_seconds": args.gpu_poll_seconds,
            },
            "tasks": plan,
        },
    )
    update_csv_outputs(plan, output_dir)
    log_runner(
        runner_log,
        f"Validated {len(configurations)} configurations and {len(plan)} serialized tasks.",
    )

    failures = 0
    for index, task in enumerate(plan, start=1):
        log_runner(
            runner_log,
            f"TASK {index}/{len(plan)}: {task['configuration']} x {task['target_weather']}.",
        )
        try:
            if not run_task(task, args, plan, output_dir, runner_log):
                failures += 1
        except KeyboardInterrupt:
            log_runner(runner_log, "Runner interrupted; current task status was preserved.")
            return 130
        except Exception as exc:  # preserve the failed task and continue the scan
            failures += 1
            task_dir = Path(str(task["task_directory"]))
            task_dir.mkdir(parents=True, exist_ok=True)
            existing_status_path = task_dir / "status.json"
            if existing_status_path.is_file():
                try:
                    existing_status = json.loads(
                        existing_status_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    existing_status = {}
                if existing_status.get("task_signature") not in (None, task["signature"]):
                    log_runner(
                        runner_log,
                        f"RUNNER ERROR for {task['configuration']} x "
                        f"{task['target_weather']}: {type(exc).__name__}: {exc}. "
                        "Existing status.json was preserved because its signature differs.",
                    )
                    continue
            status = base_status(task)
            status.update(
                {
                    "status": "runner_error",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "finished_at": utc_now(),
                    "updated_at": utc_now(),
                }
            )
            atomic_write_json(task_dir / "status.json", status)
            update_csv_outputs(plan, output_dir)
            log_runner(
                runner_log,
                f"RUNNER ERROR for {task['configuration']} x {task['target_weather']}: "
                f"{type(exc).__name__}: {exc}",
            )

    log_runner(
        runner_log,
        f"Sensitivity scan finished: tasks={len(plan)}, failures={failures}.",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
