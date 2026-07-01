# -*- coding: utf-8 -*-
import csv
import json
import os
import re
import subprocess
import sys
import argparse
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = ROOT / "Ablation"
SINGLE_DIR = ABLATION_DIR / "signle modal"
DEFAULT_DATA_ROOT = ROOT / "Data"
PYTHON = Path(sys.executable)
DEFAULT_SOURCE_WEATHER = "\u6674\u5929"
DEFAULT_TARGET_WEATHERS = ["\u9006\u5149", "\u96e8\u5929", "\u96fe\u5929", "\u9ed1\u5929"]
DEFAULT_AIS_FILE = "balanced_AIS-dataset_16classes_100persample.mat"


def discover_target_weathers(data_root, source_weather=DEFAULT_SOURCE_WEATHER):
    targets = []
    for path in sorted(data_root.iterdir(), key=lambda p: p.name):
        train_dir = path / "train"
        if not path.is_dir() or not train_dir.is_dir() or path.name in {source_weather, "AIS"}:
            continue
        has_class_dirs = any(child.is_dir() and child.name.isdigit() for child in train_dir.iterdir())
        if has_class_dirs:
            targets.append(path.name)
    return targets


METRICS = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_micro",
    "recall_micro",
    "f1_micro",
]


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


RUN_ID = timestamp()
RUN_DIR = ROOT / "experiment_runs" / f"requested_ablation_{RUN_ID}"
CMD_LOG_DIR = RUN_DIR / "command_logs"
SUMMARY_CSV = RUN_DIR / "summary.csv"
SUMMARY_JSON = RUN_DIR / "summary.json"
MANIFEST_JSON = RUN_DIR / "manifest.json"
RUNNER_STDOUT = RUN_DIR / "runner_stdout.log"
RUNNER_STDERR = RUN_DIR / "runner_stderr.log"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all requested module and single-modality ablation experiments once.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path(os.environ.get("JMDA_DATA_ROOT", DEFAULT_DATA_ROOT)),
        help="Dataset root containing source/target weather folders and AIS/. "
             "Can also be set by JMDA_DATA_ROOT. Default: project Data/.",
    )
    parser.add_argument(
        "--source_root",
        type=Path,
        default=None,
        help="Explicit source-domain folder. Default: data_root/source_weather.",
    )
    parser.add_argument(
        "--target_weathers",
        nargs="+",
        default=None,
        help="Target weather folder names under data_root. Default: 逆光 雨天 雾天 黑天 when present.",
    )
    parser.add_argument(
        "--target_roots",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit target-domain folders. If set, target_weathers is ignored.",
    )
    parser.add_argument(
        "--ais_data_path",
        type=Path,
        default=None,
        help="Explicit AIS .mat path. Default: data_root/AIS/balanced_AIS-dataset_16classes_100persample.mat, "
             "or the first .mat file under data_root/AIS.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=5,
        help="Number of repeated runs for every ablation experiment.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Training epochs for every repeated run.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size passed to module and single-modality ablation scripts.",
    )
    parser.add_argument(
        "--source_weather",
        default=DEFAULT_SOURCE_WEATHER,
        help="Source weather folder name to exclude from target discovery.",
    )
    return parser.parse_args()


def resolve_default_ais_path(data_root):
    ais_dir = data_root / "AIS"
    preferred = ais_dir / DEFAULT_AIS_FILE
    if preferred.is_file():
        return preferred

    mat_files = sorted(ais_dir.glob("*.mat")) if ais_dir.is_dir() else []
    if mat_files:
        return mat_files[0]
    return preferred


def resolve_experiment_paths(args):
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    source_root = (
        args.source_root.expanduser().resolve()
        if args.source_root is not None
        else (data_root / args.source_weather).resolve()
    )

    if args.target_roots:
        target_roots = [path.expanduser().resolve() for path in args.target_roots]
    else:
        if args.target_weathers:
            target_weathers = args.target_weathers
        else:
            existing_defaults = [
                name for name in DEFAULT_TARGET_WEATHERS
                if (data_root / name).is_dir()
            ]
            target_weathers = existing_defaults or discover_target_weathers(data_root, args.source_weather)
        target_roots = [(data_root / name).resolve() for name in target_weathers]

    ais_data_path = (
        args.ais_data_path.expanduser().resolve()
        if args.ais_data_path is not None
        else resolve_default_ais_path(data_root).resolve()
    )

    if not source_root.is_dir():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")

    missing_targets = [path for path in target_roots if not path.is_dir()]
    if missing_targets:
        raise FileNotFoundError(f"target_root does not exist: {missing_targets}")
    if not ais_data_path.is_file():
        raise FileNotFoundError(
            f"AIS .mat file does not exist: {ais_data_path}. "
            "Pass --ais_data_path explicitly if it is stored elsewhere."
        )

    return data_root, source_root, target_roots, ais_data_path


def metric_section(lines, metric_name):
    for i, line in enumerate(lines):
        if line.strip().lower() == f"{metric_name}:":
            block = []
            for j in range(i + 1, min(i + 8, len(lines))):
                text = lines[j].strip()
                if not text:
                    break
                if text.endswith(":") and not re.search(r"\d", text):
                    break
                block.append(text)

            values = []
            mean = None
            std = None
            joined = "\n".join(block)
            values = [float(x) for x in re.findall(r"'([0-9]+(?:\.[0-9]+)?)'", joined)]
            for text in block:
                nums = [float(x) for x in re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)]
                if not nums:
                    continue
                if "Std" in text:
                    std = nums[-1]
                elif "Mean" in text:
                    mean = nums[-1]
            return {"values": values, "mean": mean, "std": std}

    return {"values": [], "mean": None, "std": None}


def parse_result_log(path, experiment_type, factor, weather):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    row = {
        "experiment_type": experiment_type,
        "factor": factor,
        "weather": weather,
        "log_path": str(path),
        "epochs_logged": len(re.findall(r"Epoch \[", text)),
        "complete": False,
    }

    complete = True
    for metric in METRICS:
        parsed = metric_section(lines, metric)
        row[f"{metric}_values"] = "|".join(f"{v:.4f}" for v in parsed["values"])
        row[f"{metric}_mean"] = parsed["mean"]
        row[f"{metric}_std"] = parsed["std"]
        complete = complete and parsed["mean"] is not None and parsed["std"] is not None
    row["complete"] = complete
    return row


def collect_results(start_time):
    rows = []

    module_pattern = re.compile(
        r"^(with_tensor_with_ot|without_tensor_without_ot|without_tensor_with_ot|with_tensor_without_ot)_(.+)_\d{8}_\d{6}\.log$"
    )
    module_logs = sorted((ABLATION_DIR / "logs_module").glob("*.log"), key=lambda p: p.stat().st_mtime)
    for path in module_logs:
        if path.stat().st_mtime < start_time:
            continue
        match = module_pattern.match(path.name)
        if not match:
            continue
        factor, weather = match.groups()
        rows.append(parse_result_log(path, "module", factor, weather))

    single_pattern = re.compile(r"^(vis|ir|ais)_(.+)_\d{8}_\d{6}\.log$")
    single_logs = sorted((SINGLE_DIR / "logs_single").glob("*.log"), key=lambda p: p.stat().st_mtime)
    for path in single_logs:
        if path.stat().st_mtime < start_time:
            continue
        match = single_pattern.match(path.name)
        if not match:
            continue
        factor, weather = match.groups()
        rows.append(parse_result_log(path, "single_modal", factor, weather))

    return rows


def save_summary(rows):
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        SUMMARY_CSV.write_text("", encoding="utf-8-sig")

    SUMMARY_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def redirect_runner_stdio():
    """Keep the batch runner stable when it is launched as a hidden process."""
    sys.stdout = RUNNER_STDOUT.open("a", encoding="utf-8", buffering=1)
    sys.stderr = RUNNER_STDERR.open("a", encoding="utf-8", buffering=1)


def run_command(label, cwd, args, manifest, start_time):
    CMD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = CMD_LOG_DIR / f"{label}_{timestamp()}.log"
    command = [str(PYTHON)] + args

    record = {
        "label": label,
        "cwd": str(cwd),
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path),
        "return_code": None,
    }
    manifest["commands"].append(record)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] START {label}", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"COMMAND: {' '.join(command)}\n")
        f.write(f"CWD: {cwd}\n")
        f.write("=" * 80 + "\n")
        f.flush()
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

    record["finished_at"] = datetime.now().isoformat(timespec="seconds")
    record["return_code"] = result.returncode
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = collect_results(start_time)
    save_summary(rows)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END {label} rc={result.returncode}; "
        f"summary_rows={len(rows)}",
        flush=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {label}, see {log_path}")


def main():
    args = parse_args()
    data_root, source_root, target_roots, ais_data_path = resolve_experiment_paths(args)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CMD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    redirect_runner_stdio()

    run_start = datetime.now().timestamp() - 2
    manifest = {
        "run_id": RUN_ID,
        "root": str(ROOT),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": str(PYTHON),
        "data_root": str(data_root),
        "source_root": str(source_root),
        "target_roots": [str(path) for path in target_roots],
        "target_weathers": [path.name for path in target_roots],
        "ais_data_path": str(ais_data_path),
        "source_weather": args.source_weather,
        "num_iterations": args.num_iterations,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "commands": [],
        "summary_csv": str(SUMMARY_CSV),
        "summary_json": str(SUMMARY_JSON),
        "runner_stdout": str(RUNNER_STDOUT),
        "runner_stderr": str(RUNNER_STDERR),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    for target_root in target_roots:
        weather = target_root.name
        run_command(
            f"module_{weather}",
            ABLATION_DIR,
            [
                "module_ablation.py",
                "--source_root",
                str(source_root),
                "--target_root",
                str(target_root),
                "--ablation_mode",
                "all",
                "--num_iterations",
                str(args.num_iterations),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
            ],
            manifest,
            run_start,
        )

    for target_root in target_roots:
        weather = target_root.name
        run_command(
            f"single_modal_{weather}",
            SINGLE_DIR,
            [
                "main_single.py",
                "--source_root",
                str(source_root),
                "--target_root",
                str(target_root),
                "--ais_data_path",
                str(ais_data_path),
                "--modality",
                "all",
                "--num_iterations",
                str(args.num_iterations),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
            ],
            manifest,
            run_start,
        )

    rows = collect_results(run_start)
    save_summary(rows)
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {SUMMARY_CSV}", flush=True)
    print(f"Saved summary to {SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
