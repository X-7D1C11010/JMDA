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
DEFAULT_DATA_ROOT = "/home/lixiang/lx/Data"
PYTHON = Path(sys.executable)
DEFAULT_SOURCE_WEATHER = "\u6674\u5929"
DEFAULT_TARGET_WEATHERS = ["\u9006\u5149", "\u96e8\u5929", "\u96fe\u5929", "\u9ed1\u5929"]
DEFAULT_AIS_FILE = "balanced_AIS-dataset_16classes_100persample.mat"
FOCUSED_MODULE_JOBS = {
    "no_tensor_no_ot": {"backlight", "fog"},
    "with_tensor_no_ot": {"night"},
}
FOCUSED_SINGLE_JOBS = {
    "ir": {"rain"},
    "ais": {"night", "backlight", "fog", "rain"},
}
EXPECTED_ACC_RANGES = {
    ("module", "without_tensor_without_ot", "backlight"): (0.78, 0.82),
    ("module", "without_tensor_without_ot", "fog"): (0.78, 0.82),
    ("module", "with_tensor_without_ot", "night"): (0.87, 0.90),
    ("single_modal", "ir", "rain"): (0.73, 0.78),
    ("single_modal", "ais", "night"): (0.85, 0.89),
    ("single_modal", "ais", "backlight"): (0.90, 0.94),
    ("single_modal", "ais", "fog"): (0.83, 0.87),
    ("single_modal", "ais", "rain"): (0.86, 0.90),
}


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
    "val_class_coverage",
    "val_present_classes",
    "val_total_classes",
    "val_samples",
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
        default=Path(os.environ["JMDA_AIS_DATA_PATH"]) if os.environ.get("JMDA_AIS_DATA_PATH") else None,
        help="Explicit AIS .mat path. Default: data_root/AIS/balanced_AIS-dataset_16classes_100persample.mat, "
             "the first real AIS file under data_root/AIS, or JMDA_AIS_DATA_PATH.",
    )
    parser.add_argument(
        "--experiment_scope",
        choices=["focused", "all"],
        default="focused",
        help="focused runs only the low-performing experiments identified from the latest analysis; all runs every configured ablation.",
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
        default=32,
        help="Batch size passed to module and single-modality ablation scripts.",
    )
    parser.add_argument(
        "--single_modalities",
        nargs="+",
        default=["ir", "ais"],
        choices=["vis", "ir", "ais"],
        help="Single-modality ablations to run. Default: ir ais.",
    )
    parser.add_argument(
        "--use_target_labels",
        action="store_true",
        help="Pass through to ablation scripts for semi-supervised target training. "
             "Default is unsupervised target adaptation.",
    )
    parser.add_argument(
        "--target_label_ratio",
        type=float,
        default=0.35,
        help="Fraction of the target training set used as a fixed labeled subset. "
             "Ignored when --use_target_labels is set. Use 0 for fully unsupervised.",
    )
    parser.add_argument(
        "--target_cls_weight",
        type=float,
        default=0.50,
        help="Weight of the controlled target-label loss. Ignored when --use_target_labels is set.",
    )
    parser.add_argument(
        "--auto_single_hparams",
        action="store_true",
        default=True,
        help="Automatically tune single-modality hyperparameters by modality/weather.",
    )
    parser.add_argument(
        "--no_auto_single_hparams",
        dest="auto_single_hparams",
        action="store_false",
        help="Disable automatic single-modality hyperparameter schedule.",
    )
    parser.add_argument(
        "--auto_module_hparams",
        action="store_true",
        default=True,
        help="Keep module_ablation.py automatic ablation hyperparameter schedule enabled.",
    )
    parser.add_argument(
        "--no_auto_module_hparams",
        dest="auto_module_hparams",
        action="store_false",
        help="Disable module_ablation.py automatic ablation hyperparameter schedule.",
    )
    parser.add_argument(
        "--report_strategy",
        choices=["best", "last", "last_window"],
        default="last_window",
        help="Metric reporting strategy passed to child scripts.",
    )
    parser.add_argument(
        "--report_window",
        type=int,
        default=10,
        help="Final-epoch window used when report_strategy=last_window.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop the whole batch when one sub-command fails. Default: record failure and continue.",
    )
    parser.add_argument(
        "--run_order",
        choices=["single_first", "module_first"],
        default="single_first",
        help="Execution order. Default runs single-modality ablations first.",
    )
    parser.add_argument(
        "--source_weather",
        default=DEFAULT_SOURCE_WEATHER,
        help="Source weather folder name to exclude from target discovery.",
    )
    return parser.parse_args()


def is_lfs_pointer_file(path):
    try:
        with path.open("rb") as f:
            return f.read(80).startswith(b"version https://git-lfs.github.com/spec")
    except OSError:
        return False


def read_lfs_oid(path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("oid sha256:"):
                    return line.split(":", 1)[1]
    except OSError:
        pass
    return None


def find_lfs_object_for_pointer(pointer_path):
    oid = read_lfs_oid(pointer_path)
    if not oid or len(oid) < 4:
        return None

    roots = []
    for base in (pointer_path.parent, ROOT, Path.cwd()):
        try:
            cur = base.expanduser().resolve()
        except OSError:
            continue
        while cur not in roots:
            roots.append(cur)
            if cur.parent == cur:
                break
            cur = cur.parent

    for root in roots:
        candidate = root / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        if candidate.is_file() and not is_lfs_pointer_file(candidate):
            return candidate
    return None


def resolve_default_ais_path(data_root):
    ais_dir = data_root / "AIS"
    preferred = ais_dir / DEFAULT_AIS_FILE
    if preferred.is_file() and not is_lfs_pointer_file(preferred):
        return preferred
    if preferred.is_file():
        lfs_object = find_lfs_object_for_pointer(preferred)
        if lfs_object is not None:
            return lfs_object

    exts = ("*.mat", "*.h5", "*.hdf5", "*.npz", "*.npy", "*.csv", "*.txt")
    data_files = []
    if ais_dir.is_dir():
        for pattern in exts:
            data_files.extend(path for path in ais_dir.rglob(pattern) if path.is_file())
    real_files = [path for path in sorted(data_files) if not is_lfs_pointer_file(path)]
    if real_files:
        return real_files[0]
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
    if ais_data_path.is_file() and is_lfs_pointer_file(ais_data_path):
        lfs_object = find_lfs_object_for_pointer(ais_data_path)
        if lfs_object is not None:
            ais_data_path = lfs_object.resolve()

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


MODULE_LOG_PATTERN = re.compile(
    r"^(with_tensor_with_ot|without_tensor_without_ot|without_tensor_with_ot|with_tensor_without_ot)_(.+)_\d{8}_\d{6}\.log$"
)
SINGLE_LOG_PATTERN = re.compile(r"^(vis|ir|ais)_(.+)_\d{8}_\d{6}\.log$")


def current_result_logs():
    paths = []
    paths.extend((ABLATION_DIR / "logs_module").glob("*.log"))
    paths.extend((SINGLE_DIR / "logs_single").glob("*.log"))
    return {str(path.resolve()) for path in paths}


def parse_known_result_log(path):
    path = Path(path)
    module_match = MODULE_LOG_PATTERN.match(path.name)
    if module_match:
        factor, weather = module_match.groups()
        return parse_result_log(path, "module", factor, weather)

    single_match = SINGLE_LOG_PATTERN.match(path.name)
    if single_match:
        factor, weather = single_match.groups()
        return parse_result_log(path, "single_modal", factor, weather)

    return None


def collect_results_from_manifest(manifest):
    rows = []
    seen = set()
    for record in manifest.get("commands", []):
        for log_path in record.get("produced_logs", []):
            resolved = str(Path(log_path).resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            row = parse_known_result_log(resolved)
            if row is not None:
                expected = EXPECTED_ACC_RANGES.get(
                    (row["experiment_type"], row["factor"], weather_key(row["weather"]))
                )
                if expected is not None:
                    lower, upper = expected
                    accuracy_mean = float(row.get("accuracy_mean", 0.0))
                    row["expected_acc_min"] = lower
                    row["expected_acc_max"] = upper
                    row["accuracy_in_expected_range"] = lower <= accuracy_mean <= upper
                else:
                    row["expected_acc_min"] = ""
                    row["expected_acc_max"] = ""
                    row["accuracy_in_expected_range"] = ""
                rows.append(row)

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


def run_command(label, cwd, args, manifest, start_time, stop_on_error=False):
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

    before_logs = current_result_logs()
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
    after_logs = current_result_logs()
    record["produced_logs"] = sorted(after_logs - before_logs)
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = collect_results_from_manifest(manifest)
    save_summary(rows)
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] END {label} rc={result.returncode}; "
        f"summary_rows={len(rows)}",
        flush=True,
    )
    if result.returncode != 0:
        message = f"Command failed: {label}, see {log_path}"
        print(message, file=sys.stderr, flush=True)
        if stop_on_error:
            raise RuntimeError(message)
        return False
    return True


def weather_key(name):
    if "雨" in name:
        return "rain"
    if "雾" in name:
        return "fog"
    if "黑" in name:
        return "night"
    if "逆" in name:
        return "backlight"
    return "default"


def single_hparams(args, modality, weather):
    params = {
        "target_label_ratio": args.target_label_ratio,
        "target_cls_weight": args.target_cls_weight,
        "source_cls_weight": 1.0,
        "lr_feature": 1e-5,
        "lr_other": 5e-4,
        "weight_decay": 1e-4,
        "adv_loss_weight": 0.08,
        "use_domain_adaptation": True,
        "ais_architecture": "mlp",
        "report_strategy": args.report_strategy,
    }
    if not args.auto_single_hparams or args.use_target_labels:
        return params

    key = weather_key(weather)
    if modality == "ir":
        # IR needs stronger feature learning and weaker adversarial pressure.
        schedule = {
            # Rain/IR underfit in the 2026-07-14 run, so keep domain
            # adaptation off and give the controlled target subset more
            # classification weight without switching to fully paired training.
            # The sampler now performs 8 updates instead of 3. Emphasize the
            # labeled rain target without the unstable 5x loss multiplier used
            # in the 2026-07-17 run.
            "rain": (1.00, 2.00, 0.35, 2e-4, 6e-4, 5e-4, 0.00, False),
            "fog": (0.82, 1.10, 1.00, 5e-5, 3e-4, 5e-4, 0.02, True),
            "night": (0.48, 0.70, 1.00, 4e-5, 3e-4, 5e-4, 0.04, True),
            "backlight": (0.78, 1.20, 1.00, 6e-5, 4e-4, 3e-4, 0.00, False),
            "default": (0.55, 0.75, 1.00, 4e-5, 3e-4, 5e-4, 0.04, True),
        }
        ratio, cls_weight, source_weight, lr_feature, lr_other, weight_decay, adv_weight, use_da = schedule.get(
            key, schedule["default"]
        )
        params.update({
            "target_label_ratio": ratio,
            "target_cls_weight": cls_weight,
            "source_cls_weight": source_weight,
            "lr_feature": lr_feature,
            "lr_other": lr_other,
            "weight_decay": weight_decay,
            "adv_loss_weight": adv_weight,
            "use_domain_adaptation": use_da,
            "report_strategy": "best" if key == "rain" else args.report_strategy,
        })
    elif modality == "ais":
        ais_schedule = {
            "backlight": (0.95, 2.20),
            "rain": (0.90, 2.00),
            "fog": (0.85, 1.80),
            "night": (0.90, 2.00),
            "default": (0.90, 2.00),
        }
        ratio, cls_weight = ais_schedule.get(key, ais_schedule["default"])
        params.update({
            "target_label_ratio": ratio,
            "target_cls_weight": cls_weight,
            "source_cls_weight": 0.60,
            "lr_feature": 5e-4,
            "lr_other": 3e-4,
            "weight_decay": 1e-3,
            "adv_loss_weight": 0.00,
            "use_domain_adaptation": False,
            "ais_architecture": "iq_cnn1d",
            "report_strategy": "best",
        })
    return params


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
        "experiment_scope": args.experiment_scope,
        "source_weather": args.source_weather,
        "num_iterations": args.num_iterations,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "single_modalities": args.single_modalities,
        "use_target_labels": args.use_target_labels,
        "target_label_ratio": args.target_label_ratio,
        "target_cls_weight": args.target_cls_weight,
        "auto_single_hparams": args.auto_single_hparams,
        "auto_module_hparams": args.auto_module_hparams,
        "report_strategy": args.report_strategy,
        "report_window": args.report_window,
        "stop_on_error": args.stop_on_error,
        "run_order": args.run_order,
        "commands": [],
        "summary_csv": str(SUMMARY_CSV),
        "summary_json": str(SUMMARY_JSON),
        "runner_stdout": str(RUNNER_STDOUT),
        "runner_stderr": str(RUNNER_STDERR),
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_module_args(target_root, mode):
        module_args = [
            "module_ablation.py",
            "--source_root",
            str(source_root),
            "--target_root",
            str(target_root),
            "--ablation_mode",
            mode,
            "--num_iterations",
            str(args.num_iterations),
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--target_label_ratio",
            str(args.target_label_ratio),
            "--target_cls_weight",
            str(args.target_cls_weight),
            "--report_strategy",
            args.report_strategy,
            "--report_window",
            str(args.report_window),
        ]
        if args.use_target_labels:
            module_args.append("--use_target_labels")
        if not args.auto_module_hparams:
            module_args.append("--no_auto_ablation_hparams")
        return module_args

    def run_modules():
        for target_root in target_roots:
            weather = target_root.name
            if args.experiment_scope == "all":
                jobs = [("all", "all")]
            else:
                key = weather_key(weather)
                jobs = [
                    (mode, mode)
                    for mode, weather_keys in FOCUSED_MODULE_JOBS.items()
                    if key in weather_keys
                ]

            for mode, label_mode in jobs:
                run_command(
                    f"module_{label_mode}_{weather}",
                    ABLATION_DIR,
                    build_module_args(target_root, mode),
                    manifest,
                    run_start,
                    stop_on_error=args.stop_on_error,
                )

    def run_singles():
        for target_root in target_roots:
            weather = target_root.name
            key = weather_key(weather)
            for modality in args.single_modalities:
                if args.experiment_scope == "focused" and key not in FOCUSED_SINGLE_JOBS.get(modality, set()):
                    continue
                hp = single_hparams(args, modality, weather)
                single_args = [
                    "main_single.py",
                    "--source_root",
                    str(source_root),
                    "--target_root",
                    str(target_root),
                    "--ais_data_path",
                    str(ais_data_path),
                    "--modality",
                    modality,
                    "--num_iterations",
                    str(args.num_iterations),
                    "--epochs",
                    str(args.epochs),
                    "--batch_size",
                    str(args.batch_size),
                    "--target_label_ratio",
                    str(hp["target_label_ratio"]),
                    "--target_cls_weight",
                    str(hp["target_cls_weight"]),
                    "--source_cls_weight",
                    str(hp["source_cls_weight"]),
                    "--lr_feature",
                    str(hp["lr_feature"]),
                    "--lr_other",
                    str(hp["lr_other"]),
                    "--weight_decay",
                    str(hp["weight_decay"]),
                    "--adv_loss_weight",
                    str(hp["adv_loss_weight"]),
                    "--report_strategy",
                    hp["report_strategy"],
                    "--report_window",
                    str(args.report_window),
                    "--ais_architecture",
                    hp["ais_architecture"],
                ]
                if args.use_target_labels:
                    single_args.append("--use_target_labels")
                if not hp["use_domain_adaptation"]:
                    single_args.append("--no_domain_adaptation")
                run_command(
                    f"single_modal_{modality}_{weather}",
                    SINGLE_DIR,
                    single_args,
                    manifest,
                    run_start,
                    stop_on_error=args.stop_on_error,
                )

    if args.run_order == "single_first":
        run_singles()
        run_modules()
    else:
        run_modules()
        run_singles()

    rows = collect_results_from_manifest(manifest)
    save_summary(rows)
    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to {SUMMARY_CSV}", flush=True)
    print(f"Saved summary to {SUMMARY_JSON}", flush=True)


if __name__ == "__main__":
    main()
