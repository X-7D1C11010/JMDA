#!/usr/bin/env bash
# Run only the lambda1/domain-loss-weight sensitivity scan on the Yantai and
# VDS2Raw supervised protocols.  Each child runner performs six serial runs:
# 0.06, 0.08, 0.10, 0.12, 0.14, and the existing baseline 1.00.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/lixiang/lx/JMDA}"
PYTHON_BIN="${PYTHON_BIN:-/home/lixiang/envs/lx_pytorch/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"

WEATHER_SOURCE_ROOT="${WEATHER_SOURCE_ROOT:-/mnt/data/lixiang/data/Data/晴天}"
WEATHER_TARGET_ROOT="${WEATHER_TARGET_ROOT:-逆光}"
WEATHER_OUTPUT_ROOT="${WEATHER_OUTPUT_ROOT:-${PROJECT_ROOT}/Ablation/lambda1_sensitivity_results/backlight}"

VDS2RAW_DATASET_ROOT="${VDS2RAW_DATASET_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
VDS2RAW_OUTPUT_ROOT="${VDS2RAW_OUTPUT_ROOT:-${PROJECT_ROOT}/VDS2Raw_JMDA/runs/lambda1_sensitivity}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %Z'
}

stage_message() {
  printf '%s | %s\n' "$(timestamp)" "$*"
}

if [[ ! -d "$PROJECT_ROOT" ]]; then
  printf 'Project root not found: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'Python executable not found or not executable: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi

mkdir -p "$WEATHER_OUTPUT_ROOT" "$VDS2RAW_OUTPUT_ROOT"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

stage_message "[1/2] Starting Yantai lambda1 sensitivity scan."
"$PYTHON_BIN" -u -B Ablation/run_hparam_sensitivity.py \
  --scan-profile lambda1 \
  --source-root "$WEATHER_SOURCE_ROOT" \
  --target-root "$WEATHER_TARGET_ROOT" \
  --output-dir "$WEATHER_OUTPUT_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "$GPU_IDS" \
  --min-free-memory-mib 15360 \
  --max-gpu-utilization 15 \
  --gpu-poll-seconds 60
weather_status=$?

stage_message "[1/2] Yantai scan exited with status $weather_status."
if [[ "$weather_status" -eq 130 ]]; then
  stage_message "Yantai scan was interrupted; VDS2Raw scan will not start."
  exit 130
fi

stage_message "[2/2] Starting VDS2Raw lambda1 sensitivity scan."
"$PYTHON_BIN" -u -B VDS2Raw_JMDA/run_hparam_sensitivity.py \
  --scan-profile lambda1 \
  --dataset-root "$VDS2RAW_DATASET_ROOT" \
  --output-root "$VDS2RAW_OUTPUT_ROOT" \
  --python-executable "$PYTHON_BIN" \
  --gpu-min-free-gb 15 \
  --gpu-max-utilization 15 \
  --gpu-poll-seconds 60
vds2raw_status=$?

stage_message "[2/2] VDS2Raw scan exited with status $vds2raw_status."
stage_message "Lambda1 sensitivity scans finished: Yantai=$weather_status, VDS2Raw=$vds2raw_status."

if [[ "$weather_status" -ne 0 || "$vds2raw_status" -ne 0 ]]; then
  exit 1
fi
exit 0
