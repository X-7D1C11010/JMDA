#!/usr/bin/env bash
# Run the weather-domain and VDS2Raw supervised sensitivity suites in order.
# Both child runners remain responsible for GPU admission, per-task logging,
# resume semantics, and result aggregation.

set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/lixiang/lx/JMDA}"
PYTHON_BIN="${PYTHON_BIN:-/home/lixiang/envs/lx_pytorch/bin/python}"
GPU_IDS="${GPU_IDS:-0,1}"

WEATHER_SOURCE_ROOT="${WEATHER_SOURCE_ROOT:-/mnt/data/lixiang/data/Data/晴天}"
WEATHER_TARGET_ROOT="${WEATHER_TARGET_ROOT:-逆光}"
WEATHER_OUTPUT_ROOT="${WEATHER_OUTPUT_ROOT:-${PROJECT_ROOT}/Ablation/hparam_sensitivity_results/backlight}"

VDS2RAW_DATASET_ROOT="${VDS2RAW_DATASET_ROOT:-/home/lixiang/lx/VDS2Raw/threeclass_ready}"
VDS2RAW_OUTPUT_ROOT="${VDS2RAW_OUTPUT_ROOT:-${PROJECT_ROOT}/VDS2Raw_JMDA/runs/hparam_sensitivity}"

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
if [[ ! -f "$PROJECT_ROOT/Ablation/run_hparam_sensitivity.py" ]]; then
  printf 'Weather sensitivity runner not found under: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/VDS2Raw_JMDA/run_hparam_sensitivity.py" ]]; then
  printf 'VDS2Raw sensitivity runner not found under: %s\n' "$PROJECT_ROOT" >&2
  exit 2
fi

mkdir -p "$WEATHER_OUTPUT_ROOT" "$VDS2RAW_OUTPUT_ROOT"
cd "$PROJECT_ROOT"

# The VDS2Raw runner respects this visibility list. The weather runner also
# receives the same physical IDs explicitly and narrows its child to one GPU.
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

stage_message "[1/2] Starting supervised weather-domain sensitivity suite."
stage_message "Weather source=$WEATHER_SOURCE_ROOT target=$WEATHER_TARGET_ROOT output=$WEATHER_OUTPUT_ROOT"

"$PYTHON_BIN" -u -B Ablation/run_hparam_sensitivity.py \
  --source-root "$WEATHER_SOURCE_ROOT" \
  --target-root "$WEATHER_TARGET_ROOT" \
  --output-dir "$WEATHER_OUTPUT_ROOT" \
  --python "$PYTHON_BIN" \
  --gpu-ids "$GPU_IDS" \
  --min-free-memory-mib 15360 \
  --max-gpu-utilization 15 \
  --gpu-poll-seconds 60
weather_status=$?

stage_message "[1/2] Weather-domain suite exited with status $weather_status."
if [[ "$weather_status" -eq 130 ]]; then
  stage_message "Weather-domain suite was interrupted; VDS2Raw suite will not start."
  exit 130
fi

stage_message "[2/2] Starting supervised VDS2Raw sensitivity suite."
stage_message "VDS2Raw dataset=$VDS2RAW_DATASET_ROOT output=$VDS2RAW_OUTPUT_ROOT"

"$PYTHON_BIN" -u -B VDS2Raw_JMDA/run_hparam_sensitivity.py \
  --dataset-root "$VDS2RAW_DATASET_ROOT" \
  --output-root "$VDS2RAW_OUTPUT_ROOT" \
  --python-executable "$PYTHON_BIN" \
  --gpu-min-free-gb 15 \
  --gpu-max-utilization 15 \
  --gpu-poll-seconds 60
vds2raw_status=$?

stage_message "[2/2] VDS2Raw suite exited with status $vds2raw_status."
stage_message "Sequential sensitivity run finished: weather=$weather_status, VDS2Raw=$vds2raw_status."

if [[ "$weather_status" -ne 0 || "$vds2raw_status" -ne 0 ]]; then
  exit 1
fi
exit 0
