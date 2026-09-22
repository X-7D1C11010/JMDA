#!/usr/bin/env bash
set -euo pipefail

# Run from any directory. Override values through environment variables when
# the server environment or experiment budget differs from these defaults.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/lixiang/lx/Data/晴天}"
TARGET_ROOT="${TARGET_ROOT:-all}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
NUM_ITERATIONS="${NUM_ITERATIONS:-5}"
TRANSPORT_MODES="${TRANSPORT_MODES:-legacy_row_softmax row_softmax sinkhorn}"

for mode in ${TRANSPORT_MODES}; do
    echo "============================================================"
    echo "Running supervised transport ablation: ${mode}"
    echo "source=${SOURCE_ROOT}, target=${TARGET_ROOT}"
    echo "============================================================"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/module_ablation.py" \
        --ablation_mode full \
        --use_target_labels \
        --no_auto_ablation_hparams \
        --transport_mode "${mode}" \
        --source_root "${SOURCE_ROOT}" \
        --target_root "${TARGET_ROOT}" \
        --batch_size "${BATCH_SIZE}" \
        --epochs "${EPOCHS}" \
        --num_iterations "${NUM_ITERATIONS}"
done
