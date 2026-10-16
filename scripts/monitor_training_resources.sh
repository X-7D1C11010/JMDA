#!/usr/bin/env bash

# Read-only resource monitor for a running JMDA-Net training process.
#
# The monitor never sends signals to the target process. It writes:
#   1. a human-readable .log file; and
#   2. a machine-readable .csv file suitable for plotting or leak analysis.

set -uo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/monitor_training_resources.sh --pid PID [options]

Required:
  --pid PID                 Training process ID to monitor.

Options:
  --interval SECONDS        Sampling interval (default: 60).
  --output-dir DIRECTORY    Output directory (default: <process-cwd>/resource_monitor).
  --run-log FILE            Training stdout/stderr log. Auto-detected from /proc/PID/fd/1 or fd/2.
  --total-epochs NUMBER     Expected total completed epochs, used only for ETA (default: 0, disabled).
  --samples NUMBER          Stop after this many samples (default: 0, monitor until process exits).
  -h, --help                Show this help.

Example:
  bash scripts/monitor_training_resources.sh \
    --pid 1727151 \
    --interval 60 \
    --total-epochs 2000
EOF
}

TARGET_PID=""
INTERVAL_SECONDS=60
OUTPUT_DIR=""
RUN_LOG=""
TOTAL_EPOCHS=0
MAX_SAMPLES=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pid)
            TARGET_PID="${2:-}"
            shift 2
            ;;
        --interval)
            INTERVAL_SECONDS="${2:-}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:-}"
            shift 2
            ;;
        --run-log)
            RUN_LOG="${2:-}"
            shift 2
            ;;
        --total-epochs)
            TOTAL_EPOCHS="${2:-}"
            shift 2
            ;;
        --samples)
            MAX_SAMPLES="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

is_nonnegative_integer() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

if ! is_nonnegative_integer "$TARGET_PID" || [[ "$TARGET_PID" -lt 1 ]]; then
    echo "--pid must be a positive integer." >&2
    exit 2
fi
if ! is_nonnegative_integer "$INTERVAL_SECONDS" || [[ "$INTERVAL_SECONDS" -lt 1 ]]; then
    echo "--interval must be a positive integer." >&2
    exit 2
fi
if ! is_nonnegative_integer "$TOTAL_EPOCHS"; then
    echo "--total-epochs must be a non-negative integer." >&2
    exit 2
fi
if ! is_nonnegative_integer "$MAX_SAMPLES"; then
    echo "--samples must be a non-negative integer." >&2
    exit 2
fi
if [[ ! -d "/proc/$TARGET_PID" ]]; then
    echo "Process $TARGET_PID does not exist." >&2
    exit 1
fi

PROCESS_CWD=$(readlink -f "/proc/$TARGET_PID/cwd" 2>/dev/null || true)
if [[ -z "$PROCESS_CWD" ]]; then
    PROCESS_CWD=$(pwd)
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$PROCESS_CWD/resource_monitor"
fi
mkdir -p "$OUTPUT_DIR" || {
    echo "Cannot create output directory: $OUTPUT_DIR" >&2
    exit 1
}

detect_run_log() {
    local fd_path candidate
    for fd_path in "/proc/$TARGET_PID/fd/1" "/proc/$TARGET_PID/fd/2"; do
        candidate=$(readlink -f "$fd_path" 2>/dev/null || true)
        if [[ -n "$candidate" && -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    candidate=$(
        find "$PROCESS_CWD/Ablation/logs_module" \
            -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' \
            2>/dev/null |
        sort -nr |
        head -n 1 |
        cut -d' ' -f2-
    )
    if [[ -n "$candidate" && -f "$candidate" ]]; then
        printf '%s\n' "$candidate"
    fi
}

if [[ -z "$RUN_LOG" ]]; then
    RUN_LOG=$(detect_run_log)
fi
if [[ -n "$RUN_LOG" ]]; then
    RUN_LOG=$(readlink -f "$RUN_LOG" 2>/dev/null || printf '%s' "$RUN_LOG")
    if [[ ! -f "$RUN_LOG" ]]; then
        echo "Warning: run log does not exist; progress and ETA will be unavailable: $RUN_LOG" >&2
        RUN_LOG=""
    fi
fi

START_STAMP=$(date '+%Y%m%d_%H%M%S')
HUMAN_LOG="$OUTPUT_DIR/resource_${TARGET_PID}_${START_STAMP}.log"
CSV_LOG="$OUTPUT_DIR/resource_${TARGET_PID}_${START_STAMP}.csv"

CMDLINE=$(tr '\0' ' ' < "/proc/$TARGET_PID/cmdline" 2>/dev/null || true)
PROCESS_START_TICKS=$(awk '{print $22}' "/proc/$TARGET_PID/stat" 2>/dev/null || true)

csv_row() {
    local value escaped first=1
    for value in "$@"; do
        escaped=${value//$'\r'/}
        escaped=${escaped//\"/\"\"}
        if [[ $first -eq 0 ]]; then
            printf ',' >> "$CSV_LOG"
        fi
        printf '"%s"' "$escaped" >> "$CSV_LOG"
        first=0
    done
    printf '\n' >> "$CSV_LOG"
}

format_duration() {
    local total=${1:-0}
    local days hours minutes seconds
    days=$((total / 86400))
    hours=$(((total % 86400) / 3600))
    minutes=$(((total % 3600) / 60))
    seconds=$((total % 60))
    printf '%dd %02dh %02dm %02ds' "$days" "$hours" "$minutes" "$seconds"
}

read_status_value() {
    local key=$1
    awk -v key="$key" '$1 == key ":" {print $2; exit}' "/proc/$TARGET_PID/status" 2>/dev/null
}

read_io_value() {
    local key=$1
    awk -v key="$key" '$1 == key ":" {print $2; exit}' "/proc/$TARGET_PID/io" 2>/dev/null
}

csv_row \
    timestamp unix_time process_elapsed_s process_stat process_cpu_percent process_memory_percent \
    rss_kib vm_hwm_kib vm_swap_kib vsz_kib threads fd_count \
    read_bytes write_bytes read_bytes_per_s write_bytes_per_s \
    system_load_1m system_memory_available_kib system_memory_total_kib \
    gpu_index gpu_util_percent gpu_memory_used_mib gpu_memory_total_mib \
    gpu_temperature_c gpu_power_w process_gpu_memory_mib \
    completed_epochs expected_epochs progress_percent eta_seconds estimated_finish \
    current_batch current_weather current_iteration current_epoch progress_line \
    run_log_size_bytes run_log_modified_epoch

{
    echo "JMDA-Net resource monitor"
    echo "monitor_started=$(date '+%F %T %Z')"
    echo "pid=$TARGET_PID"
    echo "process_cwd=$PROCESS_CWD"
    echo "command=$CMDLINE"
    echo "run_log=${RUN_LOG:-not-detected}"
    echo "interval_seconds=$INTERVAL_SECONDS"
    echo "expected_total_epochs=$TOTAL_EPOCHS"
    echo "max_samples=$MAX_SAMPLES"
    echo "csv_log=$CSV_LOG"
    echo
} | tee -a "$HUMAN_LOG"

PREVIOUS_SAMPLE_TIME=""
PREVIOUS_READ_BYTES=""
PREVIOUS_WRITE_BYTES=""
SAMPLE_INDEX=0

finish_monitor() {
    {
        echo
        echo "monitor_stopped=$(date '+%F %T %Z')"
        echo "samples_recorded=$SAMPLE_INDEX"
        echo "human_log=$HUMAN_LOG"
        echo "csv_log=$CSV_LOG"
    } | tee -a "$HUMAN_LOG"
}
trap finish_monitor EXIT
trap 'exit 130' INT TERM

while true; do
    if [[ ! -d "/proc/$TARGET_PID" ]]; then
        echo "$(date '+%F %T') process_exited pid=$TARGET_PID" | tee -a "$HUMAN_LOG"
        break
    fi

    CURRENT_START_TICKS=$(awk '{print $22}' "/proc/$TARGET_PID/stat" 2>/dev/null || true)
    if [[ -n "$PROCESS_START_TICKS" && "$CURRENT_START_TICKS" != "$PROCESS_START_TICKS" ]]; then
        echo "$(date '+%F %T') pid_reused pid=$TARGET_PID; stopping monitor" | tee -a "$HUMAN_LOG"
        break
    fi

    SAMPLE_INDEX=$((SAMPLE_INDEX + 1))
    TIMESTAMP=$(date '+%F %T %Z')
    UNIX_TIME=$(date '+%s')

    PROCESS_ROW=$(ps -p "$TARGET_PID" -o etimes=,stat=,%cpu=,%mem=,rss=,vsz=,nlwp= 2>/dev/null || true)
    read -r PROCESS_ELAPSED PROCESS_STAT PROCESS_CPU PROCESS_MEMORY RSS_KIB VSZ_KIB THREADS_PS <<< "$PROCESS_ROW"

    VM_HWM_KIB=$(read_status_value VmHWM)
    VM_SWAP_KIB=$(read_status_value VmSwap)
    THREADS_STATUS=$(read_status_value Threads)
    THREADS=${THREADS_STATUS:-${THREADS_PS:-}}
    FD_COUNT=$(find "/proc/$TARGET_PID/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l | tr -d ' ')

    READ_BYTES=$(read_io_value read_bytes)
    WRITE_BYTES=$(read_io_value write_bytes)
    READ_RATE=""
    WRITE_RATE=""
    if [[ -n "$PREVIOUS_SAMPLE_TIME" \
          && "$READ_BYTES" =~ ^[0-9]+$ \
          && "$WRITE_BYTES" =~ ^[0-9]+$ \
          && "$PREVIOUS_READ_BYTES" =~ ^[0-9]+$ \
          && "$PREVIOUS_WRITE_BYTES" =~ ^[0-9]+$ ]]; then
        DELTA_SECONDS=$((UNIX_TIME - PREVIOUS_SAMPLE_TIME))
        if [[ $DELTA_SECONDS -gt 0 ]]; then
            READ_RATE=$(((READ_BYTES - PREVIOUS_READ_BYTES) / DELTA_SECONDS))
            WRITE_RATE=$(((WRITE_BYTES - PREVIOUS_WRITE_BYTES) / DELTA_SECONDS))
        fi
    fi
    PREVIOUS_SAMPLE_TIME=$UNIX_TIME
    PREVIOUS_READ_BYTES=$READ_BYTES
    PREVIOUS_WRITE_BYTES=$WRITE_BYTES

    SYSTEM_LOAD=$(awk '{print $1}' /proc/loadavg 2>/dev/null || true)
    SYSTEM_MEMORY_AVAILABLE=$(awk '$1 == "MemAvailable:" {print $2; exit}' /proc/meminfo 2>/dev/null || true)
    SYSTEM_MEMORY_TOTAL=$(awk '$1 == "MemTotal:" {print $2; exit}' /proc/meminfo 2>/dev/null || true)

    GPU_UUID=""
    PROCESS_GPU_MEMORY=""
    GPU_INDEX=""
    GPU_UTIL=""
    GPU_MEMORY_USED=""
    GPU_MEMORY_TOTAL=""
    GPU_TEMPERATURE=""
    GPU_POWER=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        while IFS=',' read -r candidate_uuid candidate_pid candidate_memory; do
            candidate_uuid=$(echo "$candidate_uuid" | xargs)
            candidate_pid=$(echo "$candidate_pid" | xargs)
            candidate_memory=$(echo "$candidate_memory" | xargs)
            if [[ "$candidate_pid" == "$TARGET_PID" ]]; then
                GPU_UUID=$candidate_uuid
                PROCESS_GPU_MEMORY=$candidate_memory
                break
            fi
        done < <(
            nvidia-smi \
                --query-compute-apps=gpu_uuid,pid,used_gpu_memory \
                --format=csv,noheader,nounits 2>/dev/null || true
        )

        if [[ -n "$GPU_UUID" ]]; then
            GPU_DETAIL=$(
                nvidia-smi \
                    --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
                    --format=csv,noheader,nounits 2>/dev/null |
                awk -F',' -v target="$GPU_UUID" '
                    {
                        for (i = 1; i <= NF; i++) {
                            gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                        }
                    }
                    $2 == target {
                        printf "%s|%s|%s|%s|%s|%s", $1, $3, $4, $5, $6, $7
                        exit
                    }
                '
            )
            IFS='|' read -r GPU_INDEX GPU_UTIL GPU_MEMORY_USED GPU_MEMORY_TOTAL GPU_TEMPERATURE GPU_POWER <<< "$GPU_DETAIL"
        fi
    fi

    COMPLETED_EPOCHS=""
    CURRENT_BATCH=""
    CURRENT_WEATHER=""
    CURRENT_ITERATION=""
    CURRENT_EPOCH=""
    PROGRESS_LINE=""
    RUN_LOG_SIZE=""
    RUN_LOG_MTIME=""
    PROGRESS_PERCENT=""
    ETA_SECONDS=""
    ESTIMATED_FINISH=""
    ETA_HUMAN="unavailable"

    if [[ -n "$RUN_LOG" && -f "$RUN_LOG" ]]; then
        RUN_LOG_SIZE=$(stat -c '%s' "$RUN_LOG" 2>/dev/null || true)
        RUN_LOG_MTIME=$(stat -c '%Y' "$RUN_LOG" 2>/dev/null || true)
        # One target contains up to five 100-epoch repetitions and normally
        # emits two epoch lines per epoch. A 3000-line tail keeps the current
        # batch/weather header available without rereading the complete log.
        RECENT_LOG=$(tail -n 3000 "$RUN_LOG" 2>/dev/null || true)
        CURRENT_BATCH=$(printf '%s\n' "$RECENT_LOG" | grep -E '^\[Batch [0-9]+/[0-9]+\]' | tail -n 1 || true)
        CURRENT_WEATHER=$(printf '%s\n' "$RECENT_LOG" | grep -E '^目标域天气:' | tail -n 1 || true)
        CURRENT_ITERATION=$(printf '%s\n' "$RECENT_LOG" | grep -E '^迭代 [0-9]+/[0-9]+' | tail -n 1 || true)
        PROGRESS_LINE=$(printf '%s\n' "$RECENT_LOG" | grep -E '^Epoch \[[0-9]+/[0-9]+\] \| (SVD|Loss)' | tail -n 1 || true)
        CURRENT_EPOCH=$(printf '%s\n' "$PROGRESS_LINE" | sed -n 's/^Epoch \[\([0-9][0-9]*\)\/\([0-9][0-9]*\)\].*/\1\/\2/p')
        COMPLETED_EPOCHS=$(grep -cE '^Epoch \[[0-9]+/[0-9]+\] \| Loss:' "$RUN_LOG" 2>/dev/null || true)

        if [[ "$TOTAL_EPOCHS" -gt 0 && "${COMPLETED_EPOCHS:-0}" -gt 0 && "${COMPLETED_EPOCHS:-0}" -le "$TOTAL_EPOCHS" ]]; then
            REMAINING_EPOCHS=$((TOTAL_EPOCHS - COMPLETED_EPOCHS))
            PROGRESS_PERCENT=$(awk -v done="$COMPLETED_EPOCHS" -v total="$TOTAL_EPOCHS" 'BEGIN {printf "%.2f", 100 * done / total}')
            ETA_SECONDS=$(awk -v elapsed="${PROCESS_ELAPSED:-0}" -v done="$COMPLETED_EPOCHS" -v remaining="$REMAINING_EPOCHS" 'BEGIN {printf "%.0f", elapsed * remaining / done}')
            ESTIMATED_FINISH=$(date -d "+${ETA_SECONDS} seconds" '+%F %T %Z' 2>/dev/null || true)
            ETA_HUMAN=$(format_duration "$ETA_SECONDS")
        fi
    fi

    csv_row \
        "$TIMESTAMP" "$UNIX_TIME" "${PROCESS_ELAPSED:-}" "${PROCESS_STAT:-}" "${PROCESS_CPU:-}" "${PROCESS_MEMORY:-}" \
        "${RSS_KIB:-}" "${VM_HWM_KIB:-}" "${VM_SWAP_KIB:-}" "${VSZ_KIB:-}" "${THREADS:-}" "$FD_COUNT" \
        "${READ_BYTES:-}" "${WRITE_BYTES:-}" "$READ_RATE" "$WRITE_RATE" \
        "$SYSTEM_LOAD" "$SYSTEM_MEMORY_AVAILABLE" "$SYSTEM_MEMORY_TOTAL" \
        "$GPU_INDEX" "$GPU_UTIL" "$GPU_MEMORY_USED" "$GPU_MEMORY_TOTAL" \
        "$GPU_TEMPERATURE" "$GPU_POWER" "$PROCESS_GPU_MEMORY" \
        "$COMPLETED_EPOCHS" "$TOTAL_EPOCHS" "$PROGRESS_PERCENT" "$ETA_SECONDS" "$ESTIMATED_FINISH" \
        "$CURRENT_BATCH" "$CURRENT_WEATHER" "$CURRENT_ITERATION" "$CURRENT_EPOCH" "$PROGRESS_LINE" \
        "$RUN_LOG_SIZE" "$RUN_LOG_MTIME"

    RSS_MIB=$(( ${RSS_KIB:-0} / 1024 ))
    VM_HWM_MIB=$(( ${VM_HWM_KIB:-0} / 1024 ))
    VM_SWAP_MIB=$(( ${VM_SWAP_KIB:-0} / 1024 ))
    {
        printf '%s sample=%d pid=%s stat=%s cpu=%s%% rss=%sMiB hwm=%sMiB swap=%sMiB threads=%s fd=%s ' \
            "$TIMESTAMP" "$SAMPLE_INDEX" "$TARGET_PID" "${PROCESS_STAT:-?}" "${PROCESS_CPU:-?}" \
            "$RSS_MIB" "$VM_HWM_MIB" "$VM_SWAP_MIB" "${THREADS:-?}" "$FD_COUNT"
        printf 'gpu=%s util=%s%% process_gpu_mem=%sMiB gpu_mem=%s/%sMiB load1=%s ' \
            "${GPU_INDEX:-?}" "${GPU_UTIL:-?}" "${PROCESS_GPU_MEMORY:-?}" \
            "${GPU_MEMORY_USED:-?}" "${GPU_MEMORY_TOTAL:-?}" "${SYSTEM_LOAD:-?}"
        printf 'epochs=%s/%s progress=%s%% eta=%s finish=%s\n' \
            "${COMPLETED_EPOCHS:-?}" "$TOTAL_EPOCHS" "${PROGRESS_PERCENT:-?}" \
            "$ETA_HUMAN" "${ESTIMATED_FINISH:-?}"
        echo "  ${CURRENT_BATCH:-batch unavailable} | ${CURRENT_WEATHER:-weather unavailable} | ${CURRENT_ITERATION:-iteration unavailable}"
        echo "  ${PROGRESS_LINE:-epoch progress unavailable}"
    } | tee -a "$HUMAN_LOG"

    if [[ "$MAX_SAMPLES" -gt 0 && "$SAMPLE_INDEX" -ge "$MAX_SAMPLES" ]]; then
        break
    fi
    sleep "$INTERVAL_SECONDS"
done

