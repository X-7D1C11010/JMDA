#!/usr/bin/env bash
set -euo pipefail

GPU_ID=1
MIN_FREE_MIB=10240
CHECK_INTERVAL_SECONDS=60
PROJECT_ROOT=/home/lixiang/lx/JMDA
DATASET_ROOT=/home/lixiang/lx/VDS2Raw/threeclass_ready
OUTPUT_ROOT=/home/lixiang/lx/JMDA/VDS2Raw_JMDA/runs
LOG_FILE=/home/lixiang/lx/JMDA/VDS2Raw_JMDA/train.log
SEEDS=42,43,44,45,46
EXPERIMENTS=(
  "supervised no_tensor_no_ot"
  "supervised no_tensor_with_ot"
  "supervised with_tensor_no_ot"
)

mkdir -p "$OUTPUT_ROOT"
: > "$LOG_FILE"
exec >> "$LOG_FILE" 2>&1
cd "$PROJECT_ROOT"

wait_for_gpu() {
  while true; do
    if ! free_mib=$(nvidia-smi -i "$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]'); then
      free_mib=unknown
    fi
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "$(date '+%F %T') GPU $GPU_ID free=${free_mib}MiB; starting $1"
      return
    fi
    echo "$(date '+%F %T') GPU $GPU_ID free=${free_mib:-unknown}MiB; waiting for ${MIN_FREE_MIB}MiB before $1"
    sleep "$CHECK_INTERVAL_SECONDS"
  done
}

for experiment in "${EXPERIMENTS[@]}"; do
  read -r protocol mode <<< "$experiment"
  mkdir -p "$OUTPUT_ROOT/$protocol"
  wait_for_gpu "$protocol/$mode"
  CUDA_VISIBLE_DEVICES="$GPU_ID" python -u -B VDS2Raw_JMDA/module_ablation.py \
    --dataset_root "$DATASET_ROOT" \
    --output_root "$OUTPUT_ROOT/$protocol" \
    --training_protocol "$protocol" \
    --ablation_mode "$mode" \
    --seeds "$SEEDS" \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda \
    --transport_mode sinkhorn \
    --projection_dim 64 \
    --nir_projection_dim 32 \
    --ais_projection_dim 32 \
    --svd_max_sweeps 10 \
    --svd_tolerance 1e-3 \
    --svd_stat_batch_size 128 \
    --lr_rgb 5e-6 \
    --lr_other 1e-4 \
    --lr_discriminator 5e-5 \
    --discriminator_update_interval 2 \
    --freeze_rgb_backbone \
    --balance_classes \
    --image_augmentation \
    --ot_warmup_epochs 10 \
    --ot_ramp_epochs 5 \
    --evaluate_test
  echo "$(date '+%F %T') completed $protocol/$mode"
done

echo "$(date '+%F %T') all experiments completed"
