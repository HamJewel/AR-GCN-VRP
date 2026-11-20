#!/usr/bin/env bash
# Train the six presets; each preset name is problem-hidden_dim-layers-k-epochs-aggregation.
# usage: bash scripts/train.sh [gpu] [model|all] [dataset_root] [output_dir]
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
MODEL="${2:-all}"
DATASET_ROOT="${3:-dataset}"
OUTPUT_DIR="${4:-output/train}"
DATA_SIZE="${DATA_SIZE:-200000}"
BATCH_SIZE="${BATCH_SIZE:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
THREADS="${THREADS:-16}"
SEED="${SEED:-1234}"

case "$MODEL" in
  all|STSP-128-6-50-3-mean|STSP-128-6-100-3-mean|SCVRP-128-6-50-3-mean|SCVRP-128-6-100-3-mean|ATSP-128-6-0-3-mean|ACVRP-128-6-0-3-mean) ;;
  *) echo 'Unknown model' >&2; exit 1 ;;
esac

for model in STSP-128-6-50-3-mean STSP-128-6-100-3-mean SCVRP-128-6-50-3-mean SCVRP-128-6-100-3-mean ATSP-128-6-0-3-mean ACVRP-128-6-0-3-mean; do
  [[ "$MODEL" == all || "$MODEL" == "$model" ]] || continue
  SYMMETRIC_OPT=()
  case "$model" in
    STSP*) PROBLEM='TSP'; DATA_DIR="$DATASET_ROOT/SVRP_train/uniform/100"; SYMMETRIC_OPT=(--symmetric) ;;
    SCVRP*) PROBLEM='CVRP'; DATA_DIR="$DATASET_ROOT/SVRP_train/uniform/100"; SYMMETRIC_OPT=(--symmetric) ;;
    ATSP*) PROBLEM='TSP'; DATA_DIR="$DATASET_ROOT/AVRP_train/tmat/100" ;;
    ACVRP*) PROBLEM='CVRP'; DATA_DIR="$DATASET_ROOT/AVRP_train/tmat/100" ;;
  esac
  IFS=- read -r task d_model n_encoders k n_epoch aggregation <<< "$model"
  echo "=== ${model} ${PROBLEM} n=100"
  "$PYTHON" -u AR_GCN/train/main.py \
    --problem "$PROBLEM" \
    --aggregation "$aggregation" \
    --d_model "$d_model" \
    --n_encoders "$n_encoders" \
    --max_k "$k" \
    --n_epoch "$n_epoch" \
    --learning_rate "$LEARNING_RATE" \
    --baseline lkh \
    --data_dir "$DATA_DIR" \
    --data_size "$DATA_SIZE" \
    --batch_size "$BATCH_SIZE" \
    --output_dir "$OUTPUT_DIR" \
    --save_name "$model" \
    --seed "$SEED" \
    --threads "$THREADS" \
    --cuda_id "$GPU" \
    --log_steps 1 \
    "${SYMMETRIC_OPT[@]}"
done
