#!/usr/bin/env bash
# Evaluate the six presets on the synthetic suites and on the TSPLIB/CVRPLIB sets.
# usage: bash scripts/test.sh [gpu] [suite|all] [model|all] [data_size|-1] [dataset_root]
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
GPU="${1:-0}"
SUITE="${2:-all}"
MODEL="${3:-all}"
SIZE="${4:--1}"
DATASET_ROOT="${5:-dataset}"
CKPT_ROOT="${CKPT_ROOT:-output/train}"
OUTPUT_DIR="${OUTPUT_DIR:-output/test}"

case "$SUITE" in
  all|synthetic|lib|small) ;;
  *) echo 'Suite: all, synthetic, lib, small' >&2; exit 1 ;;
esac
case "$MODEL" in
  all|STSP-128-6-50-3-mean|STSP-128-6-100-3-mean|SCVRP-128-6-50-3-mean|SCVRP-128-6-100-3-mean|ATSP-128-6-0-3-mean|ACVRP-128-6-0-3-mean) ;;
  *) echo 'Unknown model' >&2; exit 1 ;;
esac

for model in STSP-128-6-50-3-mean STSP-128-6-100-3-mean SCVRP-128-6-50-3-mean SCVRP-128-6-100-3-mean ATSP-128-6-0-3-mean ACVRP-128-6-0-3-mean; do
  [[ "$MODEL" == all || "$MODEL" == "$model" ]] || continue
  case "$model" in
    STSP*) family=SVRP_test; distributions=(uniform clustered explosion implosion); scales=(100 1000 5000 10000); baseline=lkh; libs=(STSPLIB) ;;
    SCVRP*) family=SVRP_test; distributions=(uniform clustered explosion implosion); scales=(100 1000 5000 10000); baseline=hgs; libs=(SCVRPLIB/X SCVRPLIB/XL) ;;
    ATSP*) family=AVRP_test; distributions=(tmat); scales=(50 100 500 1000); baseline=lkh; libs=(ATSPLIB) ;;
    ACVRP*) family=AVRP_test; distributions=(tmat); scales=(50 100 500 1000); baseline=reference; libs=() ;;
  esac
  if [[ "$SUITE" != lib ]]; then
    if [[ "$SUITE" == small ]]; then distributions=("${distributions[0]}"); scales=("${scales[0]}"); fi
    for distribution in "${distributions[@]}"; do
      for n in "${scales[@]}"; do
        rollouts=8
        if [[ "$family" == AVRP_test && "$n" == 1000 ]]; then rollouts=6; fi
        DATA_DIR="$DATASET_ROOT/$family/$distribution/$n"
        if [[ ! -d "$DATA_DIR" ]]; then
          echo "Skip $DATA_DIR: not found" >&2
          continue
        fi
        echo "=== ${model} ${distribution} n=${n}"
        "$PYTHON" -u AR_GCN/test/main.py \
          --data_dirs "$DATA_DIR" \
          --data_sizes "$SIZE" \
          --batch_sizes 1 \
          --n_rollouts "$rollouts" \
          --baseline "$baseline" \
          --cuda_id "$GPU" \
          --output_dir "$OUTPUT_DIR/$model/$distribution/$n" \
          --ckpt_path "$CKPT_ROOT/$model/ckpts/epoch-3.pth" \
          --cfg_path "$CKPT_ROOT/$model/configs.json"
      done
    done
  fi
  if [[ "$SUITE" == lib || "$SUITE" == all ]]; then
    for lib in "${libs[@]}"; do
      echo "=== ${model} ${lib}"
      "$PYTHON" -u AR_GCN/test/main.py \
        --is_lib \
        --data_dirs "$DATASET_ROOT/$lib" \
        --data_sizes -1 \
        --batch_sizes 1 \
        --n_rollouts 8 \
        --cuda_id "$GPU" \
        --output_dir "$OUTPUT_DIR/$model/$lib" \
        --ckpt_path "$CKPT_ROOT/$model/ckpts/epoch-3.pth" \
        --cfg_path "$CKPT_ROOT/$model/configs.json"
    done
  fi
done
