#!/usr/bin/env bash
# Synthetic AVRP test sets: TMAT distance matrices (edges.npy + demands.npy).
# Each instance is scaled down by --scale, so the stored edges are in [0, 1].
# usage: [VRP_SIZE=100] [DATA_SIZE=200] bash scripts/generator_AVRP_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-./dataset_generated}"
COUNT="${DATA_SIZE:-}"
DISTRIBUTION='tmat'
VRP_SIZES=(50 100 500 1000)
if [[ -n "${VRP_SIZE:-}" ]]; then VRP_SIZES=("$VRP_SIZE"); fi

for VRP_SIZE in "${VRP_SIZES[@]}"; do
  DATA_SIZE="${COUNT:-$((10000 / VRP_SIZE))}"
  if [[ -e "$DATASET_ROOT/AVRP_test/$DISTRIBUTION/$VRP_SIZE" ]]; then
    echo "Output already exists: $DATASET_ROOT/AVRP_test/$DISTRIBUTION/$VRP_SIZE" >&2
    exit 1
  fi
  echo "=== AVRP $DISTRIBUTION n=$VRP_SIZE"
  "$PYTHON" -u ./data_factory/generator.py \
    --vrp_size "$VRP_SIZE" \
    --capacity 50 \
    --min_demand 1 \
    --max_demand 10 \
    --data_size "$DATA_SIZE" \
    --output_dir "$DATASET_ROOT/AVRP_test/$DISTRIBUTION/$VRP_SIZE" \
    --distribution "$DISTRIBUTION" \
    --min_tmat 0 \
    --max_tmat 1000000 \
    --scale 1e6 \
    --device cuda
done
