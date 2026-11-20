#!/usr/bin/env bash
# Synthetic AVRP training set: 200k TMAT instances of 100 nodes (edges.npy + demands.npy).
# usage: [VRP_SIZE=100] [DATA_SIZE=200000] bash scripts/generator_AVRP_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-./dataset}"
VRP_SIZE="${VRP_SIZE:-100}"
DATA_SIZE="${DATA_SIZE:-200000}"
DISTRIBUTION="${DISTRIBUTION:-tmat}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_ROOT/AVRP_train/$DISTRIBUTION/$VRP_SIZE}"

echo "=== AVRP $DISTRIBUTION n=$VRP_SIZE size=$DATA_SIZE"
"$PYTHON" -u ./data_factory/generator.py \
  --vrp_size "$VRP_SIZE" \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size "$DATA_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --distribution "$DISTRIBUTION" \
  --min_tmat 0 \
  --max_tmat 1000000 \
  --scale 1e6 \
  --device cuda
