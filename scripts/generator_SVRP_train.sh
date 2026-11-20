#!/usr/bin/env bash
# Synthetic SVRP training set: 200k uniform instances of 100 nodes (nodes.npy + demands.npy).
# usage: [VRP_SIZE=100] [DATA_SIZE=200000] bash scripts/generator_SVRP_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-./dataset}"
VRP_SIZE="${VRP_SIZE:-100}"
DATA_SIZE="${DATA_SIZE:-200000}"
DISTRIBUTION="${DISTRIBUTION:-uniform}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASET_ROOT/SVRP_train/$DISTRIBUTION/$VRP_SIZE}"

echo "=== SVRP $DISTRIBUTION n=$VRP_SIZE size=$DATA_SIZE"
"$PYTHON" -u ./data_factory/generator.py \
  --vrp_size "$VRP_SIZE" \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size "$DATA_SIZE" \
  --output_dir "$OUTPUT_DIR" \
  --distribution "$DISTRIBUTION" \
  --cluster_num 3 \
  --cluster_diversity 10 \
  --min_range 0.1 \
  --max_range 0.5 \
  --explosion_rate 10
