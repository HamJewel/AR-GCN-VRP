#!/usr/bin/env bash
# Synthetic SVRP test sets: uniform, clustered, explosion, implosion (nodes.npy + demands.npy).
# "clustered" is the concatenation of clustered1 (3 clusters) and clustered2 (7 clusters);
# the two halves are removed once the merged set is written.
# usage: [VRP_SIZE=10000] [DATA_SIZE=10] bash scripts/generator_SVRP_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
DATASET_ROOT="${DATASET_ROOT:-./dataset_generated}"
VRP_SIZE="${VRP_SIZE:-10000}"
DATA_SIZE="${DATA_SIZE:-$((100000 / VRP_SIZE))}"

if ((DATA_SIZE < 2 || DATA_SIZE % 2)); then
  echo "DATA_SIZE must be a positive even number" >&2
  exit 1
fi

for DISTRIBUTION in uniform clustered clustered1 clustered2 explosion implosion; do
  if [[ -e "$DATASET_ROOT/SVRP_test/$DISTRIBUTION/$VRP_SIZE" ]]; then
    echo "Output already exists: $DATASET_ROOT/SVRP_test/$DISTRIBUTION/$VRP_SIZE" >&2
    exit 1
  fi
done

for DISTRIBUTION in uniform explosion implosion; do
  echo "=== SVRP $DISTRIBUTION n=$VRP_SIZE"
  (
    if [[ "$DISTRIBUTION" == explosion ]]; then export ATEN_CPU_CAPABILITY=default; fi
    "$PYTHON" -u ./data_factory/generator.py \
      --vrp_size "$VRP_SIZE" \
      --capacity 50 \
      --min_demand 1 \
      --max_demand 10 \
      --data_size "$DATA_SIZE" \
      --output_dir "$DATASET_ROOT/SVRP_test/$DISTRIBUTION/$VRP_SIZE" \
      --distribution "$DISTRIBUTION" \
      --cluster_num 3 \
      --cluster_diversity 10 \
      --min_range 0.1 \
      --max_range 0.5 \
      --explosion_rate 10
  )
done

DATA_SIZE=$((DATA_SIZE / 2))

echo "=== SVRP clustered1 n=$VRP_SIZE"
"$PYTHON" -u ./data_factory/generator.py \
  --vrp_size "$VRP_SIZE" \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size "$DATA_SIZE" \
  --output_dir "$DATASET_ROOT/SVRP_test/clustered1/$VRP_SIZE" \
  --distribution clustered \
  --cluster_num 3 \
  --cluster_diversity 10 \
  --min_range 0.1 \
  --max_range 0.5 \
  --explosion_rate 10

echo "=== SVRP clustered2 n=$VRP_SIZE"
"$PYTHON" -u ./data_factory/generator.py \
  --vrp_size "$VRP_SIZE" \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size "$DATA_SIZE" \
  --output_dir "$DATASET_ROOT/SVRP_test/clustered2/$VRP_SIZE" \
  --distribution clustered \
  --cluster_num 7 \
  --cluster_diversity 50 \
  --min_range 0.1 \
  --max_range 0.5 \
  --explosion_rate 10

echo "=== SVRP clustered n=$VRP_SIZE"
"$PYTHON" - "$DATASET_ROOT/SVRP_test" "$VRP_SIZE" <<'PY'
import sys
from pathlib import Path
import numpy as np

root, size = Path(sys.argv[1]), sys.argv[2]
out = root / 'clustered' / size
out.mkdir(parents=True)
for name in ('nodes.npy', 'demands.npy'):
    np.save(out / name, np.concatenate([np.load(root / f'clustered{i}' / size / name) for i in (1, 2)]))
PY

rm -rf "$DATASET_ROOT/SVRP_test/clustered1" "$DATASET_ROOT/SVRP_test/clustered2"
