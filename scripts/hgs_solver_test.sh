#!/usr/bin/env bash
# HGS labels for the synthetic test suites: SCVRP (symmetric) and ACVRP (asymmetric).
# Writes tours_<PROBLEM>_hgs.npy / lengths_<PROBLEM>_hgs.npy for the symmetric sets and
# tours_<PROBLEM>.npy / lengths_<PROBLEM>.npy for the asymmetric sets.
# usage: bash scripts/hgs_solver_test.sh [suite]     suite = all | svrp | avrp
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SUITE="${1:-all}"
DATASET_ROOT="${DATASET_ROOT:-./dataset_generated}"
TIME_FACTOR="${TIME_FACTOR:-1.0}"
SCALE="${SCALE:-10000}"
MAX_THREADS="${MAX_THREADS:-16}"

SVRP_DISTRIBUTIONS=(uniform clustered explosion implosion)
SVRP_SIZES=(100 1000 5000 10000)
AVRP_DISTRIBUTIONS=(tmat)
AVRP_SIZES=(50 100 500 1000)
if [[ -n "${VRP_SIZE:-}" ]]; then SVRP_SIZES=("$VRP_SIZE"); AVRP_SIZES=("$VRP_SIZE"); fi

case "$SUITE" in
  all|svrp|avrp) ;;
  *) echo 'Suite: all, svrp, avrp' >&2; exit 1 ;;
esac

if [[ "$SUITE" == all || "$SUITE" == svrp ]]; then
  for DISTRIBUTION in "${SVRP_DISTRIBUTIONS[@]}"; do
    for VRP_SIZE in "${SVRP_SIZES[@]}"; do
      DATA_DIR="$DATASET_ROOT/SVRP_test/$DISTRIBUTION/$VRP_SIZE"
      if [[ ! -d "$DATA_DIR" ]]; then
        echo "Skip $DATA_DIR: not found" >&2
        continue
      fi
      echo "=== SCVRP $DISTRIBUTION n=$VRP_SIZE"
      "$PYTHON" -u ./data_factory/hgs_solver.py \
        --problem CVRP \
        --symmetric \
        --vrp_size "$VRP_SIZE" \
        --time_factor "$TIME_FACTOR" \
        --scale "$SCALE" \
        --data_dir "$DATA_DIR" \
        --save_solution \
        --max_threads "$MAX_THREADS"
    done
  done
fi

if [[ "$SUITE" == all || "$SUITE" == avrp ]]; then
  for DISTRIBUTION in "${AVRP_DISTRIBUTIONS[@]}"; do
    for VRP_SIZE in "${AVRP_SIZES[@]}"; do
      DATA_DIR="$DATASET_ROOT/AVRP_test/$DISTRIBUTION/$VRP_SIZE"
      if [[ ! -d "$DATA_DIR" ]]; then
        echo "Skip $DATA_DIR: not found" >&2
        continue
      fi
      echo "=== ACVRP $DISTRIBUTION n=$VRP_SIZE"
      "$PYTHON" -u ./data_factory/hgs_solver.py \
        --problem CVRP \
        --vrp_size "$VRP_SIZE" \
        --time_factor "$TIME_FACTOR" \
        --scale "$SCALE" \
        --data_dir "$DATA_DIR" \
        --save_solution \
        --max_threads "$MAX_THREADS"
    done
  done
fi
