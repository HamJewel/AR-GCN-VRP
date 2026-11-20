#!/usr/bin/env bash
# LKH labels for the synthetic test suites: STSP (symmetric) and ATSP (asymmetric).
# Writes tours_<PROBLEM>_lkh.npy / lengths_<PROBLEM>_lkh.npy next to the instances.
# usage: bash scripts/lkh_solver_test.sh [suite]     suite = all | svrp | avrp
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SUITE="${1:-all}"
DATASET_ROOT="${DATASET_ROOT:-./dataset_generated}"
LKH_FILE="${LKH_FILE:-./data_factory/LKH-3.0.13/LKH}"
MAX_THREADS="${MAX_THREADS:-16}"
MAX_TRIALS="${MAX_TRIALS:-10000}"
RUNS="${RUNS:-1}"
SEED="${SEED:-1234}"
SCALE="${SCALE:-1e6}"
TIME_FACTOR="${TIME_FACTOR:-1}"

case "$SUITE" in
  all|svrp|avrp) ;;
  *) echo 'Suite: all, svrp, avrp' >&2; exit 1 ;;
esac

if [[ "$SUITE" == all || "$SUITE" == svrp ]]; then
  PROBLEM='TSP'
  MAX_CANDIDATES="${MAX_CANDIDATES:-5}"
  SVRP_DISTRIBUTIONS=(uniform clustered implosion explosion)
  SVRP_SIZES=(100 1000 5000 10000)
  if [[ -n "${VRP_SIZE:-}" ]]; then SVRP_SIZES=("$VRP_SIZE"); fi
  for VRP_SIZE in "${SVRP_SIZES[@]}"; do
    for DISTRIBUTION in "${SVRP_DISTRIBUTIONS[@]}"; do
      DATA_DIR="$DATASET_ROOT/SVRP_test/$DISTRIBUTION/$VRP_SIZE"
      if [[ ! -d "$DATA_DIR" ]]; then
        echo "Skip $DATA_DIR: not found" >&2
        continue
      fi
      OUTPUT_DIR="./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}"
      echo "=== STSP $DISTRIBUTION n=$VRP_SIZE"
      "$PYTHON" -u ./data_factory/lkh_solver.py \
        --problem "$PROBLEM" \
        --symmetric \
        --vrp_size "$VRP_SIZE" \
        --max_threads "$MAX_THREADS" \
        --max_candidates "$MAX_CANDIDATES" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --lkh_file "$LKH_FILE" \
        --max_trials "$MAX_TRIALS" \
        --time_factor "$TIME_FACTOR" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --scale "$SCALE" \
        --save_solution
    done
  done
fi

if [[ "$SUITE" == all || "$SUITE" == avrp ]]; then
  PROBLEM='TSP'
  MAX_CANDIDATES="${MAX_CANDIDATES:-6}"
  AVRP_DISTRIBUTIONS=(tmat)
  AVRP_SIZES=(50 100 500 1000)
  if [[ -n "${VRP_SIZE:-}" ]]; then AVRP_SIZES=("$VRP_SIZE"); fi
  for DISTRIBUTION in "${AVRP_DISTRIBUTIONS[@]}"; do
    for VRP_SIZE in "${AVRP_SIZES[@]}"; do
      DATA_DIR="$DATASET_ROOT/AVRP_test/$DISTRIBUTION/$VRP_SIZE"
      if [[ ! -d "$DATA_DIR" ]]; then
        echo "Skip $DATA_DIR: not found" >&2
        continue
      fi
      OUTPUT_DIR="./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}"
      echo "=== ATSP $DISTRIBUTION n=$VRP_SIZE"
      "$PYTHON" -u ./data_factory/lkh_solver.py \
        --problem "$PROBLEM" \
        --vrp_size "$VRP_SIZE" \
        --max_threads "$MAX_THREADS" \
        --max_candidates "$MAX_CANDIDATES" \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --lkh_file "$LKH_FILE" \
        --max_trials "$MAX_TRIALS" \
        --time_factor "$TIME_FACTOR" \
        --runs "$RUNS" \
        --seed "$SEED" \
        --scale "$SCALE" \
        --save_solution
    done
  done
fi
