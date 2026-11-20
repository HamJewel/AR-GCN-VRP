#!/usr/bin/env bash
# LKH labels for the training sets: writes tours_<PROBLEM>_lkh.npy / lengths_<PROBLEM>_lkh.npy.
# The default run labels the AVRP training set; SYMMETRIC=1 labels the SVRP one.
# usage: [PROBLEM=CVRP] [SYMMETRIC=0] [DATA_DIR=...] [DATA_SIZE=200000] bash scripts/lkh_solver_train.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
PROBLEM="${PROBLEM:-CVRP}"
SYMMETRIC="${SYMMETRIC:-0}"
VRP_SIZE="${VRP_SIZE:-100}"
DATA_SIZE="${DATA_SIZE:-200000}"
MAX_THREADS="${MAX_THREADS:-16}"
MAX_TRIALS="${MAX_TRIALS:-10000}"
MAX_CANDIDATES="${MAX_CANDIDATES:-6}"
RUNS="${RUNS:-1}"
SEED="${SEED:-1234}"
SCALE="${SCALE:-1e6}"
LKH_FILE="${LKH_FILE:-./data_factory/LKH-3.0.13/LKH}"
if [[ "$SYMMETRIC" == 1 ]]; then
  DEFAULT_DATA_DIR="./dataset/SVRP_train/uniform/${VRP_SIZE}"
else
  DEFAULT_DATA_DIR="./dataset/AVRP_train/tmat/${VRP_SIZE}"
fi
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}}"

SYMMETRIC_OPT=()
if [[ "$SYMMETRIC" == 1 ]]; then SYMMETRIC_OPT=(--symmetric); fi
SPECIAL_OPT=()
if [[ "$PROBLEM" == CVRP ]]; then SPECIAL_OPT=(--special); fi
TIME_OPT=()
if [[ -n "${TIME_FACTOR:-}" ]]; then TIME_OPT=(--time_factor "$TIME_FACTOR"); fi

echo "=== ${PROBLEM} ${DATA_DIR}"
"$PYTHON" -u ./data_factory/lkh_solver.py \
  --problem "$PROBLEM" \
  --vrp_size "$VRP_SIZE" \
  --data_size "$DATA_SIZE" \
  --max_threads "$MAX_THREADS" \
  --max_candidates "$MAX_CANDIDATES" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --lkh_file "$LKH_FILE" \
  --max_trials "$MAX_TRIALS" \
  --runs "$RUNS" \
  --seed "$SEED" \
  --scale "$SCALE" \
  --save_solution \
  "${TIME_OPT[@]}" "${SYMMETRIC_OPT[@]}" "${SPECIAL_OPT[@]}" "$@"
