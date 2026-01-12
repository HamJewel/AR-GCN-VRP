PROBLEM=CVRP
VRP_SIZE=100
DATA_SIZE=200000
MAX_THREADS=32
MAX_TRIALS=10000
MAX_CANDIDATES=6
RUNS=1
SEED=1234
SCALE=1e6
DATA_DIR=./dataset/AVRP_train/tmat/${VRP_SIZE}
OUTPUT_DIR=./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}
LKH_FILE=./data_factory/LKH-3.0.13/LKH
python ./data_factory/lkh_solver.py \
  --problem ${PROBLEM} \
  --vrp_size ${VRP_SIZE} \
  --data_size ${DATA_SIZE} \
  --max_threads ${MAX_THREADS} \
  --max_candidates ${MAX_CANDIDATES} \
  --data_dir ${DATA_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --lkh_file ${LKH_FILE} \
  --max_trials ${MAX_TRIALS} \
  --runs ${RUNS} \
  --seed ${SEED} \
  --scale ${SCALE} \
  --save_solution \
  --special