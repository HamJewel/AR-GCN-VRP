MAX_THREADS=30
MAX_TRIALS=10000
RUNS=1
SEED=1234
SCALE=1e6
TIME_FACTOR=1
LKH_FILE=./data_factory/LKH-3.0.13/LKH


MAX_CANDIDATES=5
VRP_SIZES=(100 1000 5000 10000)
DISTRIBUTIONS=('uniform' 'clustered' 'implosion' 'explosion')
PROBLEM=TSP
for VRP_SIZE in "${VRP_SIZES[@]}"; do
  for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
    DATA_DIR=./dataset/SVRP_test/${DISTRIBUTION}/${VRP_SIZE}
    OUTPUT_DIR=./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}
    python ./data_factory/lkh_solver.py \
      --problem ${PROBLEM} \
      --vrp_size ${VRP_SIZE} \
      --max_threads ${MAX_THREADS} \
      --max_candidates ${MAX_CANDIDATES} \
      --data_dir ${DATA_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --lkh_file ${LKH_FILE} \
      --max_trials ${MAX_TRIALS} \
      --time_factor ${TIME_FACTOR} \
      --runs ${RUNS} \
      --seed ${SEED} \
      --scale ${SCALE} \
      --save_solution \
      --symmetric
  done
done

PROBLEM=CVRP
for VRP_SIZE in "${VRP_SIZES[@]}"; do
  for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
    DATA_DIR=./dataset/SVRP_test/${DISTRIBUTION}/${VRP_SIZE}
    OUTPUT_DIR=./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}
    python ./data_factory/lkh_solver.py \
      --problem ${PROBLEM} \
      --vrp_size ${VRP_SIZE} \
      --max_threads ${MAX_THREADS} \
      --max_candidates ${MAX_CANDIDATES} \
      --data_dir ${DATA_DIR} \
      --output_dir ${OUTPUT_DIR} \
      --lkh_file ${LKH_FILE} \
      --max_trials ${MAX_TRIALS} \
      --time_factor ${TIME_FACTOR} \
      --runs ${RUNS} \
      --seed ${SEED} \
      --scale ${SCALE} \
      --save_solution \
      --symmetric \
      --special
  done
done


MAX_CANDIDATES=6
VRP_SIZES=(50 100 500 1000)
DISTRIBUTION=tmat
PROBLEM=TSP
for VRP_SIZE in "${VRP_SIZES[@]}"; do
  DATA_DIR=./dataset/SVRP_test/${DISTRIBUTION}/${VRP_SIZE}
  OUTPUT_DIR=./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}
  python ./data_factory/lkh_solver.py \
    --problem ${PROBLEM} \
    --vrp_size ${VRP_SIZE} \
    --max_threads ${MAX_THREADS} \
    --max_candidates ${MAX_CANDIDATES} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lkh_file ${LKH_FILE} \
    --max_trials ${MAX_TRIALS} \
    --time_factor ${TIME_FACTOR} \
    --runs ${RUNS} \
    --seed ${SEED} \
    --scale ${SCALE} \
    --save_solution
done

PROBLEM=CVRP
for VRP_SIZE in "${VRP_SIZES[@]}"; do
  DATA_DIR=./dataset/SVRP_test/${DISTRIBUTION}/${VRP_SIZE}
  OUTPUT_DIR=./data_factory/output/LKH_${PROBLEM}_${VRP_SIZE}
  python ./data_factory/lkh_solver.py \
    --problem ${PROBLEM} \
    --vrp_size ${VRP_SIZE} \
    --max_threads ${MAX_THREADS} \
    --max_candidates ${MAX_CANDIDATES} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --lkh_file ${LKH_FILE} \
    --max_trials ${MAX_TRIALS} \
    --time_factor ${TIME_FACTOR} \
    --runs ${RUNS} \
    --seed ${SEED} \
    --scale ${SCALE} \
    --save_solution \
    --special
done