VRP_SIZES=(50 100 500 1000)
DISTRIBUTION_OPTIONS=('tmat' 'perturbation')
DISTRIBUTION='perturbation'
for VRP_SIZE in "${VRP_SIZES[@]}"; do
  DATA_SIZE=$((10000 / VRP_SIZE))
  python ./data_factory/generator.py \
    --vrp_size ${VRP_SIZE} \
    --capacity 50 \
    --min_demand 1 \
    --max_demand 10 \
    --data_size ${DATA_SIZE} \
    --output_dir ./dataset/AVRP_test/${DISTRIBUTION}/${VRP_SIZE} \
    --svrp_distribution 'uniform' \
    --avrp_distribution ${DISTRIBUTION} \
    --min_tmat 0 \
    --max_tmat 1000000 \
    --scale 1e6 \
    --perturbation_factor 0.5 \
    --device 'cuda'
done