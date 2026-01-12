VRP_SIZE=10000
DATA_SIZE=$((100000 / VRP_SIZE))
DISTRIBUTION_OPTIONS=('uniform' 'explosion' 'implosion')

for DISTRIBUTION in "${DISTRIBUTION_OPTIONS[@]}"; do
  python ./data_factory/generator.py \
    --vrp_size ${VRP_SIZE} \
    --capacity 50 \
    --min_demand 1 \
    --max_demand 10 \
    --data_size ${DATA_SIZE} \
    --output_dir ./dataset/SVRP_test/${DISTRIBUTION}/${VRP_SIZE} \
    --svrp_distribution ${DISTRIBUTION} \
    --cluster_num 3 \
    --cluster_diversity 10 \
    --min_range 0.1 \
    --max_range 0.5 \
    --explosion_rate 10 \
    --symmetric
done

DATA_SIZE=$((DATA_SIZE / 2))
DISTRIBUTION='clustered'
python ./data_factory/generator.py \
  --vrp_size ${VRP_SIZE} \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size ${DATA_SIZE} \
  --output_dir ./dataset/SVRP_test/${DISTRIBUTION}1/${VRP_SIZE} \
  --svrp_distribution ${DISTRIBUTION} \
  --cluster_num 3 \
  --cluster_diversity 10 \
  --min_range 0.1 \
  --max_range 0.5 \
  --explosion_rate 10 \
  --symmetric

python ./data_factory/generator.py \
  --vrp_size ${VRP_SIZE} \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size ${DATA_SIZE} \
  --output_dir ./dataset/SVRP_test/${DISTRIBUTION}2/${VRP_SIZE} \
  --svrp_distribution ${DISTRIBUTION} \
  --cluster_num 7 \
  --cluster_diversity 50 \
  --min_range 0.1 \
  --max_range 0.5 \
  --explosion_rate 10 \
  --symmetric