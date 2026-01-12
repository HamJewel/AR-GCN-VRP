VRP_SIZE=100
DATA_SIZE=200000
DISTRIBUTION_OPTIONS=('uniform' 'explosion' 'implosion')
DISTRIBUTION='uniform'
python ./data_factory/generator.py \
  --vrp_size ${VRP_SIZE} \
  --capacity 50 \
  --min_demand 1 \
  --max_demand 10 \
  --data_size ${DATA_SIZE} \
  --output_dir ./dataset/SVRP_train/${DISTRIBUTION}/${VRP_SIZE} \
  --svrp_distribution ${DISTRIBUTION} \
  --cluster_num 3 \
  --cluster_diversity 10 \
  --min_range 0.1 \
  --max_range 0.5 \
  --explosion_rate 10 \
  --symmetric