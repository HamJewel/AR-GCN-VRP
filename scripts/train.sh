python ./train/main.py \
    --problem CVRP \
    --aggregation mean \
    --d_model 128 \
    --d_gate 128 \
    --n_encoders 6 \
    --max_k 100 \
    --n_epoch 3 \
    --learning_rate 5e-4 \
    --data_dir ./dataset/SVRP_train/uniform/100 \
    --data_size 200000 \
    --batch_size 100 \
    --output_dir ./output \
    --log_steps 1 \
    --symmetric \
    --save_name test

# python ./train/main.py \
#     --problem TSP \
#     --aggregation mean \
#     --d_model 128 \
#     --d_gate 128 \
#     --n_encoders 6 \
#     --max_k 0 \
#     --n_epoch 3 \
#     --learning_rate 5e-4 \
#     --data_dir ./dataset/AVRP_train/tmat/100 \
#     --data_size 200000 \
#     --batch_size 100 \
#     --output_dir ./output \
#     --log_steps 1

# python ./train/main.py \
#     --problem TSP \
#     --aggregation mean \
#     --d_model 128 \
#     --d_gate 128 \
#     --n_encoders 6 \
#     --max_k 50 \
#     --n_epoch 3 \
#     --learning_rate 5e-4 \
#     --data_dir ./dataset/SVRP_train/uniform/100 \
#     --data_size 200000 \
#     --batch_size 200 \
#     --output_dir ./output \
#     --log_steps 1 \
#     --save_name STSP-128-128-6-50-3-mean-woReset \
#     --symmetric

# python ./train/main.py \
#     --problem CVRP \
#     --aggregation mean \
#     --d_model 128 \
#     --d_gate 128 \
#     --n_encoders 6 \
#     --max_k 0 \
#     --n_epoch 3 \
#     --learning_rate 5e-4 \
#     --data_dir ./dataset/AVRP_train/tmat/100 \
#     --data_size 200000 \
#     --batch_size 100 \
#     --output_dir ./output \
#     --log_steps 1

# /usr/bin/shutdown
