EPOCH=3
DISTRIBUTION=tmat
MODEL_PATH=./output/SCVRP-128-128-6-100-3-mean
python ./test/main.py \
    --data_dirs dataset/SVRP_test/uniform/1000 \
    --data_sizes 1 \
    --batch_sizes 1 \
    --n_rollouts 1 \
    --max_k 0 \
    --output_dir ${MODEL_PATH} \
    --ckpt_path ${MODEL_PATH}/ckpts/steps-6000.pth \
    --cfg_path ${MODEL_PATH}/configs.json
# MODEL_PATHS=(./output/ATSP-128-128-6-0-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/AVRP_test/${DISTRIBUTION}/100 \
#         --data_sizes 1 \
#         --batch_sizes 1 \
#         --n_rollouts 1 \
#         --output_dir ${MODEL_PATH}/gate \
#         --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
    
    # python ./test/main.py \
    #     --data_dirs ./dataset/ATSPLIB \
    #     --data_sizes -1 \
    #     --batch_sizes 1 \
    #     --n_rollouts 8 \
    #     --output_dir ${MODEL_PATH} \
    #     --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
    #     --cfg_path ${MODEL_PATH}/configs.json \
    #     --is_lib
# done

# MODEL_PATHS=(./output/ACVRP-128-128-6-0-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/AVRP_test/${DISTRIBUTION}/1000 \
#         --data_sizes -1 \
#         --batch_sizes 1 \
#         --n_rollouts 6 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
# done

# MODEL_PATHS=(./output/ATSP-128-128-6-0-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/AVRP_test/${DISTRIBUTION}/1000 \
#         --data_sizes -1 \
#         --batch_sizes 1 \
#         --n_rollouts 6 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
# done

# DISTRIBUTIONS=(uniform clustered explosion implosion)
# MODEL_PATHS=(./output/STSP-128-128-6-50-3-mean-woReset)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#         python ./test/main.py \
#             --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
#             --data_sizes -1 -1 -1 -1 \
#             --batch_sizes 1 1 1 1 \
#             --n_rollouts 8 8 8 8 \
#             --output_dir ${MODEL_PATH} \
#             --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#             --cfg_path ${MODEL_PATH}/configs.json
#     done
    
#     python ./test/main.py \
#         --data_dirs ./dataset/STSPLIB \
#         --data_sizes -1 \
#         --batch_sizes 1 \
#         --n_rollouts 8 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json \
#         --is_lib
# done

# MODEL_PATHS=(./output/SCVRP-128-128-6-100-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#         python ./test/main.py \
#             --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
#             --data_sizes -1 -1 -1 -1 \
#             --batch_sizes 1 1 1 1 \
#             --n_rollouts 8 8 8 8 \
#             --output_dir ${MODEL_PATH} \
#             --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#             --cfg_path ${MODEL_PATH}/configs.json
#     done
    
    # python ./test/main.py \
    #     --data_dirs ./dataset/SCVRPLIB \
    #     --data_sizes -1 \
    #     --batch_sizes 1 \
    #     --n_rollouts 8 \
    #     --output_dir ${MODEL_PATH} \
    #     --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
    #     --cfg_path ${MODEL_PATH}/configs.json \
    #     --is_lib
# done

# /usr/bin/shutdown