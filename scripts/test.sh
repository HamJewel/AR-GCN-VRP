# DISTRIBUTIONS=(uniform)
# MODEL_PATHS=(./output/SCVRP-128-128-6-100-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#         python ./test/main.py \
#             --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 \
#             --data_sizes 5 5 5 \
#             --batch_sizes 1 1 1 \
#             --n_rollouts 8 8 8 \
#             --output_dir ${MODEL_PATH} \
#             --ckpt_path ${MODEL_PATH}/ckpts/steps-1000.pth \
#             --cfg_path ${MODEL_PATH}/configs.json
#     done

# DISTRIBUTIONS=(uniform clustered explosion implosion)
# MODEL_PATHS=(./output/SCVRP-128-128-6-100-3-mean ./output/SCVRP-128-128-6-50-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#         python ./test/main.py \
#             --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
#             --data_sizes -1 -1 -1 -1 \
#             --batch_sizes 1 1 1 1 \
#             --n_rollouts 8 8 8 8 \
#             --output_dir ${MODEL_PATH} \
#             --ckpt_path ${MODEL_PATH}/ckpts/epoch-3.pth \
#             --cfg_path ${MODEL_PATH}/configs.json
#     done
    
#     python ./test/main.py \
#         --data_dirs ./dataset/SCVRPLIB \
#         --data_sizes -1 \
#         --batch_sizes 1 \
#         --n_rollouts 8 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json \
#         --is_lib
# done

DISTRIBUTIONS=(tmat)
MODEL_PATHS=(./output/ATSP-128-128-6-0-3-mean)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
        python ./test/main.py \
            --data_dirs ./dataset/AVRP_test/${DISTRIBUTION}/1000 \
            --data_sizes -1 \
            --batch_sizes 1 \
            --n_rollouts 8 \
            --output_dir ${MODEL_PATH} \
            --ckpt_path ${MODEL_PATH}/ckpts/epoch-3.pth \
            --cfg_path ${MODEL_PATH}/configs.json
    done
    
    # python ./test/main.py \
    #     --data_dirs ./dataset/SCVRPLIB \
    #     --data_sizes -1 \
    #     --batch_sizes 1 \
    #     --n_rollouts 8 \
    #     --output_dir ${MODEL_PATH} \
    #     --ckpt_path ${MODEL_PATH}/ckpts/epoch-${EPOCH}.pth \
    #     --cfg_path ${MODEL_PATH}/configs.json \
    #     --is_lib
done

# MODEL_PATHS=(./output/SCVRP-128-128-6-50-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#         python ./test/main.py \
#             --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 \
#             --data_sizes 10 \
#             --batch_sizes 1 \
#             --n_rollouts 8 \
#             --output_dir ${MODEL_PATH} \
#             --ckpt_path ${MODEL_PATH}/ckpts/epoch-3.pth \
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