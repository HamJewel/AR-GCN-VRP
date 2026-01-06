STEPS=3000
# DISTRIBUTION=tmat
# MODEL_PATHS=(./output/ATSP-128-8-6-0-3-mean ./output/ATSP-128-16-6-0-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/AVRP_test/${DISTRIBUTION}/50 ./dataset/AVRP_test/${DISTRIBUTION}/100 ./dataset/AVRP_test/${DISTRIBUTION}/500 ./dataset/AVRP_test/${DISTRIBUTION}/1000 \
#         --data_sizes -1 -1 -1 -1 \
#         --batch_sizes 1 1 1 1 \
#         --n_rollout 16 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
    
#     python ./test/main.py \
#         --data_dirs ./dataset/ATSPLIB \
#         --data_sizes -1 \
#         --batch_sizes 1 \
#         --n_rollout 16 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json \
#         --is_lib
# done

# MODEL_PATHS=(./output/ACVRP-128-16-6-0-3-mean)
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/AVRP_test/${DISTRIBUTION}/50 ./dataset/AVRP_test/${DISTRIBUTION}/100 ./dataset/AVRP_test/${DISTRIBUTION}/500 ./dataset/AVRP_test/${DISTRIBUTION}/1000 \
#         --data_sizes -1 -1 -1 -1 \
#         --batch_sizes 1 1 1 1 \
#         --n_rollout 16 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
# done

DISTRIBUTIONS=(uniform clustered explosion implosion)
MODEL_PATHS=(./output/SCVRP-128-16-6-50-3-mean ./output/SCVRP-128-16-6-100-3-mean)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
        python ./test/main.py \
            --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
            --data_sizes -1 -1 -1 -1 \
            --batch_sizes 1 1 1 1 \
            --n_rollout 16 \
            --output_dir ${MODEL_PATH} \
            --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
            --cfg_path ${MODEL_PATH}/configs.json
    done
    
    python ./test/main.py \
        --data_dirs ./dataset/SCVRPLIB \
        --data_sizes -1 \
        --batch_sizes 1 \
        --n_rollout 16 \
        --output_dir ${MODEL_PATH} \
        --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
        --cfg_path ${MODEL_PATH}/configs.json \
        --is_lib
done

MODEL_PATHS=(./output/STSP-128-8-6-50-3-mean ./output/STSP-128-16-6-50-3-mean ./output/STSP-128-16-6-100-3-mean)
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
        python ./test/main.py \
            --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
            --data_sizes -1 -1 -1 -1 \
            --batch_sizes 1 1 1 1 \
            --n_rollout 16 \
            --output_dir ${MODEL_PATH} \
            --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
            --cfg_path ${MODEL_PATH}/configs.json
    done
    
    python ./test/main.py \
        --data_dirs ./dataset/STSPLIB \
        --data_sizes -1 \
        --batch_sizes 1 \
        --n_rollout 16 \
        --output_dir ${MODEL_PATH} \
        --ckpt_path ${MODEL_PATH}/ckpts/steps-${STEPS}.pth \
        --cfg_path ${MODEL_PATH}/configs.json \
        --is_lib
done


# /usr/bin/shutdown