# MODEL_PATH=./output/ACVRP-128-16-6-0-mean-False
# python ./test/main.py \
#     --data_dirs ./dataset/AVRP_test/tmat/50 ./dataset/AVRP_test/tmat/100 ./dataset/AVRP_test/tmat/500 ./dataset/AVRP_test/tmat/1000 \
#     --data_sizes -1 -1 -1 -1 \
#     --batch_sizes 1 1 1 1 \
#     --n_rollout 10 \
#     --output_dir ${MODEL_PATH} \
#     --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#     --cfg_path ${MODEL_PATH}/configs.json
    
MODEL_PATH=./output/ATSP-128-16-6-0-mean-False
python ./test/main.py \
    --data_dirs ./dataset/AVRP_test/perturbation/50 ./dataset/AVRP_test/perturbation/100 ./dataset/AVRP_test/perturbation/500 ./dataset/AVRP_test/perturbation/1000 \
    --data_sizes -1 -1 -1 -1 \
    --batch_sizes 1 1 1 1 \
    --n_rollout 10 \
    --output_dir ${MODEL_PATH} \
    --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
    --cfg_path ${MODEL_PATH}/configs.json

# DISTRIBUTIONS=(uniform clustered explosion implosion)
# MODEL_PATH=./output/STSP-128-16-6-50-mean-False
# for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
#         --data_sizes -1 -1 -1 -1 \
#         --batch_sizes 1 1 1 1 \
#         --n_rollout 10 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
# done

# MODEL_PATH=./output/SCVRP-128-16-6-100-mean-False
# for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
#         --data_sizes -1 -1 -1 -1 \
#         --batch_sizes 1 1 1 1 \
#         --n_rollout 10 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
# done

# MODEL_PATH=./output/SCVRP-128-16-6-50-mean-False
# for DISTRIBUTION in "${DISTRIBUTIONS[@]}"; do
#     python ./test/main.py \
#         --data_dirs ./dataset/SVRP_test/${DISTRIBUTION}/100 ./dataset/SVRP_test/${DISTRIBUTION}/1000 ./dataset/SVRP_test/${DISTRIBUTION}/5000 ./dataset/SVRP_test/${DISTRIBUTION}/10000 \
#         --data_sizes -1 -1 -1 -1 \
#         --batch_sizes 1 1 1 1 \
#         --n_rollout 10 \
#         --output_dir ${MODEL_PATH} \
#         --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#         --cfg_path ${MODEL_PATH}/configs.json
# done

# /usr/bin/shutdown