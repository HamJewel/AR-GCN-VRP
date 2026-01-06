# MODEL_PATH=./output/ATSP-128-16-6-0-mean-False
# python ./test/main.py \
#     --data_dirs ./dataset/ATSPLIB \
#     --data_sizes -1 \
#     --batch_sizes 1 \
#     --n_rollout 50 \
#     --output_dir ${MODEL_PATH} \
#     --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#     --cfg_path ${MODEL_PATH}/configs.json \
#     --is_lib

# MODEL_PATH=./output/STSP-128-16-6-50-mean-False
# python ./test/main.py \
#     --data_dirs ./dataset/STSPLIB \
#     --data_sizes -1 \
#     --batch_sizes 1 \
#     --n_rollout 50 \
#     --output_dir ${MODEL_PATH} \
#     --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#     --cfg_path ${MODEL_PATH}/configs.json \
#     --is_lib

# MODEL_PATH=./output/SCVRP-128-16-6-50-mean-False
# python ./test/main.py \
#     --data_dirs ./dataset/SCVRPLIB \
#     --data_sizes -1 \
#     --batch_sizes 1 \
#     --n_rollout 10 \
#     --output_dir ${MODEL_PATH} \
#     --ckpt_path ${MODEL_PATH}/ckpts/steps-5000.pth \
#     --cfg_path ${MODEL_PATH}/configs.json \
#     --is_lib

MODEL_PATH=./output/SCVRP-128-16-6-100-mean-False
python ./test/main.py \
    --data_dirs ./dataset/SCVRPLIB \
    --data_sizes -1 \
    --batch_sizes 1 \
    --n_rollout 10 \
    --output_dir ${MODEL_PATH} \
    --ckpt_path ${MODEL_PATH}/ckpts/steps-500.pth \
    --cfg_path ${MODEL_PATH}/configs.json \
    --is_lib

# /usr/bin/shutdown