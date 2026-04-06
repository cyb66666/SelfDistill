#!/bin/bash

# RS3 网格蒸馏训练启动脚本（支持多卡）

echo "======================================"
echo "RS3 Grid Distillation Training"
echo "======================================"

# ==================== 配置参数 ====================

# 数据路径
RS3_TAR_DIR="./rs3"
RS3_VAL_DIR="./rs3_val"

# 模型配置
MODEL_NAME="ViT-B-32"
BASE_MODEL_TYPE="remoteclip"  # remoteclip | georsclip | clip | custom
REMOTECLIP_PRETRAINED_PATH="/root/checkpoint/RemoteCLIP-ViT-B-32.pt"
GEORSCLIP_PRETRAINED_PATH="/root/checkpoint/RS5M_ViT-B-32.pt"
CLIP_PRETRAINED_PATH="openai"  # open_clip 内置 tag 或本地 .pt；仅 BASE_MODEL_TYPE=clip 时使用
PRETRAINED_PATH=""  # optional explicit override when BASE_MODEL_TYPE=custom
MAX_SPLIT=7
MAX_BOXES=49
PRECISION="fp32"

# 训练配置
BATCH_SIZE=16  # 每个 GPU 的 batch size（多卡时总 batch_size = BATCH_SIZE * NUM_GPUS）
BASE_NUM_WORKERS=8  # 每个 GPU 的 worker 数（多卡时总 worker=BASE_NUM_WORKERS * NUM_GPUS）
WHOLE_IMAGE_SIZE=1024
CROP_SIZE=224
VAL_TAR_COUNT=2
EPOCHS=6
LR=2e-5
WEIGHT_DECAY=0.1
WARMUP_EPOCHS=1
WARMUP_STEPS=1000
GRAD_CLIP=1.0
LOSS_TYPE="cosine_sim"
SCHEDULER_TYPE="cosine"
COOLDOWN_STEPS=0

# distillation settings
DISTILL_ALIGN="roi_to_pooled_attn"
TEACHER_LAST_ATTN_TYPE="qq+kk+vv"
DISTILL_TYPE="frozen"   # frozen | ema | active（FarSLIP 对齐）
EMA_MOMENTUM=0.99
DISTILL_LOSS_WEIGHT=0.1
CLIP_LOSS_WEIGHT=1.0
USE_CLIP_LOSS=true
USE_DISTILL_LOSS=true
USE_RAFA=false
USE_HYCD=false
RAFA_WEIGHT=0.1
HYCD_WEIGHT=0.1
HYCD_TEMPERATURE=1.0
HYCD_ALPHA_BLENDING=0.5
RAFA_PRIOR_MU=0.0
RAFA_PRIOR_SIGMA=1.0
RAFA_SHARE_RANDOM_FEAT=true
RAFA_PRIOR_STATS_PATH=""

# 输出配置
OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
RESUME=""
SAVE_FREQ=1
LOG_FREQ=10

# 设备配置
USE_AMP=true   # 启用 AMP 混合精度训练
PRECISION="fp32"  # 模型使用 fp32 精度，AMP 会自动处理 FP16 计算

# ==================== 多卡训练配置 ====================

USE_SINGLE_GPU=true

if [ "$USE_SINGLE_GPU" = false ]; then
   NUM_GPUS=2
  echo "Using $NUM_GPUS GPUs (dual RTX 4090 configuration)"
    
    PER_GPU_BATCH_SIZE=${BATCH_SIZE}
  TOTAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))
  echo "Per-GPU batch size: ${PER_GPU_BATCH_SIZE} (total batch size: ${TOTAL_BATCH_SIZE})"
    
  # 按 GPU 数量分摊 workers，避免 CPU 过载
  NUM_WORKERS=$((BASE_NUM_WORKERS * NUM_GPUS))
  echo "Total workers: ${NUM_WORKERS} (${BASE_NUM_WORKERS} per GPU)"
fi

# ==================== 运行训练 ====================

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"

if [ "$USE_AMP" = true ]; then
    AMP_ARG=""
else
    AMP_ARG="--no-amp"
fi

if [ "${USE_RAFA}" = true ]; then
  RAFA_ENABLE_ARG="--use-rafa"
else
  RAFA_ENABLE_ARG=""
fi

if [ "${USE_HYCD}" = true ]; then
  HYCD_ENABLE_ARG="--use-hycd"
else
  HYCD_ENABLE_ARG=""
fi

if [ "${USE_CLIP_LOSS}" = true ]; then
  CLIP_ENABLE_ARG=""
else
  CLIP_ENABLE_ARG="--no-clip-loss"
fi

if [ "${USE_DISTILL_LOSS}" = true ]; then
  DISTILL_ENABLE_ARG=""
else
  DISTILL_ENABLE_ARG="--no-distill-loss"
fi

if [ "${RAFA_SHARE_RANDOM_FEAT}" = true ]; then
  RAFA_SHARE_ARG="--rafa-share-random-feat"
else
  RAFA_SHARE_ARG="--no-rafa-share-random-feat"
fi

if [ -n "${PRETRAINED_PATH}" ]; then
  PRETRAINED_ARG=(--pretrained-path "${PRETRAINED_PATH}")
else
  PRETRAINED_ARG=()
fi

if [ -n "${RAFA_PRIOR_STATS_PATH}" ]; then
  RAFA_STATS_ARG=(--rafa-prior-stats-path "${RAFA_PRIOR_STATS_PATH}")
else
  RAFA_STATS_ARG=()
fi

if [ "$USE_SINGLE_GPU" = true ]; then
  echo "Starting single-GPU training..."
  NUM_WORKERS=${BASE_NUM_WORKERS}
  echo "Workers: ${NUM_WORKERS}"
   NUM_WORKERS=${BASE_NUM_WORKERS}
  echo "Workers: ${NUM_WORKERS}"
    
  /usr/local/miniconda3/envs/py310/bin/python src/training/train_distill.py \
        --rs3-tar-dir ${RS3_TAR_DIR} \
        --rs3-val-dir ${RS3_VAL_DIR} \
        --batch-size ${BATCH_SIZE} \
        --whole-image-size ${WHOLE_IMAGE_SIZE} \
        --crop-size ${CROP_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --val-tar-count ${VAL_TAR_COUNT} \
        --max-split ${MAX_SPLIT} \
        --max-boxes ${MAX_BOXES} \
        --model-name ${MODEL_NAME} \
        --base-model-type ${BASE_MODEL_TYPE} \
        --remoteclip-pretrained-path ${REMOTECLIP_PRETRAINED_PATH} \
        --georsclip-pretrained-path ${GEORSCLIP_PRETRAINED_PATH} \
        --clip-pretrained-path ${CLIP_PRETRAINED_PATH} \
        ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
        --precision ${PRECISION} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --warmup-epochs ${WARMUP_EPOCHS} \
        --warmup-steps ${WARMUP_STEPS} \
        --scheduler-type ${SCHEDULER_TYPE} \
        --cooldown-steps ${COOLDOWN_STEPS} \
        --grad-clip ${GRAD_CLIP} \
        --loss-type ${LOSS_TYPE} \
        --distill-align ${DISTILL_ALIGN} \
        --teacher-last-attn-type ${TEACHER_LAST_ATTN_TYPE} \
        --distill-type ${DISTILL_TYPE} \
        --ema-momentum ${EMA_MOMENTUM} \
        --distill-loss-weight ${DISTILL_LOSS_WEIGHT} \
        --clip-loss-weight ${CLIP_LOSS_WEIGHT} \
        ${CLIP_ENABLE_ARG:+${CLIP_ENABLE_ARG}} \
        ${DISTILL_ENABLE_ARG:+${DISTILL_ENABLE_ARG}} \
        ${RAFA_ENABLE_ARG:+${RAFA_ENABLE_ARG}} \
        ${HYCD_ENABLE_ARG:+${HYCD_ENABLE_ARG}} \
        --rafa-weight ${RAFA_WEIGHT} \
        --hycd-weight ${HYCD_WEIGHT} \
        --hycd-temperature ${HYCD_TEMPERATURE} \
        --hycd-alpha-blending ${HYCD_ALPHA_BLENDING} \
        --rafa-prior-mu ${RAFA_PRIOR_MU} \
        --rafa-prior-sigma ${RAFA_PRIOR_SIGMA} \
        ${RAFA_STATS_ARG[@]+"${RAFA_STATS_ARG[@]}"} \
        ${RAFA_SHARE_ARG:+${RAFA_SHARE_ARG}} \
        --output-dir ${OUTPUT_DIR} \
        --log-dir ${LOG_DIR} \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --save-freq ${SAVE_FREQ} \
        --log-freq ${LOG_FREQ} \
        ${AMP_ARG:+${AMP_ARG}} \
        ${RESUME:+--resume $RESUME}
else
  echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    
  /usr/local/miniconda3/envs/py310/bin/torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29600 \
 src/training/train_distill.py \
            --rs3-tar-dir ${RS3_TAR_DIR} \
            --rs3-val-dir ${RS3_VAL_DIR} \
            --batch-size ${PER_GPU_BATCH_SIZE} \
            --whole-image-size ${WHOLE_IMAGE_SIZE} \
            --crop-size ${CROP_SIZE} \
            --num-workers ${NUM_WORKERS} \
            --val-tar-count ${VAL_TAR_COUNT} \
            --max-split ${MAX_SPLIT} \
            --max-boxes ${MAX_BOXES} \
            --model-name ${MODEL_NAME} \
            --base-model-type ${BASE_MODEL_TYPE} \
            --remoteclip-pretrained-path ${REMOTECLIP_PRETRAINED_PATH} \
            --georsclip-pretrained-path ${GEORSCLIP_PRETRAINED_PATH} \
            --clip-pretrained-path ${CLIP_PRETRAINED_PATH} \
            ${PRETRAINED_ARG[@]+"${PRETRAINED_ARG[@]}"} \
            --precision ${PRECISION} \
            --epochs ${EPOCHS} \
            --lr ${LR} \
            --weight-decay ${WEIGHT_DECAY} \
            --warmup-epochs ${WARMUP_EPOCHS} \
            --warmup-steps ${WARMUP_STEPS} \
            --scheduler-type ${SCHEDULER_TYPE} \
            --cooldown-steps ${COOLDOWN_STEPS} \
            --grad-clip ${GRAD_CLIP} \
            --loss-type ${LOSS_TYPE} \
            --distill-align ${DISTILL_ALIGN} \
            --teacher-last-attn-type ${TEACHER_LAST_ATTN_TYPE} \
            --distill-type ${DISTILL_TYPE} \
            --ema-momentum ${EMA_MOMENTUM} \
            --distill-loss-weight ${DISTILL_LOSS_WEIGHT} \
            --clip-loss-weight ${CLIP_LOSS_WEIGHT} \
            ${CLIP_ENABLE_ARG:+${CLIP_ENABLE_ARG}} \
            ${DISTILL_ENABLE_ARG:+${DISTILL_ENABLE_ARG}} \
            ${RAFA_ENABLE_ARG:+${RAFA_ENABLE_ARG}} \
            ${HYCD_ENABLE_ARG:+${HYCD_ENABLE_ARG}} \
            --rafa-weight ${RAFA_WEIGHT} \
            --hycd-weight ${HYCD_WEIGHT} \
            --hycd-temperature ${HYCD_TEMPERATURE} \
            --hycd-alpha-blending ${HYCD_ALPHA_BLENDING} \
            --rafa-prior-mu ${RAFA_PRIOR_MU} \
            --rafa-prior-sigma ${RAFA_PRIOR_SIGMA} \
            ${RAFA_STATS_ARG[@]+"${RAFA_STATS_ARG[@]}"} \
            ${RAFA_SHARE_ARG:+${RAFA_SHARE_ARG}} \
            --output-dir ${OUTPUT_DIR} \
            --log-dir ${LOG_DIR} \
            --tensorboard-dir ${TENSORBOARD_DIR} \
            --save-freq ${SAVE_FREQ} \
            --log-freq ${LOG_FREQ} \
            --dist-backend 'nccl' \
            --dist-url 'env://' \
            --world-size ${NUM_GPUS} \
            ${AMP_ARG:+${AMP_ARG}} \
            ${RESUME:+--resume $RESUME}
fi

echo ""
echo "Training finished!"
echo "Check outputs in: ${OUTPUT_DIR}"
echo "TensorBoard: tensorboard --logdir ${TENSORBOARD_DIR}"
