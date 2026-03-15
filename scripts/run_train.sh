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
PRETRAINED_PATH="pretrained/RS5M_ViT-B-32.pt"
PRECISION="fp32"

# 训练配置
BATCH_SIZE=16  # 每个 GPU 的 batch size（多卡时总 batch_size = BATCH_SIZE * NUM_GPUS）
BASE_NUM_WORKERS=8  # 每个 GPU 的 worker 数（多卡时总 worker=BASE_NUM_WORKERS * NUM_GPUS）
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

# 输出配置
OUTPUT_DIR="./checkpoints"
LOG_DIR="./logs"
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

if [ "$USE_AMP" = true ]; then
    AMP_ARG=""
else
    AMP_ARG="--no-amp"
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
        --num-workers ${NUM_WORKERS} \
        --val-tar-count ${VAL_TAR_COUNT} \
        --model-name ${MODEL_NAME} \
        --pretrained-path ${PRETRAINED_PATH} \
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
        --output-dir ${OUTPUT_DIR} \
        --log-dir ${LOG_DIR} \
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
            --num-workers ${NUM_WORKERS} \
            --val-tar-count ${VAL_TAR_COUNT} \
            --model-name ${MODEL_NAME} \
            --pretrained-path ${PRETRAINED_PATH} \
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
            --output-dir ${OUTPUT_DIR} \
            --log-dir ${LOG_DIR} \
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
