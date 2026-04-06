#!/bin/bash

# SelfDistill + FarSLIP style training entry
# - keep teacher frozen
# - student trainable
# - switch distillation to roi_to_pooled by default

set -euo pipefail

echo "======================================"
echo "SelfDistill FarSLIP-style Training"
echo "======================================"

# ==================== Paths ====================
RS3_TAR_DIR="/root/data/rs3_filtered"
RS3_VAL_DIR="/root/data/rs3_filtered"
MODEL_NAME="ViT-B-32"
PRETRAINED_PATH="/root/checkpoint/RS5M_ViT-B-32.pt"

# ==================== Train hyperparams ====================
# Reference FarSLIP stage-1 (small lr + cosine + warmup), adapted to this repo/data.
EPOCHS=6
LR=1.84e-6
WEIGHT_DECAY=0.1
WARMUP_EPOCHS=1
WARMUP_STEPS=1500
GRAD_CLIP=1.0
SCHEDULER_TYPE="cosine"
COOLDOWN_STEPS=0
DISTILL_ALIGN="roi_to_pooled"
TEACHER_LAST_ATTN_TYPE="qq+kk+vv"

# loss weights
DISTILL_LOSS_WEIGHT=1.0
CLIP_LOSS_WEIGHT=0.5
USE_CLIP_LOSS=true
USE_RAFA=false
USE_HYCD=true
RAFA_WEIGHT=0.1
HYCD_WEIGHT=0.1
HYCD_TEMPERATURE=1.0
HYCD_ALPHA_BLENDING=0.5
RAFA_PRIOR_MU=0.0
RAFA_PRIOR_SIGMA=1.0
RAFA_SHARE_RANDOM_FEAT=true
RAFA_PRIOR_STATS_PATH=""   # optional .pt produced by compute_rafa_prior_stats.py

# ==================== Runtime ====================
PRECISION="fp32"
USE_AMP=true
USE_SINGLE_GPU=true
NUM_GPUS=1
BASE_NUM_WORKERS=8
BATCH_SIZE=24
WHOLE_IMAGE_SIZE=224
CROP_SIZE=224
VAL_TAR_COUNT=2
TRAIN_TAR_COUNT=30

# ==================== Outputs ====================
OUTPUT_DIR="./checkpoints/v8.4"
LOG_DIR="./logs/v8.4"
TENSORBOARD_DIR="${LOG_DIR}/tensorboard"
SAVE_FREQ=1
LOG_FREQ=50
RESUME=""

PYTHON_BIN="/usr/local/miniconda3/envs/py310/bin/python"
TORCHRUN_BIN="/usr/local/miniconda3/envs/py310/bin/torchrun"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}"

if [ "${USE_AMP}" = true ]; then
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

if [ "${RAFA_SHARE_RANDOM_FEAT}" = true ]; then
  RAFA_SHARE_ARG="--rafa-share-random-feat"
else
  RAFA_SHARE_ARG="--no-rafa-share-random-feat"
fi

if [ -n "${RAFA_PRIOR_STATS_PATH}" ]; then
  RAFA_STATS_ARG=(--rafa-prior-stats-path "${RAFA_PRIOR_STATS_PATH}")
else
  RAFA_STATS_ARG=()
fi

if [ "${USE_SINGLE_GPU}" = true ]; then
  NUM_WORKERS="${BASE_NUM_WORKERS}"
  echo "Starting single-GPU training..."
  PYTHONPATH=. "${PYTHON_BIN}" -m src.training.train_distill \
    --rs3-tar-dir "${RS3_TAR_DIR}" \
    --rs3-val-dir "${RS3_VAL_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --whole-image-size "${WHOLE_IMAGE_SIZE}" \
    --crop-size "${CROP_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --val-tar-count "${VAL_TAR_COUNT}" \
    --train-tar-count "${TRAIN_TAR_COUNT}" \
    --model-name "${MODEL_NAME}" \
    --pretrained-path "${PRETRAINED_PATH}" \
    --precision "${PRECISION}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --scheduler-type "${SCHEDULER_TYPE}" \
    --cooldown-steps "${COOLDOWN_STEPS}" \
    --grad-clip "${GRAD_CLIP}" \
    --distill-align "${DISTILL_ALIGN}" \
    --teacher-last-attn-type "${TEACHER_LAST_ATTN_TYPE}" \
    --distill-loss-weight "${DISTILL_LOSS_WEIGHT}" \
    --clip-loss-weight "${CLIP_LOSS_WEIGHT}" \
    ${CLIP_ENABLE_ARG:+${CLIP_ENABLE_ARG}} \
    ${RAFA_ENABLE_ARG:+${RAFA_ENABLE_ARG}} \
    ${HYCD_ENABLE_ARG:+${HYCD_ENABLE_ARG}} \
    --rafa-weight "${RAFA_WEIGHT}" \
    --hycd-weight "${HYCD_WEIGHT}" \
    --hycd-temperature "${HYCD_TEMPERATURE}" \
    --hycd-alpha-blending "${HYCD_ALPHA_BLENDING}" \
    --rafa-prior-mu "${RAFA_PRIOR_MU}" \
    --rafa-prior-sigma "${RAFA_PRIOR_SIGMA}" \
    ${RAFA_STATS_ARG[@]+"${RAFA_STATS_ARG[@]}"} \
    ${RAFA_SHARE_ARG:+${RAFA_SHARE_ARG}} \
    --output-dir "${OUTPUT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --save-freq "${SAVE_FREQ}" \
    --log-freq "${LOG_FREQ}" \
    ${AMP_ARG:+${AMP_ARG}} \
    ${RESUME:+--resume "$RESUME"}
else
  NUM_WORKERS=$((BASE_NUM_WORKERS * NUM_GPUS))
  echo "Starting multi-GPU training with ${NUM_GPUS} GPUs..."
  PYTHONPATH=. "${TORCHRUN_BIN}" \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29601 \
    -m src.training.train_distill \
    --rs3-tar-dir "${RS3_TAR_DIR}" \
    --rs3-val-dir "${RS3_VAL_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --whole-image-size "${WHOLE_IMAGE_SIZE}" \
    --crop-size "${CROP_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --val-tar-count "${VAL_TAR_COUNT}" \
    --train-tar-count "${TRAIN_TAR_COUNT}" \
    --model-name "${MODEL_NAME}" \
    --pretrained-path "${PRETRAINED_PATH}" \
    --precision "${PRECISION}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --warmup-epochs "${WARMUP_EPOCHS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --scheduler-type "${SCHEDULER_TYPE}" \
    --cooldown-steps "${COOLDOWN_STEPS}" \
    --grad-clip "${GRAD_CLIP}" \
    --distill-align "${DISTILL_ALIGN}" \
    --teacher-last-attn-type "${TEACHER_LAST_ATTN_TYPE}" \
    --distill-loss-weight "${DISTILL_LOSS_WEIGHT}" \
    --clip-loss-weight "${CLIP_LOSS_WEIGHT}" \
    ${CLIP_ENABLE_ARG:+${CLIP_ENABLE_ARG}} \
    ${RAFA_ENABLE_ARG:+${RAFA_ENABLE_ARG}} \
    ${HYCD_ENABLE_ARG:+${HYCD_ENABLE_ARG}} \
    --rafa-weight "${RAFA_WEIGHT}" \
    --hycd-weight "${HYCD_WEIGHT}" \
    --hycd-temperature "${HYCD_TEMPERATURE}" \
    --hycd-alpha-blending "${HYCD_ALPHA_BLENDING}" \
    --rafa-prior-mu "${RAFA_PRIOR_MU}" \
    --rafa-prior-sigma "${RAFA_PRIOR_SIGMA}" \
    ${RAFA_STATS_ARG[@]+"${RAFA_STATS_ARG[@]}"} \
    ${RAFA_SHARE_ARG:+${RAFA_SHARE_ARG}} \
    --output-dir "${OUTPUT_DIR}" \
    --log-dir "${LOG_DIR}" \
    --tensorboard-dir "${TENSORBOARD_DIR}" \
    --save-freq "${SAVE_FREQ}" \
    --log-freq "${LOG_FREQ}" \
    --dist-backend "nccl" \
    --dist-url "env://" \
    --world-size "${NUM_GPUS}" \
    ${AMP_ARG:+${AMP_ARG}} \
    ${RESUME:+--resume "$RESUME"}
fi

echo ""
echo "Training finished."
echo "Checkpoints: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "TensorBoard: tensorboard --logdir ${TENSORBOARD_DIR}"
