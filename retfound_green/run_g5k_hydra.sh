#!/usr/bin/env bash
###############################################################################
# retfound_green/run_g5k_hydra.sh
#
# Roda train.py (RETFound-Green) no node Hydra (Lyon, GH200).
# Itera RUN_START..RUN_END (default 0..9) usando o hcpa.sif preparado pelo
# tools/g5k_setup_node.sh.
###############################################################################

set -euo pipefail

APPROACH=retfound_green
ENTRY=train.py
RESULTS_FLAG=--results_dir
EXEC_FLAG=--exec_id
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

# HCPA_EPOCHS e a manopla UNICA de epocas nas 8 abordagens. Antes cada script
# lia um nome diferente (TARGET_EPOCHS / EPOCHS / HCPA_EPOCHS): passar
# HCPA_EPOCHS=5 aqui era ignorado e o run ia para 200 epocas em silencio.
TARGET_EPOCHS="${HCPA_EPOCHS:-${TARGET_EPOCHS:-${EPOCHS:-200}}}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
IMG_SIZE="${IMG_SIZE:-392}"
BACKBONE_MODEL="${BACKBONE_MODEL:-vit_small_patch14_reg4_dinov2}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-./weights/retfoundgreen_statedict.pth}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OPTIMIZER="${OPTIMIZER:-adamw}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
FOCAL_GAMMA="${FOCAL_GAMMA:-0.0}"
CLIP_GRAD_NORM="${CLIP_GRAD_NORM:-1.0}"
ENABLE_AMP="${ENABLE_AMP:-1}"
ENABLE_EMA="${ENABLE_EMA:-0}"
ENABLE_DALI="${ENABLE_DALI:-0}"
ENABLE_COSINE="${ENABLE_COSINE:-1}"
ENABLE_PERFORMANCE_PROFILER="${ENABLE_PERFORMANCE_PROFILER:-0}"
AUTO_POS_WEIGHT="${AUTO_POS_WEIGHT:-1}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
FREEZE_EPOCHS="${FREEZE_EPOCHS:-0}"

TRAIN_STATIC_ARGS=(
  --batch_size       "${BATCH_SIZE}"
  --epochs           "${TARGET_EPOCHS}"
  --lrate            "${LEARNING_RATE}"
  --img_size         "${IMG_SIZE}"
  --backbone_model   "${BACKBONE_MODEL}"
  --backbone_checkpoint "${BACKBONE_CHECKPOINT}"
  --num_workers      "${NUM_WORKERS}"
  --optimizer        "${OPTIMIZER}"
  --weight_decay     "${WEIGHT_DECAY}"
  --warmup_epochs    "${WARMUP_EPOCHS}"
  --label_smoothing  "${LABEL_SMOOTHING}"
  --pos_weight       "${POS_WEIGHT}"
  --focal_gamma      "${FOCAL_GAMMA}"
  --clip_grad_norm   "${CLIP_GRAD_NORM}"
  --head_dropout     "${HEAD_DROPOUT}"
  --freeze_epochs    "${FREEZE_EPOCHS}"
)

[[ "${ENABLE_AMP}"                  == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_amp)                  || TRAIN_STATIC_ARGS+=(--disable_amp)
[[ "${ENABLE_EMA}"                  == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_ema)                  || TRAIN_STATIC_ARGS+=(--disable_ema)
[[ "${ENABLE_DALI}"                 == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_dali)                 || TRAIN_STATIC_ARGS+=(--disable_dali)
[[ "${ENABLE_COSINE}"               == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_cosine)               || TRAIN_STATIC_ARGS+=(--disable_cosine)
[[ "${ENABLE_PERFORMANCE_PROFILER}" == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_performance_profiler) || TRAIN_STATIC_ARGS+=(--disable_performance_profiler)
[[ "${AUTO_POS_WEIGHT}"             == "1" ]] && TRAIN_STATIC_ARGS+=(--auto_pos_weight)             || TRAIN_STATIC_ARGS+=(--disable_auto_pos_weight)
[[ "${FREEZE_BACKBONE}"             == "1" ]] && TRAIN_STATIC_ARGS+=(--freeze_backbone)             || TRAIN_STATIC_ARGS+=(--unfreeze_backbone)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
