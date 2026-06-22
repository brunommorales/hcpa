#!/usr/bin/env bash
###############################################################################
# hybrid_token_reduction_opt/run_g5k_hydra.sh
#
# Roda train.py (Hybrid Token Reduction Opt) no node Hydra (Lyon).
# Itera RUN_START..RUN_END (default 0..9) usando o hcpa.sif preparado pelo
# tools/g5k_setup_node.sh.
###############################################################################

set -euo pipefail

APPROACH=hybrid_token_reduction_opt
ENTRY=train.py
RESULTS_FLAG=--results_dir
EXEC_FLAG=--exec_id
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

TARGET_EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMG_SIZE="${IMG_SIZE:-299}"
BACKBONE="${BACKBONE:-inception_v3}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-3}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
ENABLE_AMP="${ENABLE_AMP:-1}"
ENABLE_EMA="${ENABLE_EMA:-0}"
ENABLE_DALI="${ENABLE_DALI:-0}"
ENABLE_COSINE="${ENABLE_COSINE:-0}"
ENABLE_FLASH_ATTENTION="${ENABLE_FLASH_ATTENTION:-1}"
ENABLE_MIXUP_CUTMIX="${ENABLE_MIXUP_CUTMIX:-1}"
MIXUP_ALPHA_ON="${MIXUP_ALPHA_ON:-0.2}"
CUTMIX_ALPHA_ON="${CUTMIX_ALPHA_ON:-0.5}"

if [[ "${ENABLE_MIXUP_CUTMIX}" == "1" ]]; then
  MIX_ALPHA="${MIXUP_ALPHA_ON}"
  CUT_ALPHA="${CUTMIX_ALPHA_ON}"
else
  MIX_ALPHA="0.0"
  CUT_ALPHA="0.0"
fi

TRAIN_STATIC_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --epochs "${TARGET_EPOCHS}"
  --lrate "${LEARNING_RATE}"
  --img_size "${IMG_SIZE}"
  --backbone "${BACKBONE}"
  --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --keep_ratio "${KEEP_RATIO}"
  --mixup_alpha "${MIX_ALPHA}"
  --cutmix_alpha "${CUT_ALPHA}"
  --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}"
  --pos_weight "${POS_WEIGHT}"
  --amp_dtype "${AMP_DTYPE}"
)
[[ "${ENABLE_AMP}"             == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_amp)             || TRAIN_STATIC_ARGS+=(--disable_amp)
[[ "${ENABLE_EMA}"             == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_ema)             || TRAIN_STATIC_ARGS+=(--disable_ema)
[[ "${ENABLE_DALI}"            == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_dali)            || TRAIN_STATIC_ARGS+=(--disable_dali)
[[ "${ENABLE_COSINE}"          == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_cosine)          || TRAIN_STATIC_ARGS+=(--disable_cosine)
[[ "${ENABLE_FLASH_ATTENTION}" == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_flash_attention) || TRAIN_STATIC_ARGS+=(--disable_flash_attention)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
