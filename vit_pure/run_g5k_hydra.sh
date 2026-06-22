#!/usr/bin/env bash
###############################################################################
# vit_pure/run_g5k_hydra.sh
#
# Roda train.py (Pure ViT) no node Hydra (Lyon).
# Itera RUN_START..RUN_END (default 0..9) usando o hcpa.sif preparado pelo
# tools/g5k_setup_node.sh.
###############################################################################

set -euo pipefail

APPROACH=vit_pure
ENTRY=train.py
RESULTS_FLAG=--results_dir
EXEC_FLAG=--exec_id
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

TARGET_EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMG_SIZE="${IMG_SIZE:-320}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EMBED_DIM="${EMBED_DIM:-768}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-12}"
NUM_HEADS="${NUM_HEADS:-12}"
MLP_RATIO="${MLP_RATIO:-4.0}"
DROPOUT="${DROPOUT:-0.1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
USE_CLS_TOKEN="${USE_CLS_TOKEN:-1}"
ENABLE_AMP="${ENABLE_AMP:-1}"
ENABLE_EMA="${ENABLE_EMA:-0}"
ENABLE_DALI="${ENABLE_DALI:-0}"
ENABLE_COSINE="${ENABLE_COSINE:-1}"
ENABLE_FLASH_ATTENTION="${ENABLE_FLASH_ATTENTION:-1}"  # flash/SDPA ON: ~2-3x no GH200 (AMP bf16 ja ativo)

TRAIN_STATIC_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --epochs "${TARGET_EPOCHS}"
  --lrate "${LEARNING_RATE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --img_size "${IMG_SIZE}"
  --patch_size "${PATCH_SIZE}"
  --embed_dim "${EMBED_DIM}"
  --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --mlp_ratio "${MLP_RATIO}"
  --dropout "${DROPOUT}"
)
[[ "${USE_CLS_TOKEN}"          == "1" ]] && TRAIN_STATIC_ARGS+=(--use_cls_token)         || TRAIN_STATIC_ARGS+=(--no_cls_token)
[[ "${ENABLE_AMP}"             == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_amp)            || TRAIN_STATIC_ARGS+=(--disable_amp)
[[ "${ENABLE_EMA}"             == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_ema)            || TRAIN_STATIC_ARGS+=(--disable_ema)
[[ "${ENABLE_DALI}"            == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_dali)           || TRAIN_STATIC_ARGS+=(--disable_dali)
[[ "${ENABLE_COSINE}"          == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_cosine)         || TRAIN_STATIC_ARGS+=(--disable_cosine)
[[ "${ENABLE_FLASH_ATTENTION}" == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_flash_attention) || TRAIN_STATIC_ARGS+=(--disable_flash_attention)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
