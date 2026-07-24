#!/usr/bin/env bash
###############################################################################
# hybrid_token_reduction/run_g5k_hydra.sh
#
# Roda train.py (Hybrid Token Reduction) no node Hydra (Lyon).
# Itera RUN_START..RUN_END (default 0..9) usando o hcpa.sif preparado pelo
# tools/g5k_setup_node.sh.
###############################################################################

set -euo pipefail

APPROACH=hybrid_token_reduction
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
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMG_SIZE="${IMG_SIZE:-299}"
BACKBONE="${BACKBONE:-inception_v3}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-3}"
# APENAS TOKEN REDUCTION: esta variante isola o efeito do token reduction
# sobre o hibrido simples. Nenhuma outra otimizacao (sem AMP, sem cosine),
# para que a diferenca vs. hybrid_simple seja SO' o token reduction.
ENABLE_AMP="${ENABLE_AMP:-0}"
ENABLE_EMA="${ENABLE_EMA:-0}"
ENABLE_DALI="${ENABLE_DALI:-0}"
ENABLE_COSINE="${ENABLE_COSINE:-0}"

TRAIN_STATIC_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --epochs "${TARGET_EPOCHS}"
  --lrate "${LEARNING_RATE}"
  --img_size "${IMG_SIZE}"
  --backbone "${BACKBONE}"
  --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
  --num_heads "${NUM_HEADS}"
  --keep_ratio "${KEEP_RATIO}"
  --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}"
)
[[ "${ENABLE_AMP}"    == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_amp)    || TRAIN_STATIC_ARGS+=(--disable_amp)
[[ "${ENABLE_EMA}"    == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_ema)    || TRAIN_STATIC_ARGS+=(--disable_ema)
[[ "${ENABLE_DALI}"   == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_dali)   || TRAIN_STATIC_ARGS+=(--disable_dali)
[[ "${ENABLE_COSINE}" == "1" ]] && TRAIN_STATIC_ARGS+=(--enable_cosine) || TRAIN_STATIC_ARGS+=(--disable_cosine)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
