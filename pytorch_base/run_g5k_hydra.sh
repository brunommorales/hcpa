#!/usr/bin/env bash
###############################################################################
# pytorch_base/run_g5k_hydra.sh
#
# Roda dr_hcpa_v2_2024.py (PyTorch baseline) no node G5K preparado pelo
# tools/g5k_setup_node.sh. Itera RUN_START..RUN_END (default 0..9).
#
# Funciona em qualquer cluster (hydra/chuc/chicoree); o node vem do state.env.
# Args IDENTICOS ao pytorch_base/distributed_run_arm.slurm (mesma config do
# estudo cross-GPU do GPPD) para manter comparabilidade.
###############################################################################

set -euo pipefail

APPROACH=pytorch_base
ENTRY=dr_hcpa_v2_2024.py
RESULTS_FLAG=--results
EXEC_FLAG=--exec
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

# HCPA_EPOCHS e a manopla UNICA de epocas nas 8 abordagens. Antes cada script
# lia um nome diferente (TARGET_EPOCHS / EPOCHS / HCPA_EPOCHS): passar
# HCPA_EPOCHS=5 aqui era ignorado e o run ia para 200 epocas em silencio.
TARGET_EPOCHS="${HCPA_EPOCHS:-${TARGET_EPOCHS:-200}}"

TRAIN_STATIC_ARGS=(
  --model inception_v3
  --normalize preprocess
  --batch_size 96
  --epochs "${TARGET_EPOCHS}"
  --optimizer adamw
  --lrate 5e-4
  --freeze_epochs 0
  --fine_tune_lr_factor 0.1
  --fine_tune_lr 5e-4
  --warmup_epochs 5
  --min_lr 1e-6
  --label_smoothing 0.0
  --fundus_crop_ratio 1.0
  --disable_amp
  --disable_cosine
)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
