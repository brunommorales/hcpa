#!/usr/bin/env bash
###############################################################################
# pytorch_opt/run_g5k_hydra.sh
#
# Roda dr_hcpa_v2_2024.py (PyTorch) no node Hydra (Lyon) preparado pelo
# tools/g5k_setup_node.sh. Itera RUN_START..RUN_END no padrao 0..9.
#
# Pre-requisitos:
#   - tools/g5k_setup_node.sh ja submeteu o job e fez kadeploy
#   - hcpa.sif buildado no node (dentro de pytorch_opt/)
#   - data/all-tfrec presente em ${PROJECT_DIR}/data/all-tfrec
#
# Uso:
#   bash run_g5k_hydra.sh
#   RUN_START=0 RUN_END=2 bash run_g5k_hydra.sh    # so 3 runs
#   TARGET_EPOCHS=50 bash run_g5k_hydra.sh         # epocas reduzidas
###############################################################################

set -euo pipefail

APPROACH=pytorch_opt
ENTRY=dr_hcpa_v2_2024.py
RESULTS_FLAG=--results
EXEC_FLAG=--exec
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

TARGET_EPOCHS="${TARGET_EPOCHS:-200}"

TRAIN_STATIC_ARGS=(
  --model inception_v3
  --normalize preprocess
  --batch_size 96
  --epochs "${TARGET_EPOCHS}"
  --lrate 5e-4
  --freeze_epochs 0
  --fine_tune_lr_factor 0.1
  --fine_tune_lr 5e-4
  --warmup_epochs 5
  --min_lr 1e-6
  --mixup_alpha 0.0
  --cutmix_alpha 0.0
  --label_smoothing 0.0
  --focal_gamma 0.0
  --pos_weight 1.0
  --fundus_crop_ratio 1.0
  --tta_views 1
)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
