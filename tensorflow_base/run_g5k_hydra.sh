#!/usr/bin/env bash
###############################################################################
# tensorflow_base/run_g5k_hydra.sh
#
# Roda dr_hcpa_v2_2024.py (TensorFlow/Keras baseline) no node G5K preparado
# pelo tools/g5k_setup_node.sh. Itera RUN_START..RUN_END (default 0..9).
#
# Funciona em qualquer cluster (hydra/chuc/chicoree); o node vem do state.env.
# Args IDENTICOS ao tensorflow_base/distributed_run_arm.slurm. SEM --use_dali e
# SEM --jit_compile (isso e do tensorflow_opt); este e o baseline puro.
# OBS: o entry do tensorflow_base NAO aceita --seed (SEED_FLAG vazio).
###############################################################################

set -euo pipefail

APPROACH=tensorflow_base
ENTRY=dr_hcpa_v2_2024.py
RESULTS_FLAG=--results
EXEC_FLAG=--exec
SEED_FLAG=""               # tensorflow_base nao tem --seed
TRAIN_ACCEPTS_DATASET=1

TARGET_EPOCHS="${TARGET_EPOCHS:-200}"

TRAIN_STATIC_ARGS=(
  --model InceptionV3
  --normalize preprocess
  --batch_size 96
  --epochs "${TARGET_EPOCHS}"
  --optimizer adamw
  --lrate 5e-4
  --weight_decay 1e-5
  --clipnorm 1.0
  --warmup_epochs 5
  --min_lr 1e-6
  --label_smoothing 0.0
  --early_stop_patience "${EARLY_STOP_PATIENCE:-0}"
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA:-0.0001}"
  --exact-val-every-epoch
  --verbose 2
)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"

# Otimizacoes TF (espelha distributed_run_arm.slurm). Sem DALI no baseline.
export TF_GPU_ALLOCATOR="cuda_malloc_async"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export TF_ENABLE_GPU_GC="1"
export TF_NUM_INTEROP_THREADS="0"
export TF_CPP_MIN_LOG_LEVEL="2"
export DALI_LOG="ERROR"

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
