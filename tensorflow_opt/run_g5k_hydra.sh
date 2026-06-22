#!/usr/bin/env bash
###############################################################################
# tensorflow_opt/run_g5k_hydra.sh
#
# Roda dr_hcpa_v2_2024.py (TensorFlow/Keras) no node Hydra (Lyon).
# Itera RUN_START..RUN_END (default 0..9) usando o hcpa.sif preparado pelo
# tools/g5k_setup_node.sh.
###############################################################################

set -euo pipefail

APPROACH=tensorflow_opt
ENTRY=dr_hcpa_v2_2024.py
RESULTS_FLAG=--results
EXEC_FLAG=--exec
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

TARGET_EPOCHS="${TARGET_EPOCHS:-200}"

# IDENTICO ao run_g5k_arm.oar (config que produziu o h200: 3880 img/s, val_auc 0.96, 1001s/200ep).
# --jit_compile e a chave do throughput: XLA funde o train_step (sem ele a GPU fica ~57%,
# ociosa entre kernels). --cache_dir none = sem cache tf.data (igual h200; staging local cobre I/O).
TRAIN_STATIC_ARGS=(
  --batch_size 96
  --model InceptionV3
  --normalize preprocess
  --epochs "${TARGET_EPOCHS}"
  --freeze_epochs 1
  --fine_tune_at -200
  --fine_tune_lr 2e-4
  --lrate 0.003
  --mixup_alpha 0.3
  --cutmix_alpha 0.6
  --label_smoothing 0.1
  --focal_gamma 2.0
  --pos_weight 2.0
  --fundus_crop_ratio 0.9
  --warmup_epochs 5
  --min_lr 1e-6
  --tta_views 2
  --recompute_backbone
  --jit_compile
  --auc_target 0.95
  --cache_dir none
  --exact_eval_interval 5
)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"

# Otimizacoes TF para GH200 (espelha distributed_run_arm.slurm)
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
