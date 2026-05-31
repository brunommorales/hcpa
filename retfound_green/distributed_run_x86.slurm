#!/usr/bin/env bash
#SBATCH --job-name=retfound_green
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9%1
#SBATCH --time=24:00:00
#SBATCH --partition=tupi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

MODULE_NAME="retfound_green"
HCPA_DISTRIBUTED_ARCH="x86"
IMAGE="${IMAGE:-retfound_green_distributed_x86:latest}"
DATASET_NAME="${DATASET_NAME:-all-tfrec}"
TFREC_STAGE_NAME="${TFREC_STAGE_NAME:-all-tfrec-${MODULE_NAME}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"
REQUIRE_SSD="${REQUIRE_SSD:-0}"
DISTRIBUTED_STAGE_TFREC="${DISTRIBUTED_STAGE_TFREC:-0}"
DISTRIBUTED_STAGE_SUPPORT="${DISTRIBUTED_STAGE_SUPPORT:-0}"
FINAL_SYNC="${FINAL_SYNC:-0}"

BATCH_SIZE="${BATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-200}"
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
DATASET="${DATASET:-all}"
SEED="${SEED:-42}"

hcpa_module_set_support() {
  TOOLS_DIR_HOST="$(hcpa_choose_existing_dir "${TOOLS_DIR_HOST:-}" "${HOST_ROOT_DIR}/tools" "${HOME_ROOT_DIR}/tools")"
  SHARED_ROOT_DIR="$(hcpa_choose_existing_dir "${SHARED_PY_ROOT:-}" "${HOST_ROOT_DIR}" "${HOME_ROOT_DIR}")"
  PYTORCH_OPT_DIR_HOST="$(hcpa_choose_existing_dir "${PYTORCH_OPT_DIR_HOST:-}" "${HOST_ROOT_DIR}/pytorch_opt" "${HOME_ROOT_DIR}/pytorch_opt")"

  HCPA_SUPPORT_LABELS=("tools" "hybrid_shared" "pytorch_opt")
  HCPA_SUPPORT_DIRS=("${TOOLS_DIR_HOST}" "${SHARED_ROOT_DIR}/hybrid_shared" "${PYTORCH_OPT_DIR_HOST}")
  HCPA_DOCKER_MOUNT_SPECS=(
    "${TOOLS_DIR_HOST}:${CONT_WORKDIR}/tools:ro"
    "${SHARED_ROOT_DIR}/hybrid_shared:${CONT_WORKDIR}/hybrid_shared:ro"
    "${PYTORCH_OPT_DIR_HOST}:${CONT_WORKDIR}/pytorch_opt:ro"
  )
}

hcpa_module_emit_env_manifest() {
  cat <<EOF
storage_mode=home_direct_distributed
dataset=${DATASET}
batch_size=${BATCH_SIZE}
epochs=${EPOCHS}
learning_rate=${LEARNING_RATE}
img_size=${IMG_SIZE}
backbone_model=${BACKBONE_MODEL}
backbone_checkpoint=${BACKBONE_CHECKPOINT}
head_dropout=${HEAD_DROPOUT}
num_workers=${NUM_WORKERS}
optimizer=${OPTIMIZER}
weight_decay=${WEIGHT_DECAY}
warmup_epochs=${WARMUP_EPOCHS}
label_smoothing=${LABEL_SMOOTHING}
pos_weight=${POS_WEIGHT}
auto_pos_weight=${AUTO_POS_WEIGHT}
focal_gamma=${FOCAL_GAMMA}
clip_grad_norm=${CLIP_GRAD_NORM}
enable_amp=${ENABLE_AMP}
enable_ema=${ENABLE_EMA}
enable_dali=${ENABLE_DALI}
enable_cosine=${ENABLE_COSINE}
enable_performance_profiler=${ENABLE_PERFORMANCE_PROFILER}
freeze_backbone=${FREEZE_BACKBONE}
require_ssd=${REQUIRE_SSD}
distributed_stage_tfrec=${DISTRIBUTED_STAGE_TFREC}
distributed_stage_support=${DISTRIBUTED_STAGE_SUPPORT}
EOF
}

hcpa_module_emit_run_manifest() {
  cat <<EOF
storage_mode=home_direct_distributed
dataset=${DATASET}
backbone_model=${BACKBONE_MODEL}
backbone_checkpoint=${BACKBONE_CHECKPOINT}
img_size=${IMG_SIZE}
head_dropout=${HEAD_DROPOUT}
enable_amp=${ENABLE_AMP}
enable_ema=${ENABLE_EMA}
enable_dali=${ENABLE_DALI}
enable_cosine=${ENABLE_COSINE}
enable_performance_profiler=${ENABLE_PERFORMANCE_PROFILER}
freeze_backbone=${FREEZE_BACKBONE}
auto_pos_weight=${AUTO_POS_WEIGHT}
EOF
}

hcpa_module_build_train_args() {
  TRAIN_ARGS=(
    --tfrec_dir "${TRAIN_TFREC_DIR}"
    --results_dir "${TRAIN_RESULTS_DIR}"
    --exec_id "${RUN_ID}"
    --dataset "${DATASET}"
    --backbone_model "${BACKBONE_MODEL}"
    --backbone_checkpoint "${BACKBONE_CHECKPOINT}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --seed "${RUN_SEED}"
    --lrate "${LEARNING_RATE}"
    --img_size "${IMG_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --optimizer "${OPTIMIZER}"
    --weight_decay "${WEIGHT_DECAY}"
    --warmup_epochs "${WARMUP_EPOCHS}"
    --label_smoothing "${LABEL_SMOOTHING}"
    --pos_weight "${POS_WEIGHT}"
    --focal_gamma "${FOCAL_GAMMA}"
    --clip_grad_norm "${CLIP_GRAD_NORM}"
    --head_dropout "${HEAD_DROPOUT}"
  )

  if [[ "${ENABLE_AMP}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_amp)
  else
    TRAIN_ARGS+=(--disable_amp)
  fi
  if [[ "${ENABLE_EMA}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_ema)
  else
    TRAIN_ARGS+=(--disable_ema)
  fi
  if [[ "${ENABLE_DALI}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_dali)
  else
    TRAIN_ARGS+=(--disable_dali)
  fi
  if [[ "${ENABLE_COSINE}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_cosine)
  else
    TRAIN_ARGS+=(--disable_cosine)
  fi
  if [[ "${ENABLE_PERFORMANCE_PROFILER}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_performance_profiler)
  else
    TRAIN_ARGS+=(--disable_performance_profiler)
  fi
  if [[ "${AUTO_POS_WEIGHT}" == "1" ]]; then
    TRAIN_ARGS+=(--auto_pos_weight)
  else
    TRAIN_ARGS+=(--disable_auto_pos_weight)
  fi
  if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
    TRAIN_ARGS+=(--freeze_backbone)
  else
    TRAIN_ARGS+=(--unfreeze_backbone)
  fi
}

PERSISTENT_ROOT_DIR="${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}"
COMMON_LAUNCHER="${PERSISTENT_ROOT_DIR}/tools/hcpa_pytorch_distributed_common.sh"
if [[ ! -f "${COMMON_LAUNCHER}" ]]; then
  echo "Erro: launcher distribuido ausente: ${COMMON_LAUNCHER}" >&2
  exit 1
fi
source "${COMMON_LAUNCHER}"
