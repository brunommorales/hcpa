#!/usr/bin/env bash
#SBATCH --job-name=hybrid_token_reduction_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-0%0
#SBATCH --time=08:00:00
#SBATCH --partition=grace
#SBATCH --nodelist=grace1
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

MODULE_NAME="hybrid_token_reduction_opt"
HCPA_DISTRIBUTED_ARCH="arm"
IMAGE="${IMAGE:-hybrid_token_reduction_opt_distributed_arm:latest}"
DATASET_NAME="${DATASET_NAME:-all-tfrec}"
TFREC_STAGE_NAME="${TFREC_STAGE_NAME:-all-tfrec}"
DOCKER_GPU_RUNTIME="${DOCKER_GPU_RUNTIME:-nvidia}"
SRUN_MAX_RETRIES="${SRUN_MAX_RETRIES:-3}"
SRUN_RETRY_SLEEP="${SRUN_RETRY_SLEEP:-30}"

BATCH_SIZE="${BATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMG_SIZE="${IMG_SIZE:-299}"
BACKBONE="${BACKBONE:-inception_v3}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"
KEEP_RATIO="${KEEP_RATIO:-0.5}"
ENABLE_AMP="${ENABLE_AMP:-1}"
AMP_DTYPE="${AMP_DTYPE:-auto}"
ENABLE_EMA="${ENABLE_EMA:-0}"
ENABLE_DALI="${ENABLE_DALI:-0}"
ENABLE_COSINE="${ENABLE_COSINE:-0}"
ENABLE_MIXUP_CUTMIX="${ENABLE_MIXUP_CUTMIX:-1}"
ENABLE_FLASH_ATTENTION="${ENABLE_FLASH_ATTENTION:-1}"
ENABLE_COMPILE="${ENABLE_COMPILE:-0}"
MIXUP_ALPHA_ON="${MIXUP_ALPHA_ON:-0.2}"
CUTMIX_ALPHA_ON="${CUTMIX_ALPHA_ON:-0.5}"
FREEZE_BACKBONE_EPOCHS="${FREEZE_BACKBONE_EPOCHS:-3}"
POS_WEIGHT="${POS_WEIGHT:-1.0}"
RUN_NAME="${RUN_NAME:-stable_fast_opt}"
SEED="${SEED:-42}"

hcpa_module_set_support() {
  TOOLS_DIR_HOST="$(hcpa_choose_existing_dir "${TOOLS_DIR_HOST:-}" "${HOST_ROOT_DIR}/tools" "${HOME_ROOT_DIR}/tools")"
  HYBRID_SIMPLE_DIR_HOST="$(hcpa_choose_existing_dir "${HYBRID_SIMPLE_DIR_HOST:-}" "${HOST_ROOT_DIR}/hybrid_simple" "${HOME_ROOT_DIR}/hybrid_simple")"
  SHARED_ROOT_DIR="$(hcpa_choose_existing_dir "${SHARED_PY_ROOT:-}" "${HOST_ROOT_DIR}" "${HOME_ROOT_DIR}")"
  PYTORCH_OPT_DIR_HOST="$(hcpa_choose_existing_dir "${PYTORCH_OPT_DIR_HOST:-}" "${HOST_ROOT_DIR}/pytorch_opt" "${HOME_ROOT_DIR}/pytorch_opt")"

  HCPA_SUPPORT_LABELS=("tools" "hybrid_simple" "hybrid_shared" "pytorch_opt")
  HCPA_SUPPORT_DIRS=("${TOOLS_DIR_HOST}" "${HYBRID_SIMPLE_DIR_HOST}" "${SHARED_ROOT_DIR}/hybrid_shared" "${PYTORCH_OPT_DIR_HOST}")
  HCPA_DOCKER_MOUNT_SPECS=(
    "${TOOLS_DIR_HOST}:${CONT_WORKDIR}/tools:ro"
    "${HYBRID_SIMPLE_DIR_HOST}:${CONT_WORKDIR}/hybrid_simple:ro"
    "${SHARED_ROOT_DIR}/hybrid_shared:${CONT_WORKDIR}/hybrid_shared:ro"
    "${PYTORCH_OPT_DIR_HOST}:${CONT_WORKDIR}/pytorch_opt:ro"
  )
}

hcpa_module_profile_stem() {
  printf '%s' "${MODULE_NAME}_${RUN_NAME}_r${RUN_ID}"
}

hcpa_module_emit_env_manifest() {
  cat <<EOF
run_name=${RUN_NAME}
batch_size=${BATCH_SIZE}
epochs=${EPOCHS}
learning_rate=${LEARNING_RATE}
img_size=${IMG_SIZE}
backbone=${BACKBONE}
num_transformer_layers=${NUM_TRANSFORMER_LAYERS}
num_heads=${NUM_HEADS}
keep_ratio=${KEEP_RATIO}
enable_amp=${ENABLE_AMP}
amp_dtype=${AMP_DTYPE}
enable_ema=${ENABLE_EMA}
enable_dali=${ENABLE_DALI}
enable_cosine=${ENABLE_COSINE}
enable_mixup_cutmix=${ENABLE_MIXUP_CUTMIX}
mixup_alpha_on=${MIXUP_ALPHA_ON}
cutmix_alpha_on=${CUTMIX_ALPHA_ON}
pos_weight=${POS_WEIGHT}
enable_flash_attention=${ENABLE_FLASH_ATTENTION}
enable_compile=${ENABLE_COMPILE}
freeze_backbone_epochs=${FREEZE_BACKBONE_EPOCHS}
EOF
}

hcpa_module_emit_run_manifest() {
  local run_mixup_alpha="0.0"
  local run_cutmix_alpha="0.0"
  if [[ "${ENABLE_MIXUP_CUTMIX}" == "1" ]]; then
    run_mixup_alpha="${MIXUP_ALPHA_ON}"
    run_cutmix_alpha="${CUTMIX_ALPHA_ON}"
  fi
  cat <<EOF
run_name=${RUN_NAME}
backbone=${BACKBONE}
num_transformer_layers=${NUM_TRANSFORMER_LAYERS}
num_heads=${NUM_HEADS}
keep_ratio=${KEEP_RATIO}
enable_amp=${ENABLE_AMP}
amp_dtype=${AMP_DTYPE}
enable_ema=${ENABLE_EMA}
enable_dali=${ENABLE_DALI}
enable_cosine=${ENABLE_COSINE}
enable_mixup_cutmix=${ENABLE_MIXUP_CUTMIX}
mixup_alpha=${run_mixup_alpha}
cutmix_alpha=${run_cutmix_alpha}
pos_weight=${POS_WEIGHT}
enable_flash_attention=${ENABLE_FLASH_ATTENTION}
enable_compile=${ENABLE_COMPILE}
EOF
}

hcpa_module_build_train_args() {
  local run_mixup_alpha="0.0"
  local run_cutmix_alpha="0.0"
  if [[ "${ENABLE_MIXUP_CUTMIX}" == "1" ]]; then
    run_mixup_alpha="${MIXUP_ALPHA_ON}"
    run_cutmix_alpha="${CUTMIX_ALPHA_ON}"
  fi

  TRAIN_ARGS=(
    --tfrec_dir "${TRAIN_TFREC_DIR}"
    --results_dir "${TRAIN_RESULTS_DIR}"
    --exec_id "${RUN_ID}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --seed "${RUN_SEED}"
    --lrate "${LEARNING_RATE}"
    --img_size "${IMG_SIZE}"
    --backbone "${BACKBONE}"
    --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
    --num_heads "${NUM_HEADS}"
    --keep_ratio "${KEEP_RATIO}"
    --mixup_alpha "${run_mixup_alpha}"
    --cutmix_alpha "${run_cutmix_alpha}"
    --freeze_backbone_epochs "${FREEZE_BACKBONE_EPOCHS}"
    --pos_weight "${POS_WEIGHT}"
    --amp_dtype "${AMP_DTYPE}"
  )

  if [[ "${ENABLE_FLASH_ATTENTION}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_flash_attention)
  else
    TRAIN_ARGS+=(--disable_flash_attention)
  fi
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
  if [[ "${ENABLE_COMPILE}" == "1" ]]; then
    TRAIN_ARGS+=(--enable_compile)
  else
    TRAIN_ARGS+=(--disable_compile)
  fi
}

PERSISTENT_ROOT_DIR="${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}"
COMMON_LAUNCHER="${PERSISTENT_ROOT_DIR}/tools/hcpa_pytorch_distributed_common.sh"
if [[ ! -f "${COMMON_LAUNCHER}" ]]; then
  echo "Erro: launcher distribuído ausente: ${COMMON_LAUNCHER}" >&2
  exit 1
fi
source "${COMMON_LAUNCHER}"
