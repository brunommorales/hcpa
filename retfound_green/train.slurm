#!/usr/bin/env bash
#SBATCH --job-name=retfound_green
#SBATCH --chdir=/home/users/bmmorales/projects/hcpa/retfound_green
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9%1
#SBATCH --time=09:00:00
#SBATCH --partition=grace
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euxo pipefail

MODULE_NAME="retfound_green"
DEFAULT_PROJ_DIR="/home/users/bmmorales/projects/hcpa"
DEFAULT_TFREC_DIR="${DEFAULT_PROJ_DIR}/data/all-tfrec"
DEFAULT_CHECKPOINT_NAME="retfoundgreen_statedict.pth"
DEFAULT_BACKBONE_CHECKPOINT="${DEFAULT_PROJ_DIR}/${MODULE_NAME}/weights/${DEFAULT_CHECKPOINT_NAME}"

resolve_project_root() {
  local candidate="${PROJ_DIR:-${HCPA_PROJ_DIR:-}}"
  local submit_dir="${SLURM_SUBMIT_DIR:-}"

  if [[ -n "${candidate}" ]]; then
    :
  elif [[ -d "${DEFAULT_PROJ_DIR}/${MODULE_NAME}" ]]; then
    candidate="${DEFAULT_PROJ_DIR}"
  elif [[ -n "${submit_dir}" && -d "${submit_dir}/${MODULE_NAME}" ]]; then
    candidate="${submit_dir}"
  elif [[ -n "${submit_dir}" && "$(basename "${submit_dir}")" == "${MODULE_NAME}" ]]; then
    candidate="$(cd "${submit_dir}/.." && pwd)"
  else
    candidate="${DEFAULT_PROJ_DIR}"
  fi

  cd "${candidate}" && pwd
}

resolve_path() {
  local path_value="$1"
  if [[ -e "${path_value}" ]]; then
    cd "$(dirname "${path_value}")" && printf '%s/%s' "$(pwd)" "$(basename "${path_value}")"
  else
    printf '%s' "${path_value}"
  fi
}

sanitize_tag() {
  local value="${1:-}"
  value="$(printf '%s' "${value}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${value}" || "${value}" == "(null)" || "${value}" == "null" ]]; then
    printf '%s' "unknown"
    return
  fi
  value="$(printf '%s' "${value}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  value="$(printf '%s' "${value}" | sed -E 's/-+/-/g')"
  printf '%s' "${value}"
}

print_host_gpu_summary() {
  echo "Host: $(hostname)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    echo "Warning: nvidia-smi is not available on host."
  fi
}

host_cuda_sanity_check() {
  python3 - <<'PY'
import sys

try:
    import torch
    import timm
except Exception as exc:
    sys.exit(f"Host Python dependencies are incomplete: {exc}. Use USE_DOCKER=1.")

print(f"Host torch version: {torch.__version__}")
print(f"Host torch CUDA: {torch.version.cuda}")
print(f"Host CUDA available: {torch.cuda.is_available()}")
print(f"Host timm version: {getattr(timm, '__version__', 'unknown')}")

if not torch.cuda.is_available():
    sys.exit("Host PyTorch has no CUDA. Use USE_DOCKER=1 or install CUDA PyTorch.")
PY
}

ensure_docker_image() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Error: docker/podman is not available on this node. Use USE_DOCKER=0 only if host Python has CUDA+timm." >&2
    exit 1
  fi

  if docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Using Docker image: ${IMAGE_TAG}"
    return
  fi

  echo "Docker image ${IMAGE_TAG} not found; building from ${MODULE_DIR}"
  docker build --build-arg UID="${HOST_UID}" --build-arg GID="${HOST_GID}" -t "${IMAGE_TAG}" "${MODULE_DIR}"
}

run_launch_cmd() {
  local launch_cmd_str
  printf -v launch_cmd_str '%q ' "${LAUNCH_CMD[@]}"

  if [[ "${USE_DOCKER}" == "1" ]]; then
    ensure_docker_image
    mkdir -p "${PROJ_DIR}/.cache" "${PROJ_DIR}/.cache/hf" "${PROJ_DIR}/.cache/torch" "${PROJ_DIR}/tmp"

    DOCKER_USERNS_ARGS=()
    if docker --version 2>/dev/null | grep -qi podman; then
      DOCKER_USERNS_ARGS=(--userns=keep-id)
    fi

    docker run --rm --name "${MODULE_NAME}_${JOB_ID_TAG}_r${RUN_ID}" \
      "${DOCKER_USERNS_ARGS[@]}" \
      --gpus all \
      --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 \
      --network host \
      --user "${HOST_UID}:${HOST_GID}" \
      -v "${PROJ_DIR}:${CONTAINER_WORKDIR}" \
      -v "${TFREC_DIR}:${CONTAINER_TFREC_DIR}:ro" \
      -w "${CONTAINER_MODULE_DIR}" \
      -e HOME="${CONTAINER_WORKDIR}" \
      -e PYTHONPATH="${CONTAINER_WORKDIR}" \
      -e KERAS_BACKEND="torch" \
      -e PYTHONUNBUFFERED="1" \
      -e XDG_CACHE_HOME="${CONTAINER_WORKDIR}/.cache" \
      -e HF_HOME="${CONTAINER_WORKDIR}/.cache/hf" \
      -e TRANSFORMERS_CACHE="${CONTAINER_WORKDIR}/.cache/hf" \
      -e TORCH_HOME="${CONTAINER_WORKDIR}/.cache/torch" \
      -e TMPDIR="${CONTAINER_WORKDIR}/tmp" \
      -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
      -e PROFILE_TOOL="${PROFILE_TOOL}" \
      -e HCPA_ENABLE_TORCH_PROFILER="${HCPA_ENABLE_TORCH_PROFILER}" \
      -e HCPA_EMIT_NVTX="${HCPA_EMIT_NVTX}" \
      -e HCPA_PROFILE_RANK0_ONLY="${HCPA_PROFILE_RANK0_ONLY}" \
      -e HCPA_PROFILE_STAGES="${HCPA_PROFILE_STAGES}" \
      -e HCPA_TORCH_PROFILER_WAIT="${HCPA_TORCH_PROFILER_WAIT}" \
      -e HCPA_TORCH_PROFILER_WARMUP="${HCPA_TORCH_PROFILER_WARMUP}" \
      -e HCPA_TORCH_PROFILER_ACTIVE="${HCPA_TORCH_PROFILER_ACTIVE}" \
      -e HCPA_TORCH_PROFILER_REPEAT="${HCPA_TORCH_PROFILER_REPEAT}" \
      -e HCPA_TORCH_PROFILER_MAX_STEPS="${HCPA_TORCH_PROFILER_MAX_STEPS}" \
      -e HCPA_TORCH_PROFILER_RECORD_SHAPES="${HCPA_TORCH_PROFILER_RECORD_SHAPES}" \
      -e HCPA_TORCH_PROFILER_PROFILE_MEMORY="${HCPA_TORCH_PROFILER_PROFILE_MEMORY}" \
      -e HCPA_TORCH_PROFILER_WITH_STACK="${HCPA_TORCH_PROFILER_WITH_STACK}" \
      "${IMAGE_TAG}" \
      bash -lc "
        set -euo pipefail
        mkdir -p \"${CONTAINER_RUN_RESULTS_DIR}\" \"${CONTAINER_WORKDIR}/tmp\"
        cd \"${CONTAINER_MODULE_DIR}\"
        python - <<'PY'
import sys
import torch
import timm

print(f'Container torch version: {torch.__version__}')
print(f'Container torch CUDA: {torch.version.cuda}')
print(f'Container CUDA available: {torch.cuda.is_available()}')
print(f'Container CUDA device count: {torch.cuda.device_count()}')
print(f'Container timm version: {getattr(timm, \"__version__\", \"unknown\")}')

if torch.cuda.is_available():
    print(f'Container GPU 0: {torch.cuda.get_device_name(0)}')
else:
    sys.exit('CUDA is not available inside the container.')
PY
        ${launch_cmd_str}
      "
  else
    host_cuda_sanity_check
    "${LAUNCH_CMD[@]}"
  fi
}

PROJ_DIR="$(resolve_project_root)"
MODULE_DIR="${PROJ_DIR}/${MODULE_NAME}"
TFREC_DIR="$(resolve_path "${TFREC_DIR:-${DEFAULT_TFREC_DIR}}")"
BACKBONE_CHECKPOINT="$(resolve_path "${BACKBONE_CHECKPOINT:-${DEFAULT_BACKBONE_CHECKPOINT}}")"

if [[ ! -d "${MODULE_DIR}" ]]; then
  echo "Error: module directory does not exist: ${MODULE_DIR}" >&2
  exit 1
fi
if [[ ! -d "${TFREC_DIR}" ]]; then
  echo "Error: TFREC_DIR does not exist: ${TFREC_DIR}" >&2
  exit 1
fi
if [[ ! -f "${BACKBONE_CHECKPOINT}" ]]; then
  echo "Error: RETFound-Green checkpoint not found: ${BACKBONE_CHECKPOINT}" >&2
  echo "Expected file: ${DEFAULT_BACKBONE_CHECKPOINT}" >&2
  exit 1
fi

mkdir -p "${MODULE_DIR}/logs" "${MODULE_DIR}/results"
cd "${MODULE_DIR}"

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_ID=0; fi
JOB_ID_TAG="${ARRAY_JOB_ID}"

RUN_START="${RUN_START:-${ARRAY_TASK_ID}}"
RUN_END="${RUN_END:-${ARRAY_TASK_ID}}"
if ! [[ "${RUN_START}" =~ ^[0-9]+$ && "${RUN_END}" =~ ^[0-9]+$ ]]; then
  echo "Error: RUN_START and RUN_END must be integers." >&2
  exit 1
fi
if (( RUN_START < 0 || RUN_END > 9 || RUN_START > RUN_END )); then
  echo "Error: invalid run range (${RUN_START}-${RUN_END}). Expected 0..9." >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-200}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
IMG_SIZE="${IMG_SIZE:-392}"
BACKBONE_MODEL="${BACKBONE_MODEL:-vit_small_patch14_reg4_dinov2}"
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
SEED="${SEED:-42}"
DATASET="${DATASET:-all}"

USE_DOCKER="${USE_DOCKER:-1}"
IMAGE_TAG="${IMAGE_TAG:-${MODULE_NAME}:latest}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
CONTAINER_MODULE_DIR="${CONTAINER_WORKDIR}/${MODULE_NAME}"
CONTAINER_TFREC_DIR="${CONTAINER_WORKDIR}/data/all-tfrec"
PROFILE_TOOL="${PROFILE_TOOL:-none}"
HCPA_ENABLE_TORCH_PROFILER="${HCPA_ENABLE_TORCH_PROFILER:-0}"
HCPA_EMIT_NVTX="${HCPA_EMIT_NVTX:-0}"
HCPA_PROFILE_RANK0_ONLY="${HCPA_PROFILE_RANK0_ONLY:-1}"
HCPA_PROFILE_STAGES="${HCPA_PROFILE_STAGES:-train,val,test,eval}"
HCPA_TORCH_PROFILER_WAIT="${HCPA_TORCH_PROFILER_WAIT:-1}"
HCPA_TORCH_PROFILER_WARMUP="${HCPA_TORCH_PROFILER_WARMUP:-1}"
HCPA_TORCH_PROFILER_ACTIVE="${HCPA_TORCH_PROFILER_ACTIVE:-3}"
HCPA_TORCH_PROFILER_REPEAT="${HCPA_TORCH_PROFILER_REPEAT:-1}"
HCPA_TORCH_PROFILER_MAX_STEPS="${HCPA_TORCH_PROFILER_MAX_STEPS:-0}"
HCPA_TORCH_PROFILER_RECORD_SHAPES="${HCPA_TORCH_PROFILER_RECORD_SHAPES:-1}"
HCPA_TORCH_PROFILER_PROFILE_MEMORY="${HCPA_TORCH_PROFILER_PROFILE_MEMORY:-1}"
HCPA_TORCH_PROFILER_WITH_STACK="${HCPA_TORCH_PROFILER_WITH_STACK:-0}"

GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)}"
GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW:-gpu}")"
if [[ -z "${GPU_TAG}" || "${GPU_TAG}" == "unknown" ]]; then
  GPU_TAG="gpu"
fi
GPU_DESC="${SLURM_GPUS_PER_TASK:-1}x${GPU_TAG}"
NODE_NAME_RAW="${SLURM_NODELIST:-$(hostname)}"
NODEGROUP_RAW="${NODE_NAME_RAW%%,*}"
NODEGROUP_RAW="${NODEGROUP_RAW%%[*}"
NODEGROUP_RAW="$(printf '%s' "${NODEGROUP_RAW}" | sed -E 's/[0-9]+$//')"
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-nodes}")"
BATCH_TAG="bs$(sanitize_tag "${BATCH_SIZE}")"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/result${JOB_ID_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}}"
HOST_RESULTS_ROOT="${HOST_RESULTS_ROOT:-${MODULE_DIR}/${RESULTS_SUBDIR}}"

if [[ "${HOST_RESULTS_ROOT}" != "${MODULE_DIR}/"* ]]; then
  echo "Error: HOST_RESULTS_ROOT must stay inside ${MODULE_DIR}" >&2
  exit 1
fi

mkdir -p "${HOST_RESULTS_ROOT}"

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "array_task_id=${ARRAY_TASK_ID}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "module=${MODULE_NAME}"
  echo "project_root=${PROJ_DIR}"
  echo "module_dir=${MODULE_DIR}"
  echo "tfrec_dir=${TFREC_DIR}"
  echo "backbone_checkpoint=${BACKBONE_CHECKPOINT}"
  echo "backbone_model=${BACKBONE_MODEL}"
  echo "dataset=${DATASET}"
  echo "nodelist=${SLURM_NODELIST:-}"
  echo "nodegroup=${NODEGROUP_TAG}"
  echo "gpu=${GPU_DESC}"
  echo "results_root=${HOST_RESULTS_ROOT}"
  echo "results_subdir=${RESULTS_SUBDIR}"
  echo "run_start=${RUN_START}"
  echo "run_end=${RUN_END}"
  echo "batch_size=${BATCH_SIZE}"
  echo "epochs=${EPOCHS}"
  echo "learning_rate=${LEARNING_RATE}"
  echo "img_size=${IMG_SIZE}"
  echo "head_dropout=${HEAD_DROPOUT}"
  echo "num_workers=${NUM_WORKERS}"
  echo "optimizer=${OPTIMIZER}"
  echo "weight_decay=${WEIGHT_DECAY}"
  echo "warmup_epochs=${WARMUP_EPOCHS}"
  echo "label_smoothing=${LABEL_SMOOTHING}"
  echo "pos_weight=${POS_WEIGHT}"
  echo "auto_pos_weight=${AUTO_POS_WEIGHT}"
  echo "focal_gamma=${FOCAL_GAMMA}"
  echo "clip_grad_norm=${CLIP_GRAD_NORM}"
  echo "enable_amp=${ENABLE_AMP}"
  echo "enable_ema=${ENABLE_EMA}"
  echo "enable_dali=${ENABLE_DALI}"
  echo "enable_cosine=${ENABLE_COSINE}"
  echo "enable_performance_profiler=${ENABLE_PERFORMANCE_PROFILER}"
  echo "freeze_backbone=${FREEZE_BACKBONE}"
  echo "use_docker=${USE_DOCKER}"
  echo "image_tag=${IMAGE_TAG}"
  echo "profile_tool=${PROFILE_TOOL}"
  echo "hcpa_enable_torch_profiler=${HCPA_ENABLE_TORCH_PROFILER}"
  echo "hcpa_emit_nvtx=${HCPA_EMIT_NVTX}"
  echo "storage_mode=home_direct"
} > "${HOST_RESULTS_ROOT}/env_manifest.txt"

echo "=========================================="
echo "RETFound-Green Training"
echo "=========================================="
echo "Storage Mode: home_direct"
echo "Submit hint: sbatch train.slurm"
echo "Job ID Tag: ${JOB_ID_TAG}"
echo "Run Range: ${RUN_START}-${RUN_END}"
echo "Project Root: ${PROJ_DIR}"
echo "Module Dir: ${MODULE_DIR}"
echo "TFRecords: ${TFREC_DIR}"
echo "Checkpoint: ${BACKBONE_CHECKPOINT}"
echo "Results Root: ${HOST_RESULTS_ROOT}"
echo "Backbone: ${BACKBONE_MODEL}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Image Size: ${IMG_SIZE}"
echo "Freeze Backbone: ${FREEZE_BACKBONE}"
echo "Use Docker: ${USE_DOCKER}"
echo "Image Tag: ${IMAGE_TAG}"
echo "=========================================="
print_host_gpu_summary

AMP_ARGS=()
if [[ "${ENABLE_AMP}" == "1" ]]; then
  AMP_ARGS+=(--enable_amp)
else
  AMP_ARGS+=(--disable_amp)
fi

EMA_ARGS=()
if [[ "${ENABLE_EMA}" == "1" ]]; then
  EMA_ARGS+=(--enable_ema)
else
  EMA_ARGS+=(--disable_ema)
fi

DALI_ARGS=()
if [[ "${ENABLE_DALI}" == "1" ]]; then
  DALI_ARGS+=(--enable_dali)
else
  DALI_ARGS+=(--disable_dali)
fi

COSINE_ARGS=()
if [[ "${ENABLE_COSINE}" == "1" ]]; then
  COSINE_ARGS+=(--enable_cosine)
else
  COSINE_ARGS+=(--disable_cosine)
fi

PERF_ARGS=()
if [[ "${ENABLE_PERFORMANCE_PROFILER}" == "1" ]]; then
  PERF_ARGS+=(--enable_performance_profiler)
else
  PERF_ARGS+=(--disable_performance_profiler)
fi

AUTO_POS_ARGS=()
if [[ "${AUTO_POS_WEIGHT}" == "1" ]]; then
  AUTO_POS_ARGS+=(--auto_pos_weight)
else
  AUTO_POS_ARGS+=(--disable_auto_pos_weight)
fi

FREEZE_ARGS=()
if [[ "${FREEZE_BACKBONE}" == "1" ]]; then
  FREEZE_ARGS+=(--freeze_backbone)
else
  FREEZE_ARGS+=(--unfreeze_backbone)
fi

for RUN_ID in $(seq "${RUN_START}" "${RUN_END}"); do
  RUN_SEED=$((SEED + RUN_ID))
  RUN_RESULTS_DIR_HOST="${HOST_RESULTS_ROOT}/run_${RUN_ID}"
  CONTAINER_RUN_RESULTS_DIR="${CONTAINER_MODULE_DIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
  mkdir -p "${RUN_RESULTS_DIR_HOST}" "${RUN_RESULTS_DIR_HOST}/profiling"

  TRAIN_SCRIPT_PATH="${MODULE_DIR}/train.py"
  TRAIN_TFREC_DIR="${TFREC_DIR}"
  TRAIN_RESULTS_DIR="${RUN_RESULTS_DIR_HOST}"
  TRAIN_CHECKPOINT_PATH="${BACKBONE_CHECKPOINT}"
  if [[ "${USE_DOCKER}" == "1" ]]; then
    TRAIN_SCRIPT_PATH="${CONTAINER_MODULE_DIR}/train.py"
    TRAIN_TFREC_DIR="${CONTAINER_TFREC_DIR}"
    TRAIN_RESULTS_DIR="${CONTAINER_RUN_RESULTS_DIR}"
    if [[ "${BACKBONE_CHECKPOINT}" == "${PROJ_DIR}/"* ]]; then
      TRAIN_CHECKPOINT_PATH="${CONTAINER_WORKDIR}/${BACKBONE_CHECKPOINT#${PROJ_DIR}/}"
    else
      echo "Error: BACKBONE_CHECKPOINT must be inside ${PROJ_DIR} when USE_DOCKER=1." >&2
      exit 1
    fi
  fi

  {
    echo "run_id=${RUN_ID}"
    echo "seed=${RUN_SEED}"
    echo "exec_id=${RUN_ID}"
    echo "results_dir=${RUN_RESULTS_DIR_HOST}"
    echo "train_script_path=${TRAIN_SCRIPT_PATH}"
    echo "train_tfrec_dir=${TRAIN_TFREC_DIR}"
    echo "train_results_dir=${TRAIN_RESULTS_DIR}"
    echo "train_checkpoint_path=${TRAIN_CHECKPOINT_PATH}"
    echo "storage_mode=home_direct"
    echo "use_docker=${USE_DOCKER}"
  } > "${RUN_RESULTS_DIR_HOST}/run_manifest.txt"

  echo "[RUN] ${MODULE_NAME} job=${JOB_ID_TAG} run=${RUN_ID}"
  LAUNCH_CMD=(
    python3
    -u
    "${TRAIN_SCRIPT_PATH}"
    --tfrec_dir "${TRAIN_TFREC_DIR}"
    --results_dir "${TRAIN_RESULTS_DIR}"
    --exec_id "${RUN_ID}"
    --dataset "${DATASET}"
    --backbone_model "${BACKBONE_MODEL}"
    --backbone_checkpoint "${TRAIN_CHECKPOINT_PATH}"
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
    "${AMP_ARGS[@]}"
    "${EMA_ARGS[@]}"
    "${DALI_ARGS[@]}"
    "${COSINE_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${AUTO_POS_ARGS[@]}"
    "${FREEZE_ARGS[@]}"
  )

  run_launch_cmd
done

echo "RETFound-Green training complete. Results are in ${HOST_RESULTS_ROOT}"
