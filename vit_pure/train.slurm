#!/usr/bin/env bash
#SBATCH --job-name=vit_pure
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-9%1
#SBATCH --time=09:00:00
#SBATCH --partition=grace
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euxo pipefail

# ============================================================================
# PURE VIT TRAINING SCRIPT
# ============================================================================

MODULE_NAME="vit_pure"
DEFAULT_TFREC_DIR="/home/users/bmmorales/projects/hcpa/data/all-tfrec"
PERSISTENT_ROOT_DIR="${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}"
RUNTIME_HELPER="${PERSISTENT_ROOT_DIR}/tools/hcpa_ssd_runtime.sh"
if [[ ! -f "${RUNTIME_HELPER}" ]]; then
  echo "Erro: helper de runtime ausente: ${RUNTIME_HELPER}" >&2
  exit 1
fi
source "${RUNTIME_HELPER}"

resolve_home_project_root() {
  local module_name="$1"
  local home_default="${HOME:-/home/users/bmmorales}/projects/hcpa"
  local candidate="${HOME_PROJ_DIR:-${PERSISTENT_PROJ_DIR:-}}"
  local submit_dir="${SLURM_SUBMIT_DIR:-}"

  if [[ -n "${candidate}" ]]; then
    :
  elif [[ -d "${home_default}/${module_name}" ]]; then
    candidate="${home_default}"
  elif [[ -n "${submit_dir}" ]]; then
    if [[ -d "${submit_dir}/${module_name}" && -f "${submit_dir}/${module_name}/train.py" ]]; then
      candidate="${submit_dir}"
    elif [[ "$(basename "${submit_dir}")" == "${module_name}" ]]; then
      candidate="${submit_dir}/.."
    else
      candidate="${submit_dir}"
    fi
  else
    candidate="${home_default}"
  fi

  cd "${candidate}" && pwd
}

resolve_execution_project_root() {
  local module_name="$1"
  local fallback_root="$2"
  local candidate="${EXEC_PROJ_DIR:-${PROJ_DIR:-}}"
  local require_ssd="${REQUIRE_SSD:-1}"

  if [[ -n "${candidate}" ]]; then
    :
  elif [[ -d "${SSD_BASE_DIR}" || "${require_ssd}" == "1" ]]; then
    candidate="$(hcpa_runtime_stage_module_to_ssd "${module_name}" "${fallback_root}/${module_name}")" || return 1
    candidate="$(cd "${candidate}/.." && pwd)"
  else
    candidate="${fallback_root}"
  fi

  cd "${candidate}" && pwd
}

resolve_tfrec_dir() {
  local module_dir="$1"
  local exec_root="$2"
  local home_root="$3"
  local explicit_dir="${TFREC_DIR:-}"
  local candidate
  local candidates=()
  if [[ -n "${explicit_dir}" ]]; then
    candidates+=("${explicit_dir}")
  elif [[ "${module_dir}" != "${home_root}/"* ]]; then
    candidate="$(hcpa_runtime_stage_shared_tfrec_to_ssd "${home_root}/data/all-tfrec")" || return 1
    candidates+=("${candidate}")
  else
    candidates+=("${home_root}/data/all-tfrec")
    candidates+=("${DEFAULT_TFREC_DIR}")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      cd "${candidate}" && pwd
      return 0
    fi
  done
  printf '%s' "${candidates[0]}"
}

resolve_dir_path() {
  local dir_path="$1"
  if [[ -d "${dir_path}" ]]; then
    cd "${dir_path}" && pwd
  else
    printf '%s' "${dir_path}"
  fi
}

sanitize_tag() {
  local s="${1:-}"
  s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${s}" || "${s}" == "(null)" || "${s}" == "null" ]]; then
    printf '%s' 'unknown'
    return
  fi
  s="$(printf '%s' "${s}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  s="$(printf '%s' "${s}" | sed -E 's/-+/-/g')"
  printf '%s' "${s}"
}

parse_int_or_default() {
  local raw="${1:-}"
  local default_value="${2:-1}"
  if [[ "${raw}" =~ ^[0-9]+$ ]]; then
    printf '%s' "${raw}"
  elif [[ "${raw}" =~ ([0-9]+) ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
  else
    printf '%s' "${default_value}"
  fi
}

print_host_gpu_summary() {
  echo "Host: $(hostname)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    echo "Aviso: nvidia-smi não está disponível no host."
  fi
}

ensure_docker_image() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "Erro: docker não está disponível neste nó."
    exit 1
  fi

  if docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "Using Docker image: ${IMAGE_TAG}"
    return
  fi

  echo "Docker image ${IMAGE_TAG} não encontrada; construindo a partir de ${MODULE_DIR}"
  docker build --build-arg UID="${HOST_UID}" --build-arg GID="${HOST_GID}" -t "${IMAGE_TAG}" "${MODULE_DIR}"
}

host_cuda_sanity_check() {
  python3 - <<'PY'
import sys
import torch

print(f"Host torch version: {torch.__version__}")
print(f"Host torch CUDA: {torch.version.cuda}")
print(f"Host CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    sys.exit("Torch do host está sem CUDA. Use USE_DOCKER=1 ou instale um build CUDA do PyTorch.")
PY
}

maybe_run_final_sync() {
  if [[ "${FINAL_SYNC}" == "1" ]]; then
    echo "Executando sync final (FINAL_SYNC=1)."
    sync
  else
    echo "Pulando sync final (FINAL_SYNC=${FINAL_SYNC}); usando apenas rsync para persistência."
  fi
}

sync_results_to_home() {
  if [[ -z "${HOST_RESULTS_DIR:-}" || -z "${HOME_RESULTS_DIR:-}" ]]; then
    return 0
  fi

  if [[ "${HOST_RESULTS_DIR}" == "${HOME_RESULTS_DIR}" ]]; then
    echo "Resultados já estão em armazenamento persistente."
    maybe_run_final_sync
    return 0
  fi

  if ! command -v rsync >/dev/null 2>&1; then
    echo "Erro: rsync não está disponível no nó."
    return 1
  fi

  mkdir -p "${HOME_RESULTS_DIR}"
  echo "Sincronizando artefatos finais de ${HOST_RESULTS_DIR} para ${HOME_RESULTS_DIR} (*.csv, *.pdf)"
  rsync -a --prune-empty-dirs \
    --include '*/' \
    --include '*.csv' \
    --include '*.pdf' \
    --exclude '*' \
    "${HOST_RESULTS_DIR}/" "${HOME_RESULTS_DIR}/"
  if [[ "${SYNC_LOGS_TO_HOME}" == "1" && -d "${HOST_LOGS_DIR}" ]]; then
    mkdir -p "${HOME_LOGS_DIR}"
    echo "Sincronizando logs de ${HOST_LOGS_DIR} para ${HOME_LOGS_DIR}"
    rsync -a "${HOST_LOGS_DIR}/" "${HOME_LOGS_DIR}/"
  fi
  maybe_run_final_sync
}

cleanup_and_sync() {
  local exit_code="${1:-0}"
  local sync_exit_code=0

  set +e
  if [[ -d "${HOST_RESULTS_DIR:-}" ]]; then
    sync_results_to_home
    sync_exit_code=$?
  fi

  if (( exit_code == 0 && sync_exit_code != 0 )); then
    return "${sync_exit_code}"
  fi

  return "${exit_code}"
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
      -v "${TOOLS_DIR_HOST}:${CONTAINER_WORKDIR}/tools:ro" \
      -v "${SHARED_PY_ROOT}/hybrid_shared:${CONTAINER_WORKDIR}/hybrid_shared:ro" \
      -v "${PYTORCH_OPT_DIR_HOST}:${CONTAINER_WORKDIR}/pytorch_opt:ro" \
      -v "${TFREC_DIR}:${CONTAINER_TFREC_DIR}:ro" \
      -w "${CONTAINER_WORKDIR}" \
      -e HOME="${CONTAINER_WORKDIR}" \
      -e PYTHONPATH="${CONTAINER_WORKDIR}" \
      -e PYTORCH_OPT_DIR="${CONTAINER_WORKDIR}/pytorch_opt" \
      -e KERAS_BACKEND="torch" \
      -e PYTHONUNBUFFERED="1" \
      -e XDG_CACHE_HOME="${CONTAINER_WORKDIR}/.cache" \
      -e HF_HOME="${CONTAINER_WORKDIR}/.cache/hf" \
      -e TRANSFORMERS_CACHE="${CONTAINER_WORKDIR}/.cache/hf" \
      -e TORCH_HOME="${CONTAINER_WORKDIR}/.cache/torch" \
      -e TMPDIR="${CONTAINER_WORKDIR}/tmp" \
      -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
      -e MASTER_ADDR="${MASTER_ADDR:-}" \
      -e MASTER_PORT="${MASTER_PORT:-}" \
      -e RANK="${RANK:-0}" \
      -e WORLD_SIZE="${WORLD_SIZE:-1}" \
      -e PROFILE_TOOL="${PROFILE_TOOL}" \
      -e ENABLE_TORCH_TRACE="${ENABLE_TORCH_TRACE}" \
      -e ENABLE_NVTX="${ENABLE_NVTX}" \
      -e PROFILE_RANK0_ONLY="${PROFILE_RANK0_ONLY}" \
      -e PROFILE_STAGES="${PROFILE_STAGES}" \
      -e TORCH_PROFILE_WAIT="${TORCH_PROFILE_WAIT}" \
      -e TORCH_PROFILE_WARMUP="${TORCH_PROFILE_WARMUP}" \
      -e TORCH_PROFILE_ACTIVE="${TORCH_PROFILE_ACTIVE}" \
      -e TORCH_PROFILE_REPEAT="${TORCH_PROFILE_REPEAT}" \
      -e TORCH_PROFILE_MAX_STEPS="${TORCH_PROFILE_MAX_STEPS}" \
      -e TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES}" \
      -e TORCH_PROFILE_PROFILE_MEMORY="${TORCH_PROFILE_PROFILE_MEMORY}" \
      -e TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK}" \
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
      -e HCPA_TORCH_PROFILER_DIR="${HCPA_TORCH_PROFILER_DIR}" \
      -e HCPA_PROFILE_ARTIFACT_ROOT="${HCPA_PROFILE_ARTIFACT_ROOT}" \
      -e HCPA_PROFILE_ARTIFACT_STEM="${HCPA_PROFILE_ARTIFACT_STEM}" \
      -e HCPA_PROFILE_ARTIFACT_PREFIX="${HCPA_PROFILE_ARTIFACT_PREFIX}" \
      -e NSYS_EXPORT_SQLITE="${NSYS_EXPORT_SQLITE}" \
      -e NCU_EXPORT_CSV="${NCU_EXPORT_CSV}" \
      -e NCU_SET="${NCU_SET}" \
      -e NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES}" \
      "${IMAGE_TAG}" \
      bash -lc "
        set -euo pipefail
        source \"${CONTAINER_WORKDIR}/tools/hcpa_profiler_launcher.sh\"
        hcpa_profile_init_env
        mkdir -p \"${CONTAINER_RUN_RESULTS_DIR}\" \"${CONTAINER_RUN_RESULTS_DIR}/torchelastic_logs\" \"${CONTAINER_WORKDIR}/tmp\" \"${CONTAINER_PROFILE_DIR}\" \"${CONTAINER_PROFILE_DIR}/torch_traces\"
        hcpa_profile_write_manifest
        cd \"${CONTAINER_MODULE_DIR}\"
        python - <<'PY'
import sys
import torch

print(f'Container torch version: {torch.__version__}')
print(f'Container torch CUDA: {torch.version.cuda}')
print(f'Container CUDA available: {torch.cuda.is_available()}')
print(f'Container CUDA device count: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'Container GPU 0: {torch.cuda.get_device_name(0)}')
else:
    sys.exit('CUDA não está disponível dentro do container.')
PY
        ${launch_cmd_str}
        hcpa_profile_postprocess
      "
  else
    host_cuda_sanity_check
    "${LAUNCH_CMD[@]}"
    hcpa_profile_postprocess
  fi
}

SSD_BASE_DIR="${SSD_BASE_DIR:-/ssd/bmmorales}"
HOME_PROJ_DIR="$(resolve_home_project_root "${MODULE_NAME}")"
PROJ_DIR="$(resolve_execution_project_root "${MODULE_NAME}" "${HOME_PROJ_DIR}")"
MODULE_DIR="${PROJ_DIR}/${MODULE_NAME}"
HOME_MODULE_DIR="${HOME_PROJ_DIR}/${MODULE_NAME}"
TFREC_DIR="$(resolve_tfrec_dir "${MODULE_DIR}" "${PROJ_DIR}" "${HOME_PROJ_DIR}")"
SUBMIT_DIR="$(resolve_dir_path "${SLURM_SUBMIT_DIR:-${MODULE_DIR}}")"
IMAGE_TAG="${IMAGE_TAG:-${MODULE_NAME}:latest}"
CONTAINER_WORKDIR="${CONTAINER_WORKDIR:-/workspace}"
CONTAINER_MODULE_DIR="${CONTAINER_WORKDIR}/${MODULE_NAME}"
CONTAINER_TFREC_DIR="${CONTAINER_WORKDIR}/data/all-tfrec"
USE_DOCKER="${USE_DOCKER:-1}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
TOOLS_DIR_HOST="${TOOLS_DIR_HOST:-${HOME_PROJ_DIR}/tools}"
SHARED_PY_ROOT="${SHARED_PY_ROOT:-${HOME_PROJ_DIR}}"
PYTORCH_OPT_DIR_HOST="${PYTORCH_OPT_DIR_HOST:-${HOME_PROJ_DIR}/pytorch_opt}"
FINAL_SYNC="${FINAL_SYNC:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -d "${MODULE_DIR}" ]]; then
  echo "Erro: diretório do módulo não existe: ${MODULE_DIR}"
  echo "Defina PROJ_DIR explicitamente ou submeta o job a partir da raiz do projeto."
  exit 1
fi

cd "${MODULE_DIR}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
IMG_SIZE="${IMG_SIZE:-320}"
PATCH_SIZE="${PATCH_SIZE:-16}"
EMBED_DIM="${EMBED_DIM:-768}"
NUM_TRANSFORMER_LAYERS="${NUM_TRANSFORMER_LAYERS:-12}"
NUM_HEADS="${NUM_HEADS:-12}"
MLP_RATIO="${MLP_RATIO:-4.0}"
DROPOUT="${DROPOUT:-0.1}"
USE_CLS_TOKEN="${USE_CLS_TOKEN:-1}"

ENABLE_AMP="${ENABLE_AMP:-1}"
ENABLE_EMA="${ENABLE_EMA:-0}"
ENABLE_DALI="${ENABLE_DALI:-0}"
ENABLE_COSINE="${ENABLE_COSINE:-1}"
ENABLE_FLASH_ATTENTION="${ENABLE_FLASH_ATTENTION:-0}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SEED="${SEED:-42}"

if [[ ! -d "${TFREC_DIR}" ]]; then
  echo "Erro: TFREC_DIR não existe: ${TFREC_DIR}"
  exit 1
fi

ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_TASK_ID}}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_ID=0; fi
if ! [[ "${ARRAY_TASK_MAX}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_MAX="${ARRAY_TASK_ID}"; fi
JOB_ID_TAG="${ARRAY_JOB_ID}"
RUN_START="${RUN_START:-${ARRAY_TASK_ID}}"
RUN_END="${RUN_END:-${ARRAY_TASK_ID}}"
if ! [[ "${RUN_START}" =~ ^[0-9]+$ && "${RUN_END}" =~ ^[0-9]+$ ]]; then
  echo "Erro: RUN_START e RUN_END devem ser inteiros."
  exit 1
fi
if (( RUN_START < 0 || RUN_END > 9 || RUN_START > RUN_END )); then
  echo "Erro: faixa de runs inválida (${RUN_START}-${RUN_END}). Esperado 0..9."
  exit 1
fi

GPUS_PER_NODE="$(parse_int_or_default "${SLURM_GPUS_PER_TASK:-1}" 1)"
NNODES="$(parse_int_or_default "${SLURM_JOB_NUM_NODES:-1}" 1)"
TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS <= 0 )); then TOTAL_GPUS=1; fi
GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)}"
GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW:-gpu}")"
if [[ -z "${GPU_TAG}" || "${GPU_TAG}" == "unknown" ]]; then
  GPU_TAG="gpu"
fi
GPU_DESC="${TOTAL_GPUS}x${GPU_TAG}"
NODE_NAME_RAW="${SLURM_NODELIST:-$(hostname)}"
NODEGROUP_RAW="${NODE_NAME_RAW%%,*}"
NODEGROUP_RAW="${NODEGROUP_RAW%%[*}"
NODEGROUP_RAW="$(printf '%s' "${NODEGROUP_RAW}" | sed -E 's/[0-9]+$//')"
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-nodes}")"
BATCH_TAG="bs$(sanitize_tag "${BATCH_SIZE}")"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/result${JOB_ID_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}}"
HOST_RESULTS_DIR="${HOST_RESULTS_DIR:-${MODULE_DIR}/${RESULTS_SUBDIR}}"
if [[ "${HOST_RESULTS_DIR}" != "${MODULE_DIR}/"* ]]; then
  echo "Erro: HOST_RESULTS_DIR deve ficar dentro de ${MODULE_DIR}"
  exit 1
fi
HOME_RESULTS_DIR="${HOME_RESULTS_DIR:-${HOME_MODULE_DIR}/${RESULTS_SUBDIR}}"
if [[ "${HOME_RESULTS_DIR}" != "${HOME_MODULE_DIR}/"* ]]; then
  echo "Erro: HOME_RESULTS_DIR deve ficar dentro de ${HOME_MODULE_DIR}"
  exit 1
fi
if [[ "${SUBMIT_DIR}" == "${PROJ_DIR}" ]]; then
  PERSISTENT_SUBMIT_DIR="${HOME_PROJ_DIR}"
elif [[ "${SUBMIT_DIR}" == "${PROJ_DIR}/"* ]]; then
  PERSISTENT_SUBMIT_DIR="${HOME_PROJ_DIR}/${SUBMIT_DIR#${PROJ_DIR}/}"
else
  PERSISTENT_SUBMIT_DIR="${SUBMIT_DIR}"
fi
HOST_LOGS_DIR="${SUBMIT_DIR}/logs"
HOME_LOGS_DIR="${PERSISTENT_SUBMIT_DIR}/logs"
SYNC_LOGS_TO_HOME="${SYNC_LOGS_TO_HOME:-0}"
if [[ "${SYNC_LOGS_TO_HOME}" == "auto" ]]; then
  if [[ "${SUBMIT_DIR}" == "${HOME_PROJ_DIR}" || "${SUBMIT_DIR}" == "${HOME_PROJ_DIR}/"* ]]; then
    SYNC_LOGS_TO_HOME="0"
  else
    SYNC_LOGS_TO_HOME="1"
  fi
fi
RELATIVE_RESULTS_DIR="${HOST_RESULTS_DIR#${MODULE_DIR}/}"
CONTAINER_RESULTS_ROOT="${CONTAINER_MODULE_DIR}/${RELATIVE_RESULTS_DIR}"
RUN_RESULTS_DIR_HOST=""
CONTAINER_RUN_RESULTS_DIR=""
HOME_RUN_RESULTS_DIR=""
TRAIN_SCRIPT_PATH=""
TRAIN_TFREC_DIR=""
TRAIN_RESULTS_DIR=""

PROFILER_HELPER="${PROJ_DIR}/tools/hcpa_profiler_launcher.sh"
if [[ ! -f "${PROFILER_HELPER}" ]]; then
  PROFILER_HELPER="${HOME_PROJ_DIR}/tools/hcpa_profiler_launcher.sh"
fi
if [[ ! -f "${PROFILER_HELPER}" ]]; then
  echo "Erro: helper de profiling não encontrado em ${PROFILER_HELPER}"
  exit 1
fi
source "${PROFILER_HELPER}"

trap 'exit_code=$?; trap - EXIT; cleanup_and_sync "${exit_code}"; exit "$?"' EXIT

mkdir -p logs "${HOST_RESULTS_DIR}"
{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "array_task_id=${ARRAY_TASK_ID}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "module=${MODULE_NAME}"
  echo "persistent_project_root=${HOME_PROJ_DIR}"
  echo "execution_project_root=${PROJ_DIR}"
  echo "submit_dir=${SUBMIT_DIR}"
  echo "persistent_submit_dir=${PERSISTENT_SUBMIT_DIR}"
  echo "nodelist=${SLURM_NODELIST:-}"
  echo "nodegroup=${NODEGROUP_TAG}"
  echo "gpu=${GPU_DESC}"
  echo "batch_size=${BATCH_SIZE}"
  echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  echo "epochs=${EPOCHS}"
  echo "enable_amp=${ENABLE_AMP}"
  echo "enable_ema=${ENABLE_EMA}"
  echo "enable_dali=${ENABLE_DALI}"
  echo "enable_cosine=${ENABLE_COSINE}"
  echo "enable_flash_attention=${ENABLE_FLASH_ATTENTION}"
  echo "patch_size=${PATCH_SIZE}"
  echo "embed_dim=${EMBED_DIM}"
  echo "ssd_base_dir=${SSD_BASE_DIR}"
  echo "persistent_results_dir=${HOME_RESULTS_DIR}"
  echo "execution_results_dir=${HOST_RESULTS_DIR}"
  echo "sync_logs_to_home=${SYNC_LOGS_TO_HOME}"
  echo "final_sync=${FINAL_SYNC}"
  echo "results_subdir=${RESULTS_SUBDIR}"
  echo "run_start=${RUN_START}"
  echo "run_end=${RUN_END}"
} > "${HOST_RESULTS_DIR}/env_manifest.txt"

echo "=========================================="
echo "Pure ViT Training"
echo "=========================================="
echo "Submit hint: sbatch --nodelist=<node> train.slurm"
echo "Job ID Tag: ${JOB_ID_TAG}"
echo "Run Range: ${RUN_START}-${RUN_END}"
echo "GPU Count: ${TOTAL_GPUS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Gradient Accumulation Steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Image Size: ${IMG_SIZE}"
echo "Patch Size: ${PATCH_SIZE}"
echo "Embed Dim: ${EMBED_DIM}"
echo "Transformer Layers: ${NUM_TRANSFORMER_LAYERS}"
echo "Heads: ${NUM_HEADS}"
echo "Flash/SDPA Attention: ${ENABLE_FLASH_ATTENTION}"
echo "Token Source: raw image patches"
echo "Persistent Project Root: ${HOME_PROJ_DIR}"
echo "Execution Project Root: ${PROJ_DIR}"
echo "Submit Dir: ${SUBMIT_DIR}"
echo "Persistent Submit Dir: ${PERSISTENT_SUBMIT_DIR}"
echo "Persistent Module Dir: ${HOME_MODULE_DIR}"
echo "Execution Module Dir: ${MODULE_DIR}"
echo "SSD Base Dir: ${SSD_BASE_DIR}"
echo "TFRecords: ${TFREC_DIR}"
echo "Persistent Results Root: ${HOME_RESULTS_DIR}"
echo "Execution Results Root: ${HOST_RESULTS_DIR}"
echo "Sync Logs To Home: ${SYNC_LOGS_TO_HOME}"
echo "Final Sync: ${FINAL_SYNC}"
echo "=========================================="
print_host_gpu_summary

RANK="${SLURM_NODEID:-0}"

for RUN_ID in $(seq "${RUN_START}" "${RUN_END}"); do
  RUN_SEED=$((SEED + RUN_ID))
  EXEC_ID="${RUN_ID}"
  RUN_RESULTS_DIR_HOST="${HOST_RESULTS_DIR}/run_${RUN_ID}"
  CONTAINER_RUN_RESULTS_DIR="${CONTAINER_RESULTS_ROOT}/run_${RUN_ID}"
  HOME_RUN_RESULTS_DIR="${HOME_RESULTS_DIR}/run_${RUN_ID}"
  TRAIN_SCRIPT_PATH="${MODULE_DIR}/train.py"
  TRAIN_TFREC_DIR="${TFREC_DIR}"
  TRAIN_RESULTS_DIR="${RUN_RESULTS_DIR_HOST}"
  if [[ "${USE_DOCKER}" == "1" ]]; then
    TRAIN_SCRIPT_PATH="${CONTAINER_MODULE_DIR}/train.py"
    TRAIN_TFREC_DIR="${CONTAINER_TFREC_DIR}"
    TRAIN_RESULTS_DIR="${CONTAINER_RUN_RESULTS_DIR}"
  fi

  HOST_PROFILE_DIR="${RUN_RESULTS_DIR_HOST}/profiling"
  CONTAINER_PROFILE_DIR="${CONTAINER_RUN_RESULTS_DIR}/profiling"
  PROFILE_ARTIFACT_STEM="${MODULE_NAME}_r${RUN_ID}"
  hcpa_profile_init_env
  mkdir -p "${HOST_PROFILE_DIR}" "${HOST_PROFILE_DIR}/torch_traces" "${RUN_RESULTS_DIR_HOST}" "${RUN_RESULTS_DIR_HOST}/torchelastic_logs"
  if [[ "${USE_DOCKER}" == "1" ]]; then
    export HCPA_TORCH_PROFILER_DIR="${CONTAINER_PROFILE_DIR}/torch_traces"
    export HCPA_PROFILE_ARTIFACT_ROOT="${CONTAINER_PROFILE_DIR}"
    export HCPA_PROFILE_ARTIFACT_STEM="${PROFILE_ARTIFACT_STEM}"
    export HCPA_PROFILE_ARTIFACT_PREFIX="${CONTAINER_PROFILE_DIR}/${PROFILE_ARTIFACT_STEM}"
  else
    hcpa_profile_prepare_paths "${HOST_PROFILE_DIR}" "${PROFILE_ARTIFACT_STEM}"
    hcpa_profile_write_manifest
  fi

  {
    echo "run_id=${RUN_ID}"
    echo "seed=${RUN_SEED}"
    echo "exec_id=${EXEC_ID}"
    echo "persistent_run_results=${HOME_RUN_RESULTS_DIR}"
    echo "execution_run_results=${RUN_RESULTS_DIR_HOST}"
    echo "img_size=${IMG_SIZE}"
    echo "patch_size=${PATCH_SIZE}"
    echo "embed_dim=${EMBED_DIM}"
    echo "num_transformer_layers=${NUM_TRANSFORMER_LAYERS}"
    echo "num_heads=${NUM_HEADS}"
    echo "use_cls_token=${USE_CLS_TOKEN}"
    echo "enable_amp=${ENABLE_AMP}"
    echo "enable_ema=${ENABLE_EMA}"
    echo "enable_dali=${ENABLE_DALI}"
    echo "enable_cosine=${ENABLE_COSINE}"
    echo "enable_flash_attention=${ENABLE_FLASH_ATTENTION}"
    echo "gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}"
  } > "${RUN_RESULTS_DIR_HOST}/run_manifest.txt"

  echo "[RUN] ${MODULE_NAME} job=${JOB_ID_TAG} run=${RUN_ID}"

  CLS_ARGS=()
  if [[ "${USE_CLS_TOKEN}" == "1" ]]; then
    CLS_ARGS+=(--use_cls_token)
  else
    CLS_ARGS+=(--no_cls_token)
  fi

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

  FLASH_ARGS=()
  if [[ "${ENABLE_FLASH_ATTENTION}" == "1" ]]; then
    FLASH_ARGS+=(--enable_flash_attention)
  else
    FLASH_ARGS+=(--disable_flash_attention)
  fi

  TRAIN_ARGS=(
    --tfrec_dir "${TRAIN_TFREC_DIR}"
    --results_dir "${TRAIN_RESULTS_DIR}"
    --exec_id "${EXEC_ID}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --seed "${RUN_SEED}"
    --lrate "${LEARNING_RATE}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --img_size "${IMG_SIZE}"
    --patch_size "${PATCH_SIZE}"
    --embed_dim "${EMBED_DIM}"
    --num_transformer_layers "${NUM_TRANSFORMER_LAYERS}"
    --num_heads "${NUM_HEADS}"
    --mlp_ratio "${MLP_RATIO}"
    --dropout "${DROPOUT}"
    "${CLS_ARGS[@]}"
    "${AMP_ARGS[@]}"
    "${EMA_ARGS[@]}"
    "${DALI_ARGS[@]}"
    "${COSINE_ARGS[@]}"
    "${FLASH_ARGS[@]}"
  )

  if [[ ${TOTAL_GPUS} -gt 1 ]]; then
    export MASTER_ADDR="${MASTER_ADDR:-$(hostname -f)}"
    export MASTER_PORT="${MASTER_PORT:-$((29500 + RUN_ID))}"
    export RANK="${RANK}"
    export WORLD_SIZE="${TOTAL_GPUS}"

    echo "Launching distributed training: world_size=${WORLD_SIZE} rank=${RANK}"
    LAUNCH_CMD=(
      torchrun
      --nnodes="${NNODES}"
      --nproc_per_node="${GPUS_PER_NODE}"
      --node_rank="${RANK}"
      --rdzv_backend=c10d
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
      "${TRAIN_SCRIPT_PATH}"
      "${TRAIN_ARGS[@]}"
    )
  else
    echo "Launching single-GPU training on GPU 0"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    LAUNCH_CMD=(
      python3
      -u
      "${TRAIN_SCRIPT_PATH}"
      "${TRAIN_ARGS[@]}"
    )
  fi

  hcpa_profile_wrap_command "${LAUNCH_CMD[@]}"
  LAUNCH_CMD=("${PROFILED_CMD[@]}")

  run_launch_cmd
done

echo "Training complete! Iniciando sincronização final para a home."
