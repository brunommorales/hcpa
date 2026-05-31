#!/usr/bin/env bash

set -euxo pipefail

: "${MODULE_NAME:?MODULE_NAME precisa estar definido no wrapper.}"
: "${HCPA_DISTRIBUTED_ARCH:?HCPA_DISTRIBUTED_ARCH precisa estar definido no wrapper.}"

required_functions=(
  hcpa_module_set_support
  hcpa_module_emit_env_manifest
  hcpa_module_emit_run_manifest
  hcpa_module_build_train_args
)
for fn_name in "${required_functions[@]}"; do
  if ! declare -F "${fn_name}" >/dev/null 2>&1; then
    echo "Erro: função obrigatória ausente no wrapper: ${fn_name}" >&2
    exit 64
  fi
done

if ! declare -F hcpa_module_profile_stem >/dev/null 2>&1; then
  hcpa_module_profile_stem() {
    printf '%s' "${MODULE_NAME}_r${RUN_ID}"
  }
fi

detect_iface_ip() {
  local target_host="$1"
  local preferred_iface="${2:-}"
  srun -N1 -n1 --ntasks-per-node=1 -w "${target_host}" bash -s "${preferred_iface}" <<'EOS'
set -euo pipefail
pref_iface="${1:-}"
detect_private() {
  ip -o -4 addr show scope global up | awk '
    function is_private(ip) {
      split(ip, o, ".")
      if (o[1]==10) return 1
      if (o[1]==192 && o[2]==168) return 1
      if (o[1]==172 && o[2]>=16 && o[2]<=31) return 1
      return 0
    }
    {
      split($4, a, "/")
      ip = a[1]
      if (is_private(ip)) {
        print $2, ip
        exit
      }
      last_if = $2
      last_ip = ip
    }
    END {
      if (NR > 0 && last_if != "" && last_ip != "")
        print last_if, last_ip
    }
  '
}
if [[ -n "${pref_iface}" ]]; then
  ip -o -4 addr show dev "${pref_iface}" scope global up | awk '{split($4, a, "/"); print $2, a[1]; exit}'
else
  detect_private
fi
EOS
}

parse_tasks_per_node() {
  local spec="${SLURM_TASKS_PER_NODE:-}"
  spec="${spec// /}"
  local total=0
  local token count repeat
  local tokens=()
  IFS=',' read -ra tokens <<<"${spec}"
  for token in "${tokens[@]}"; do
    if [[ -z "${token}" ]]; then
      continue
    fi
    if [[ "${token}" =~ ^([0-9]+)\(x([0-9]+)\)$ ]]; then
      count="${BASH_REMATCH[1]}"
      repeat="${BASH_REMATCH[2]}"
      total=$(( total + count * repeat ))
    elif [[ "${token}" =~ ^([0-9]+)$ ]]; then
      total=$(( total + BASH_REMATCH[1] ))
    fi
  done
  echo "${total}"
}

sanitize_tag() {
  local raw="${1:-}"
  raw="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${raw}" || "${raw}" == "(null)" || "${raw}" == "null" ]]; then
    printf '%s' "unknown"
    return 0
  fi
  raw="$(printf '%s' "${raw}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  raw="$(printf '%s' "${raw}" | sed -E 's/-+/-/g')"
  printf '%s' "${raw}"
}

hcpa_choose_existing_dir() {
  local explicit="${1:-}"
  shift || true
  local candidate
  if [[ -n "${explicit}" ]]; then
    candidate="$(hcpa_resolve_storage_alias_dir "${explicit}")"
    if [[ -n "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
    printf '%s' "${explicit}"
    return 0
  fi
  for candidate in "$@"; do
    candidate="$(hcpa_resolve_storage_alias_dir "${candidate}")"
    if [[ -n "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  printf '%s' "${1:-}"
}

hcpa_resolve_storage_alias_dir() {
  local original="${1:-}"
  local root candidate suffix
  if [[ -z "${original}" ]]; then
    return 1
  fi
  if [[ -d "${original}" ]]; then
    printf '%s' "${original}"
    return 0
  fi
  case "${original}" in
    /ssd/bmmorales/*)
      suffix="${original#/ssd/bmmorales}"
      ;;
    /scratch/bmmorales/*)
      suffix="${original#/scratch/bmmorales}"
      ;;
    *)
      return 1
      ;;
  esac
  for root in /ssd/bmmorales /scratch/bmmorales; do
    candidate="${root}${suffix}"
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done
  return 1
}

PERSISTENT_ROOT_DIR="${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}"
SSD_BASE_DIR="${SSD_BASE_DIR:-/ssd/bmmorales}"
REQUIRE_SSD="${REQUIRE_SSD:-1}"
FINAL_SYNC="${FINAL_SYNC:-0}"
SYNC_LOGS_TO_HOME="${SYNC_LOGS_TO_HOME:-0}"
RUNTIME_HELPER="${PERSISTENT_ROOT_DIR}/tools/hcpa_ssd_runtime.sh"
if [[ ! -f "${RUNTIME_HELPER}" ]]; then
  echo "[CHECK][FAIL] helper de runtime ausente: ${RUNTIME_HELPER}" >&2
  exit 91
fi
source "${RUNTIME_HELPER}"
hcpa_runtime_init_paths "${MODULE_NAME}"
FAST_STORAGE_BASE="$(hcpa_runtime_resolve_fast_storage_base)"
STORAGE_ROOT_CANDIDATES_STR="$(printf '%s\n%s\n' '/ssd/bmmorales' '/scratch/bmmorales')"

IMAGE="${IMAGE:-${MODULE_NAME}_distributed_${HCPA_DISTRIBUTED_ARCH}:latest}"
DATASET_NAME="${DATASET_NAME:-all-tfrec}"
DEFAULT_TFREC_SOURCE_DIR="${DEFAULT_TFREC_SOURCE_DIR:-${HOME_ROOT_DIR}/data/all-tfrec}"
TFREC_SOURCE_DIR="${TFREC_SOURCE_DIR:-${DEFAULT_TFREC_SOURCE_DIR}}"
TFREC_STAGE_NAME="${TFREC_STAGE_NAME:-${DATASET_NAME}}"
DISTRIBUTED_STAGE_TFREC="${DISTRIBUTED_STAGE_TFREC:-1}"

DOCKER_TMPDIR="${DOCKER_TMPDIR:-${FAST_STORAGE_BASE}/tmp/docker_${SLURM_JOB_ID:-$$}}"
export DOCKER_TMPDIR DOCKER_BUILDKIT=1
if ! mkdir -p "${DOCKER_TMPDIR}" 2>/dev/null; then
  DOCKER_TMPDIR="/tmp/docker_${SLURM_JOB_ID:-$$}"
  mkdir -p "${DOCKER_TMPDIR}"
fi

if [[ -n "${HOST_TFREC_DIR:-}" ]]; then
  if [[ -d "${HOST_TFREC_DIR}" ]]; then
    HOST_TFREC_DIR="$(hcpa_runtime_resolve_dir_path "${HOST_TFREC_DIR}")"
  fi
elif [[ "${DISTRIBUTED_STAGE_TFREC}" == "1" ]]; then
  if ! HOST_TFREC_DIR="$(hcpa_runtime_stage_shared_tfrec_to_ssd "${TFREC_SOURCE_DIR}" "${TFREC_STAGE_NAME}")"; then
    echo "[SSD][WARN] stage dos TFRecords falhou; usando ${TFREC_SOURCE_DIR} diretamente" >&2
    HOST_TFREC_DIR="${TFREC_SOURCE_DIR}"
  fi
  if [[ -d "${HOST_TFREC_DIR}" ]]; then
    HOST_TFREC_DIR="$(hcpa_runtime_resolve_dir_path "${HOST_TFREC_DIR}")"
  fi
else
  HOST_TFREC_DIR="${TFREC_SOURCE_DIR}"
  if [[ -d "${HOST_TFREC_DIR}" ]]; then
    HOST_TFREC_DIR="$(hcpa_runtime_resolve_dir_path "${HOST_TFREC_DIR}")"
  fi
fi

CONT_WORKDIR="${CONT_WORKDIR:-/workspace}"
CONT_TFREC_DIR="${CONT_TFREC_DIR:-${CONT_WORKDIR}/data/all-tfrec}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
INTERNAL_IFACE="${INTERNAL_IFACE:-}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_TASK_ID}}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
JOB_ID_TAG="${ARRAY_JOB_ID}"
RUN_START="${RUN_START:-${ARRAY_TASK_ID}}"
RUN_END="${RUN_END:-${ARRAY_TASK_ID}}"
SEED="${SEED:-42}"
SRUN_MAX_RETRIES="${SRUN_MAX_RETRIES:-1}"
SRUN_RETRY_SLEEP="${SRUN_RETRY_SLEEP:-30}"

if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_ID=0; fi
if ! [[ "${ARRAY_TASK_MAX}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_MAX="${ARRAY_TASK_ID}"; fi
if ! [[ "${RUN_START}" =~ ^[0-9]+$ && "${RUN_END}" =~ ^[0-9]+$ ]]; then
  echo "[CHECK][FAIL] RUN_START e RUN_END devem ser inteiros." >&2
  exit 92
fi
if (( RUN_START < 0 || RUN_END > 9 || RUN_START > RUN_END )); then
  echo "[CHECK][FAIL] Faixa de runs inválida: ${RUN_START}-${RUN_END}. Esperado 0..9." >&2
  exit 92
fi
if ! [[ "${SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "[CHECK][FAIL] SEED deve ser inteiro." >&2
  exit 92
fi

GPUS_PER_TASK="${SLURM_GPUS_PER_TASK:-1}"
if ! [[ "${GPUS_PER_TASK}" =~ ^[0-9]+$ ]]; then
  GPUS_PER_TASK=1
fi
TASK_COUNT="${SLURM_NTASKS:-}"
if ! [[ "${TASK_COUNT}" =~ ^[0-9]+$ ]]; then
  TASK_COUNT="$(parse_tasks_per_node)"
fi
if ! [[ "${TASK_COUNT}" =~ ^[0-9]+$ ]] || (( TASK_COUNT <= 0 )); then
  NODE_COUNT="${SLURM_JOB_NUM_NODES:-1}"
  if ! [[ "${NODE_COUNT}" =~ ^[0-9]+$ ]]; then
    NODE_COUNT=1
  fi
  TASK_COUNT="${NODE_COUNT}"
fi
NNODES="${SLURM_JOB_NUM_NODES:-1}"
if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || (( NNODES <= 0 )); then
  NNODES=1
fi
GPUS_PER_NODE="${GPUS_PER_TASK}"
TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
if (( TOTAL_GPUS <= 0 )); then
  TOTAL_GPUS=1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
  if (( ${#_cvd[@]} > 0 )); then
    GPUS_PER_NODE=${#_cvd[@]}
    TOTAL_GPUS=$(( NNODES * GPUS_PER_NODE ))
  fi
fi
echo "[CHECK][INFO] NODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} -> TOTAL_GPUS=${TOTAL_GPUS}"

PARTITION_TAG_RAW="${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}"
if [[ -z "${PARTITION_TAG_RAW}" && -n "${SLURM_JOB_ID:-}" ]]; then
  PARTITION_TAG_RAW="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*Partition=\([^ ]*\).*/\1/p' | head -n1)"
fi
PARTITION_TAG="$(sanitize_tag "${PARTITION_TAG_RAW:-unknown}")"
DATASET_TAG="$(sanitize_tag "${DATASET_NAME}")"
CLUSTER_TAG="$(sanitize_tag "${SLURM_CLUSTER_NAME:-unknown}")"
GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)}"
GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW:-gpu}")"
if [[ -z "${GPU_TAG}" || "${GPU_TAG}" == "unknown" ]]; then
  GPU_TAG="gpu"
fi
GPU_DESC="${TOTAL_GPUS}x${GPU_TAG}"
NODEGROUP_RAW=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  FIRST_SEG="${SLURM_NODELIST%%,*}"
  FIRST_SEG="${FIRST_SEG%%[*}"
  NODEGROUP_RAW="$(printf '%s' "${FIRST_SEG}" | sed -E 's/[0-9]+$//')"
fi
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-nodes}")"

BATCH_TAG="bs$(sanitize_tag "${TRAIN_BATCH_SIZE:-${BATCH_SIZE:-unknown}}")"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/result${JOB_ID_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}}"
HOST_RESULTS_DIR="${HOST_RESULTS_DIR:-${HOST_PROJ_DIR}/${RESULTS_SUBDIR}}"
HOME_RESULTS_DIR="${HOME_RESULTS_DIR:-${HOME_PROJ_DIR}/${RESULTS_SUBDIR}}"
if [[ "${HOST_RESULTS_DIR}" != "${HOST_PROJ_DIR}/"* ]]; then
  echo "[CHECK][FAIL] HOST_RESULTS_DIR deve ficar dentro de ${HOST_PROJ_DIR}" >&2
  exit 93
fi
if [[ "${HOME_RESULTS_DIR}" != "${HOME_PROJ_DIR}/"* ]]; then
  echo "[CHECK][FAIL] HOME_RESULTS_DIR deve ficar dentro de ${HOME_PROJ_DIR}" >&2
  exit 93
fi
RUN_DIR_BASE="${HOST_RESULTS_DIR}"
HOST_LOGS_DIR="${HOST_PROJ_DIR}/logs"
HOME_LOGS_DIR="${HOME_PROJ_DIR}/logs"

HCPA_SUPPORT_LABELS=()
HCPA_SUPPORT_DIRS=()
HCPA_DOCKER_MOUNT_SPECS=()
hcpa_module_set_support

if (( ${#HCPA_SUPPORT_LABELS[@]} != ${#HCPA_SUPPORT_DIRS[@]} )); then
  echo "Erro: HCPA_SUPPORT_LABELS e HCPA_SUPPORT_DIRS precisam ter o mesmo tamanho." >&2
  exit 65
fi

hcpa_stage_support_dirs_to_fast_storage() {
  HCPA_SUPPORT_SOURCE_SPECS_STR=""
  if [[ "${DISTRIBUTED_STAGE_SUPPORT:-1}" != "1" ]]; then
    return 0
  fi

  local source_specs=()
  local idx label old_path source_dir staged_dir home_candidate mount_idx mount_spec host_mount tail_mount
  for idx in "${!HCPA_SUPPORT_DIRS[@]}"; do
    label="${HCPA_SUPPORT_LABELS[idx]}"
    old_path="${HCPA_SUPPORT_DIRS[idx]}"
    source_dir="${old_path}"
    home_candidate="${HOME_ROOT_DIR}/${label}"

    if [[ -d "${home_candidate}" ]]; then
      case "${old_path}" in
        "${HOME_ROOT_DIR}/${label}"|"${HOST_ROOT_DIR}/${label}"|"/ssd/bmmorales/${label}"|"/scratch/bmmorales/${label}")
          source_dir="${home_candidate}"
          ;;
      esac
    fi
    source_specs+=("${label}=${source_dir}")

    if ! staged_dir="$(hcpa_runtime_stage_module_to_ssd "${label}" "${source_dir}")"; then
      echo "[SSD][WARN] stage do suporte ${label} falhou; usando ${source_dir}" >&2
      staged_dir="${source_dir}"
    fi

    HCPA_SUPPORT_DIRS[idx]="${staged_dir}"
    for mount_idx in "${!HCPA_DOCKER_MOUNT_SPECS[@]}"; do
      mount_spec="${HCPA_DOCKER_MOUNT_SPECS[mount_idx]}"
      host_mount="${mount_spec%%:*}"
      tail_mount="${mount_spec#*:}"
      if [[ "${host_mount}" == "${old_path}" ]]; then
        HCPA_DOCKER_MOUNT_SPECS[mount_idx]="${staged_dir}:${tail_mount}"
      fi
    done
  done

  if (( ${#source_specs[@]} > 0 )); then
    HCPA_SUPPORT_SOURCE_SPECS_STR="$(printf '%s\n' "${source_specs[@]}")"
  fi
}

hcpa_stage_support_dirs_to_fast_storage

HCPA_SUPPORT_SPECS_STR=""
if (( ${#HCPA_SUPPORT_LABELS[@]} > 0 )); then
  HCPA_SUPPORT_SPECS_STR="$(for idx in "${!HCPA_SUPPORT_LABELS[@]}"; do printf '%s=%s\n' "${HCPA_SUPPORT_LABELS[idx]}" "${HCPA_SUPPORT_DIRS[idx]}"; done)"
fi

PROFILER_HELPER="$(hcpa_runtime_resolve_support_file "tools/hcpa_profiler_launcher.sh" "${PROFILER_HELPER:-}")"
CLEAN_TEMP_HELPER="$(hcpa_runtime_resolve_support_file "clean_temp_loggers.py" "${CLEAN_TEMP_HELPER:-}")"
if [[ ! -f "${PROFILER_HELPER}" ]]; then
  echo "[CHECK][FAIL] helper de profiling ausente: ${PROFILER_HELPER}" >&2
  exit 91
fi
if [[ ! -f "${CLEAN_TEMP_HELPER}" ]]; then
  echo "[CHECK][FAIL] helper de limpeza ausente: ${CLEAN_TEMP_HELPER}" >&2
  exit 91
fi
source "${PROFILER_HELPER}"
hcpa_profile_init_env
trap 'exit_code=$?; trap - EXIT; hcpa_runtime_cleanup_and_sync "${exit_code}"; exit "$?"' EXIT

# PROFILER_HELPER pode ter caído no fallback para HOME (ex: SSD sem permissão de escrita).
# Garante que o mount de tools no container aponta para onde o arquivo realmente existe.
_tools_actual_dir="$(cd "$(dirname "${PROFILER_HELPER}")" && pwd)"
for _idx in "${!HCPA_DOCKER_MOUNT_SPECS[@]}"; do
  _spec="${HCPA_DOCKER_MOUNT_SPECS[_idx]}"
  _tail="${_spec#*:}"
  if [[ "${_tail}" == "${CONT_WORKDIR:-/workspace}/tools:ro" ]]; then
    HCPA_DOCKER_MOUNT_SPECS[_idx]="${_tools_actual_dir}:${_tail}"
    break
  fi
done
unset _tools_actual_dir _idx _spec _tail

echo "[CHECK] Iniciando pré-checagem local."
precheck_ok=true

require_var() {
  local var_name="$1"
  local human_label="$2"
  local value="${!var_name:-}"
  if [[ -z "${value}" ]]; then
    echo "[CHECK][FAIL] ${human_label} (${var_name}) vazio ou não definido."
    precheck_ok=false
  else
    echo "[CHECK][OK] ${human_label}: ${value}"
  fi
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" ]]; then
    echo "[CHECK][OK] Diretório ${label}: ${path}"
  else
    echo "[CHECK][FAIL] Diretório ${label} ausente: ${path}"
    precheck_ok=false
  fi
}

if command -v docker >/dev/null 2>&1; then
  echo "[CHECK][OK] docker disponível no nó de submissão."
else
  echo "[CHECK][FAIL] docker indisponível no nó de submissão."
  precheck_ok=false
fi

require_var IMAGE "Imagem Docker"
require_var HOST_PROJ_DIR "Diretório do módulo no host"
require_var HOST_TFREC_DIR "Diretório de TFRecords"
require_var CONT_WORKDIR "Diretório dentro do container"

check_dir "${HOST_PROJ_DIR}" "módulo"
check_dir "${HOST_TFREC_DIR}" "TFRecords"
for idx in "${!HCPA_SUPPORT_DIRS[@]}"; do
  check_dir "${HCPA_SUPPORT_DIRS[idx]}" "${HCPA_SUPPORT_LABELS[idx]}"
done
mkdir -p "${HOST_RESULTS_DIR}" && check_dir "${HOST_RESULTS_DIR}" "resultados (escrita)"
if [[ ! -w "${HOST_RESULTS_DIR}" ]]; then
  echo "[CHECK][FAIL] Diretório de resultados sem permissão de escrita: ${HOST_RESULTS_DIR}"
  precheck_ok=false
else
  echo "[CHECK][OK] Resultados escrevíveis em: ${HOST_RESULTS_DIR}"
fi
echo "[CHECK][OK] UID/GID locais: ${HOST_UID}:${HOST_GID}"

if [[ "${precheck_ok}" != true ]]; then
  echo "[CHECK] Pré-checagem falhou; abortando job."
  exit 90
fi
echo "[CHECK] Pré-checagem local concluída com sucesso."

mkdir -p "${HOST_LOGS_DIR}"

readarray -t SLURM_HOSTS < <(scontrol show hostnames "${SLURM_NODELIST}")
MASTER_HOST="${SLURM_HOSTS[0]}"
MASTER_ADDR="${MASTER_HOST}"
MASTER_IFACE=""
NODE_NET_INFO=""

for host in "${SLURM_HOSTS[@]}"; do
  info="$(detect_iface_ip "${host}" "${INTERNAL_IFACE}")"
  host_iface=""
  host_ip=""
  if [[ -n "${info}" ]]; then
    read -r host_iface host_ip <<<"${info}"
  fi
  if [[ "${host}" == "${MASTER_HOST}" ]]; then
    MASTER_IFACE="${host_iface}"
    if [[ -n "${host_ip}" ]]; then
      MASTER_ADDR="${host_ip}"
    fi
  fi
  NODE_NET_INFO+=" ${host}:${host_iface}:${host_ip}"
  echo "[NET] ${host}: iface=${host_iface:-<none>} ip=${host_ip:-<none>}"
done
NODE_NET_INFO="${NODE_NET_INFO# }"
MASTER_PORT="${MASTER_PORT:-29500}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR CONT_WORKDIR CONT_TFREC_DIR HOST_UID HOST_GID HOST_RESULTS_DIR RUN_DIR_BASE RESULTS_SUBDIR TOTAL_GPUS HOST_ROOT_DIR HOME_ROOT_DIR DATASET_NAME MODULE_NAME DOCKER_GPU_RUNTIME
export MASTER_ADDR MASTER_PORT OMP_NUM_THREADS NCCL_DEBUG TORCH_DISTRIBUTED_DEBUG NODE_NET_INFO INTERNAL_IFACE NNODES GPUS_PER_NODE
export HCPA_SUPPORT_SPECS_STR HCPA_SUPPORT_SOURCE_SPECS_STR DOCKER_TMPDIR CLEAN_TEMP_HELPER SRUN_MAX_RETRIES SRUN_RETRY_SLEEP STORAGE_ROOT_CANDIDATES_STR FAST_STORAGE_BASE DISTRIBUTED_STAGE_SUPPORT

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "array_task_id=${ARRAY_TASK_ID}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "module=${MODULE_NAME}"
  echo "cluster=${CLUSTER_TAG}"
  echo "partition=${PARTITION_TAG}"
  echo "persistent_module_dir=${HOME_PROJ_DIR}"
  echo "execution_module_dir=${HOST_PROJ_DIR}"
  echo "persistent_results_dir=${HOME_RESULTS_DIR}"
  echo "execution_results_dir=${HOST_RESULTS_DIR}"
  echo "nodelist=${SLURM_NODELIST:-}"
  echo "nodes_resolved=${SLURM_HOSTS[*]}"
  echo "nodegroup=${NODEGROUP_TAG}"
  echo "image=${IMAGE}"
  echo "world_size=${TOTAL_GPUS}"
  echo "master_addr=${MASTER_ADDR}"
  echo "master_iface=${MASTER_IFACE}"
  echo "node_net_info=${NODE_NET_INFO}"
  echo "gpu_desc=${GPU_DESC}"
  echo "run_start=${RUN_START}"
  echo "run_end=${RUN_END}"
  echo "seed_base=${SEED}"
  echo "dataset_name=${DATASET_NAME}"
  echo "host_tfrec_dir=${HOST_TFREC_DIR}"
  echo "fast_storage_base_submit=${FAST_STORAGE_BASE}"
  echo "storage_root_candidates=/ssd/bmmorales,/scratch/bmmorales"
  hcpa_module_emit_env_manifest
} > "${HOST_RESULTS_DIR}/env_manifest.txt"

srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label bash <<'EOS'
set -euo pipefail

resolve_storage_dir() {
  local original="${1:-}"
  local suffix candidate root
  if [[ -z "${original}" ]]; then
    printf '%s' "${original}"
    return 0
  fi
  if [[ -d "${original}" ]]; then
    printf '%s' "${original}"
    return 0
  fi
  case "${original}" in
    /ssd/bmmorales/*)
      suffix="${original#/ssd/bmmorales}"
      ;;
    /scratch/bmmorales/*)
      suffix="${original#/scratch/bmmorales}"
      ;;
    *)
      printf '%s' "${original}"
      return 0
      ;;
  esac
  while IFS= read -r root; do
    [[ -n "${root}" ]] || continue
    candidate="${root}${suffix}"
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done <<< "${STORAGE_ROOT_CANDIDATES_STR:-}"
  printf '%s' "${original}"
}

resolve_fast_storage_base_on_node() {
  local root
  if [[ -n "${FAST_STORAGE_BASE:-}" && -d "${FAST_STORAGE_BASE}" ]]; then
    printf '%s' "${FAST_STORAGE_BASE}"
    return 0
  fi
  while IFS= read -r root; do
    [[ -n "${root}" ]] || continue
    if [[ -d "${root}" ]]; then
      printf '%s' "${root}"
      return 0
    fi
  done <<< "${STORAGE_ROOT_CANDIDATES_STR:-}"
  printf '%s' "${FAST_STORAGE_BASE:-/scratch/bmmorales}"
}

stage_support_dirs_on_node() {
  if [[ "${DISTRIBUTED_STAGE_SUPPORT:-1}" != "1" || -z "${HCPA_SUPPORT_SOURCE_SPECS_STR:-}" ]]; then
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[$(hostname)] rsync indisponível; usando suportes já montados."
    return 0
  fi

  local local_base spec label source_dir target_dir
  local_base="$(resolve_fast_storage_base_on_node)"
  mkdir -p "${local_base}" || return 0

  while IFS= read -r spec; do
    [[ -n "${spec}" ]] || continue
    label="${spec%%=*}"
    source_dir="${spec#*=}"
    target_dir="${local_base}/${label}"
    [[ -d "${source_dir}" ]] || continue
    [[ "${source_dir}" != "${target_dir}" ]] || continue
    echo "[$(hostname)] sincronizando suporte ${label}: ${source_dir} -> ${target_dir}"
    if command -v flock >/dev/null 2>&1; then
      if ! (
        flock 9
        rsync -a --delete \
          --exclude '.git/' \
          --exclude '.venv/' \
          --exclude '__pycache__/' \
          --exclude '*.pyc' \
          --exclude '.cache/' \
          --exclude 'tmp/' \
          --exclude 'logs/' \
          --exclude 'results/' \
          --exclude 'data/' \
          "${source_dir}/" "${target_dir}/"
      ) 9>"${local_base}/.stage_${label}.lock"; then
        echo "[$(hostname)] WARN: stage do suporte ${label} falhou; usando origem/montagem existente." >&2
        continue
      fi
    else
      if ! rsync -a --delete \
        --exclude '.git/' \
        --exclude '.venv/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.cache/' \
        --exclude 'tmp/' \
        --exclude 'logs/' \
        --exclude 'results/' \
        --exclude 'data/' \
        "${source_dir}/" "${target_dir}/"; then
        echo "[$(hostname)] WARN: stage do suporte ${label} falhou; usando origem/montagem existente." >&2
        continue
      fi
    fi
  done <<< "${HCPA_SUPPORT_SOURCE_SPECS_STR}"
}

stage_support_dirs_on_node

local_fast_base="$(resolve_fast_storage_base_on_node)"
tmp_base="${local_fast_base}/tmp"
if ! mkdir -p "${tmp_base}" 2>/dev/null; then
  tmp_base="/tmp"
fi
export DOCKER_TMPDIR="${tmp_base}/docker_${SLURM_JOB_ID:-$$}"
mkdir -p "${DOCKER_TMPDIR}" || true

NET_IFACE=""
NET_IP=""
LOCAL_NODE="$(hostname)"
for entry in ${NODE_NET_INFO:-}; do
  IFS=":" read -r entry_host entry_iface entry_ip <<<"${entry}"
  if [[ "${entry_host}" == "${LOCAL_NODE}" ]]; then
    NET_IFACE="${entry_iface}"
    NET_IP="${entry_ip}"
    break
  fi
done
if [[ -n "${NET_IFACE}" || -n "${NET_IP}" ]]; then
  echo "[$(hostname)] Rede interna detectada: iface=${NET_IFACE:-<none>} ip=${NET_IP:-<none>}"
else
  echo "[$(hostname)] Rede interna não detectada automaticamente."
fi

echo "[$(hostname)] Docker:"
docker --version

echo "[$(hostname)] NVIDIA:"
nvidia-smi || { echo "nvidia-smi falhou em ${HOSTNAME}"; exit 3; }

echo "[$(hostname)] Using image: ${IMAGE}"
LOCAL_HOST_PROJ_DIR="$(resolve_storage_dir "${HOST_PROJ_DIR}")"
LOCAL_HOST_TFREC_DIR="$(resolve_storage_dir "${HOST_TFREC_DIR}")"
echo "[$(hostname)] módulo local: ${LOCAL_HOST_PROJ_DIR}"
echo "[$(hostname)] tfrecords local: ${LOCAL_HOST_TFREC_DIR}"
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[$(hostname)] Image ${IMAGE} não encontrada; iniciando docker build."
  BUILD_DIR="${LOCAL_HOST_PROJ_DIR}"
  [[ -d "${BUILD_DIR}" ]] || { echo "Diretório de build ausente em ${HOSTNAME}: ${BUILD_DIR}"; exit 128; }
  (
    cd "${BUILD_DIR}"
    docker build --build-arg UID="${HOST_UID}" --build-arg GID="${HOST_GID}" -t "${IMAGE}" .
  )
fi

[[ -d "${LOCAL_HOST_PROJ_DIR}" ]] || { echo "HOST_PROJ_DIR ausente em ${HOSTNAME}: ${LOCAL_HOST_PROJ_DIR}"; exit 2; }
[[ -d "${LOCAL_HOST_TFREC_DIR}" ]] || { echo "HOST_TFREC_DIR ausente em ${HOSTNAME}: ${LOCAL_HOST_TFREC_DIR}"; exit 2; }

while IFS= read -r spec; do
  [[ -n "${spec}" ]] || continue
  label="${spec%%=*}"
  path="${spec#*=}"
  path="$(resolve_storage_dir "${path}")"
  [[ -d "${path}" ]] || { echo "${label} ausente em ${HOSTNAME}: ${path}"; exit 2; }
done <<< "${HCPA_SUPPORT_SPECS_STR:-}"
EOS

run_distributed_once() {
  srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'EOS'
set -euxo pipefail

resolve_storage_dir() {
  local original="${1:-}"
  local suffix candidate root
  if [[ -z "${original}" ]]; then
    printf '%s' "${original}"
    return 0
  fi
  if [[ -d "${original}" ]]; then
    printf '%s' "${original}"
    return 0
  fi
  case "${original}" in
    /ssd/bmmorales/*)
      suffix="${original#/ssd/bmmorales}"
      ;;
    /scratch/bmmorales/*)
      suffix="${original#/scratch/bmmorales}"
      ;;
    *)
      printf '%s' "${original}"
      return 0
      ;;
  esac
  while IFS= read -r root; do
    [[ -n "${root}" ]] || continue
    candidate="${root}${suffix}"
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done <<< "${STORAGE_ROOT_CANDIDATES_STR:-}"
  printf '%s' "${original}"
}

resolve_fast_storage_base_on_node() {
  local root
  if [[ -n "${FAST_STORAGE_BASE:-}" && -d "${FAST_STORAGE_BASE}" ]]; then
    printf '%s' "${FAST_STORAGE_BASE}"
    return 0
  fi
  while IFS= read -r root; do
    [[ -n "${root}" ]] || continue
    if [[ -d "${root}" ]]; then
      printf '%s' "${root}"
      return 0
    fi
  done <<< "${STORAGE_ROOT_CANDIDATES_STR:-}"
  printf '%s' "${FAST_STORAGE_BASE:-/scratch/bmmorales}"
}

stage_support_dirs_on_node() {
  if [[ "${DISTRIBUTED_STAGE_SUPPORT:-1}" != "1" || -z "${HCPA_SUPPORT_SOURCE_SPECS_STR:-}" ]]; then
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[$(hostname)] rsync indisponível; usando suportes já montados."
    return 0
  fi

  local local_base spec label source_dir target_dir
  local_base="$(resolve_fast_storage_base_on_node)"
  mkdir -p "${local_base}" || return 0

  while IFS= read -r spec; do
    [[ -n "${spec}" ]] || continue
    label="${spec%%=*}"
    source_dir="${spec#*=}"
    target_dir="${local_base}/${label}"
    [[ -d "${source_dir}" ]] || continue
    [[ "${source_dir}" != "${target_dir}" ]] || continue
    echo "[$(hostname)] sincronizando suporte ${label}: ${source_dir} -> ${target_dir}"
    if command -v flock >/dev/null 2>&1; then
      if ! (
        flock 9
        rsync -a --delete \
          --exclude '.git/' \
          --exclude '.venv/' \
          --exclude '__pycache__/' \
          --exclude '*.pyc' \
          --exclude '.cache/' \
          --exclude 'tmp/' \
          --exclude 'logs/' \
          --exclude 'results/' \
          --exclude 'data/' \
          "${source_dir}/" "${target_dir}/"
      ) 9>"${local_base}/.stage_${label}.lock"; then
        echo "[$(hostname)] WARN: stage do suporte ${label} falhou; usando origem/montagem existente." >&2
        continue
      fi
    else
      if ! rsync -a --delete \
        --exclude '.git/' \
        --exclude '.venv/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        --exclude '.cache/' \
        --exclude 'tmp/' \
        --exclude 'logs/' \
        --exclude 'results/' \
        --exclude 'data/' \
        "${source_dir}/" "${target_dir}/"; then
        echo "[$(hostname)] WARN: stage do suporte ${label} falhou; usando origem/montagem existente." >&2
        continue
      fi
    fi
  done <<< "${HCPA_SUPPORT_SOURCE_SPECS_STR}"
}

stage_support_dirs_on_node

NODE_RANK="${SLURM_NODEID}"
PROC_PER_NODE="${GPUS_PER_NODE}"
WORLD_SIZE="${TOTAL_GPUS}"
CONTAINER_NAME="hcpa_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"

detect_local_iface() {
  local preferred="${1:-}"
  if [[ -n "${preferred}" ]]; then
    local out=""
    set +e
    out=$(ip -o -4 addr show dev "${preferred}" scope global up 2>/dev/null | awk '{split($4, a, "/"); print $2, a[1]; exit}')
    local st=$?
    set -e
    if [[ ${st} -eq 0 && -n "${out}" ]]; then
      echo "${out}"
      return 0
    fi
  fi
  ip -o -4 addr show scope global up | awk '
    function is_private(ip) {
      split(ip, o, ".")
      if (o[1]==10) return 1
      if (o[1]==192 && o[2]==168) return 1
      if (o[1]==172 && o[2]>=16 && o[2]<=31) return 1
      return 0
    }
    {
      split($4, a, "/")
      ip = a[1]
      if (is_private(ip)) {
        print $2, ip
        exit
      }
      last_if = $2
      last_ip = ip
    }
    END {
      if (NR > 0 && last_if != "" && last_ip != "")
        print last_if, last_ip
    }
  '
}

LOCAL_IFACE="${INTERNAL_IFACE}"
LOCAL_IP=""
LOCAL_NODE="$(hostname)"
for entry in ${NODE_NET_INFO:-}; do
  IFS=":" read -r entry_host entry_iface entry_ip <<<"${entry}"
  if [[ "${entry_host}" == "${LOCAL_NODE}" ]]; then
    if [[ -z "${LOCAL_IFACE}" ]]; then
      LOCAL_IFACE="${entry_iface}"
    fi
    LOCAL_IP="${entry_ip}"
    break
  fi
done

DETECT_OUTPUT="$(detect_local_iface "${LOCAL_IFACE}")"
if [[ -n "${DETECT_OUTPUT}" ]]; then
  read -r detected_iface detected_ip <<<"${DETECT_OUTPUT}"
  if [[ -z "${LOCAL_IFACE}" ]]; then
    LOCAL_IFACE="${detected_iface}"
  fi
  if [[ -z "${LOCAL_IP}" ]]; then
    LOCAL_IP="${detected_ip}"
  fi
fi

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${LOCAL_IFACE}}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${LOCAL_IFACE}}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

local_fast_base="$(resolve_fast_storage_base_on_node)"
tmp_base="${local_fast_base}/tmp"
if ! mkdir -p "${tmp_base}" 2>/dev/null; then
  tmp_base="/tmp"
fi
export DOCKER_TMPDIR="${tmp_base}/docker_${SLURM_JOB_ID:-$$}"
local_tmp="${tmp_base}/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}"
cache_root="${local_tmp}/cache"
mpl_dir="${local_tmp}/mplconfig"
mkdir -p "${DOCKER_TMPDIR}" "${local_tmp}" "${cache_root}" "${cache_root}/hf" "${cache_root}/torch" "${cache_root}/torchinductor/${SLURM_NODEID}" "${cache_root}/triton/${SLURM_NODEID}" "${mpl_dir}"

if [[ -n "${NCCL_SOCKET_IFNAME}" ]]; then
  echo "[$(hostname)] NCCL interface ativa: ${NCCL_SOCKET_IFNAME} (ip=${LOCAL_IP:-<unknown>})"
else
  echo "[$(hostname)] NCCL interface não definida; rede padrão será usada."
fi

DOCKER_RUNTIME_ARG=()
if [[ -n "${DOCKER_GPU_RUNTIME:-}" ]]; then
  DOCKER_RUNTIME_ARG=(--runtime "${DOCKER_GPU_RUNTIME}")
fi
cont_module_dir="${CONT_WORKDIR}/${MODULE_NAME}"

LOCAL_HOST_PROJ_DIR="$(resolve_storage_dir "${HOST_PROJ_DIR}")"
LOCAL_HOST_TFREC_DIR="$(resolve_storage_dir "${HOST_TFREC_DIR}")"
echo "[$(hostname)] módulo resolvido: ${LOCAL_HOST_PROJ_DIR}"
echo "[$(hostname)] tfrecords resolvido: ${LOCAL_HOST_TFREC_DIR}"

DOCKER_EXTRA_MOUNTS=()
while IFS= read -r mount_spec; do
  [[ -n "${mount_spec}" ]] || continue
  host_mount="${mount_spec%%:*}"
  tail_mount="${mount_spec#*:}"
  host_mount="$(resolve_storage_dir "${host_mount}")"
  DOCKER_EXTRA_MOUNTS+=(-v "${host_mount}:${tail_mount}")
done <<< "${HCPA_DOCKER_MOUNT_SPECS_STR:-}"

DOCKER_USERNS_ARGS=()
if docker --version 2>/dev/null | grep -qi podman; then
  DOCKER_USERNS_ARGS=(--userns=keep-id)
fi

docker run "${DOCKER_RUNTIME_ARG[@]}" --rm --name "${CONTAINER_NAME}" \
  "${DOCKER_USERNS_ARGS[@]}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${local_tmp}:${local_tmp}" \
  -v "${LOCAL_HOST_PROJ_DIR}:${cont_module_dir}" \
  "${DOCKER_EXTRA_MOUNTS[@]}" \
  -v "${LOCAL_HOST_TFREC_DIR}:${CONT_TFREC_DIR}:ro" \
  -w "${cont_module_dir}" \
  -e HOME="${CONT_WORKDIR}" \
  -e PYTHONPATH="${CONT_WORKDIR}:${cont_module_dir}" \
  -e PYTORCH_OPT_DIR="${CONT_WORKDIR}/pytorch_opt" \
  -e KERAS_BACKEND="torch" \
  -e PYTHONUNBUFFERED="1" \
  -e XDG_CACHE_HOME="${cache_root}" \
  -e HF_HOME="${cache_root}/hf" \
  -e TRANSFORMERS_CACHE="${cache_root}/hf" \
  -e TORCH_HOME="${cache_root}/torch" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
  -e TMPDIR="${local_tmp}" \
  -e TMP="${local_tmp}" \
  -e TEMP="${local_tmp}" \
  -e MPLCONFIGDIR="${mpl_dir}" \
  -e TORCHINDUCTOR_CACHE_DIR="${cache_root}/torchinductor/${SLURM_NODEID}" \
  -e TRITON_CACHE_DIR="${cache_root}/triton/${SLURM_NODEID}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e NODE_RANK="${NODE_RANK}" \
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
  -e NSYS_EXPORT_SQLITE="${NSYS_EXPORT_SQLITE}" \
  -e NCU_EXPORT_CSV="${NCU_EXPORT_CSV}" \
  -e NCU_SET="${NCU_SET}" \
  -e NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES}" \
  -e TRAIN_ARGS_STR="${TRAIN_ARGS_STR}" \
  -e PROFILE_ARTIFACT_STEM="${PROFILE_ARTIFACT_STEM}" \
  "${IMAGE}" \
  bash -lc "
    set -euo pipefail
    source \"${CONT_WORKDIR}/tools/hcpa_profiler_launcher.sh\"
    hcpa_profile_init_env
    mkdir -p \"./${RESULTS_SUBDIR}/run_${RUN_ID}\" \
             \"./${RESULTS_SUBDIR}/run_${RUN_ID}/torchelastic_logs\" \
             \"./${RESULTS_SUBDIR}/run_${RUN_ID}/profiling\" \
             \"./${RESULTS_SUBDIR}/run_${RUN_ID}/profiling/torch_traces\" \
             \"${local_tmp}\" \
             \"${mpl_dir}\" \
             \"${cache_root}\" \
             \"${cache_root}/hf\" \
             \"${cache_root}/torch\" \
             \"${cache_root}/torchinductor/${SLURM_NODEID}\" \
             \"${cache_root}/triton/${SLURM_NODEID}\"
    export HCPA_TORCH_PROFILER_DIR=\"./${RESULTS_SUBDIR}/run_${RUN_ID}/profiling/torch_traces\"
    export HCPA_PROFILE_ARTIFACT_ROOT=\"./${RESULTS_SUBDIR}/run_${RUN_ID}/profiling\"
    export HCPA_PROFILE_ARTIFACT_STEM=\"${PROFILE_ARTIFACT_STEM}\"
    export HCPA_PROFILE_ARTIFACT_PREFIX=\"\${HCPA_PROFILE_ARTIFACT_ROOT}/\${HCPA_PROFILE_ARTIFACT_STEM}\"
    hcpa_profile_write_manifest
    python -c 'import torch; print(\"CUDA devices:\", torch.cuda.device_count())'
    python -c 'import hybrid_shared.stability; print(\"hybrid_shared.stability import OK\")'
    read -r -a TRAIN_ARGS <<< \"\${TRAIN_ARGS_STR}\"
    LAUNCH_CMD=(
      torchrun
      --nnodes=${NNODES}
      --nproc_per_node=${GPUS_PER_NODE}
      --node_rank=\${NODE_RANK}
      --rdzv_backend=c10d
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}
      --log-dir=\"./${RESULTS_SUBDIR}/run_${RUN_ID}/torchelastic_logs\"
      train.py
      \"\${TRAIN_ARGS[@]}\"
    )
    hcpa_profile_wrap_command \"\${LAUNCH_CMD[@]}\"
    \"\${PROFILED_CMD[@]}\"
    hcpa_profile_postprocess
  "
EOS
}

for RUN_ID in $(seq "${RUN_START}" "${RUN_END}"); do
  RUN_LABEL="$((RUN_ID - RUN_START + 1))/$((RUN_END - RUN_START + 1))"
  RUN_SEED=$((SEED + RUN_ID))
  RUN_RESULTS_DIR="${HOST_RESULTS_DIR}/run_${RUN_ID}"
  TRAIN_TFREC_DIR="${CONT_TFREC_DIR}"
  TRAIN_RESULTS_DIR="./${RESULTS_SUBDIR}/run_${RUN_ID}"
  PROFILE_ARTIFACT_STEM="$(hcpa_module_profile_stem)"
  mkdir -p "${RUN_RESULTS_DIR}" "${RUN_RESULTS_DIR}/torchelastic_logs" "${RUN_RESULTS_DIR}/profiling" "${RUN_RESULTS_DIR}/profiling/torch_traces"

  {
    echo "run_id=${RUN_ID}"
    echo "seed=${RUN_SEED}"
    echo "persistent_run_results=${HOME_RESULTS_DIR}/run_${RUN_ID}"
    echo "execution_run_results=${RUN_RESULTS_DIR}"
    hcpa_module_emit_run_manifest
  } > "${RUN_RESULTS_DIR}/run_manifest.txt"

  TRAIN_ARGS=()
  hcpa_module_build_train_args
  TRAIN_ARGS_STR="$(printf '%s ' "${TRAIN_ARGS[@]}")"
  TRAIN_ARGS_STR="${TRAIN_ARGS_STR% }"
  HCPA_DOCKER_MOUNT_SPECS_STR=""
  if (( ${#HCPA_DOCKER_MOUNT_SPECS[@]} > 0 )); then
    HCPA_DOCKER_MOUNT_SPECS_STR="$(printf '%s\n' "${HCPA_DOCKER_MOUNT_SPECS[@]}")"
  fi
  export RUN_ID RUN_SEED TRAIN_ARGS_STR PROFILE_ARTIFACT_STEM HCPA_DOCKER_MOUNT_SPECS_STR
  export MASTER_PORT=$((29500 + RUN_ID))

  attempt=1
  while true; do
    if run_distributed_once; then
      break
    else
      status=$?
    fi
    if (( attempt >= SRUN_MAX_RETRIES )); then
      echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] falhou após ${attempt} tentativas (status=${status})." >&2
      exit "${status}"
    fi
    echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] srun retornou status ${status}; nova tentativa em ${SRUN_RETRY_SLEEP}s..." >&2
    sleep "${SRUN_RETRY_SLEEP}"
    attempt=$((attempt + 1))
  done

  echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] concluído."
  python3 "${CLEAN_TEMP_HELPER}" --root "${RUN_RESULTS_DIR}" --scope results-only --max-print 0 || true
done

echo "Job complete."
