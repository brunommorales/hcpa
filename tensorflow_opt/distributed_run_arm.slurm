#!/usr/bin/env bash
#SBATCH --job-name=tensorflow_opt
#SBATCH --nodes=1
#SBATCH --array=0-9%1
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --nodelist=grace1
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euxo pipefail

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

# ========================== CONFIG ==========================
IMAGE="tf_distributed_arm:latest"
MODULE_NAME="tensorflow_opt"
DOCKER_GPU_RUNTIME="${DOCKER_GPU_RUNTIME:-nvidia}"

SSD_BASE_DIR="${SSD_BASE_DIR:-/ssd/bmmorales}"
PERSISTENT_ROOT_DIR="${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}"
REQUIRE_SSD="${REQUIRE_SSD:-1}"
FINAL_SYNC="${FINAL_SYNC:-0}"
SYNC_LOGS_TO_HOME="${SYNC_LOGS_TO_HOME:-0}"
RUNTIME_HELPER="${PERSISTENT_ROOT_DIR}/tools/hcpa_ssd_runtime.sh"
if [[ ! -f "${RUNTIME_HELPER}" ]]; then
  echo "[CHECK][FAIL] helper de runtime ausente: ${RUNTIME_HELPER}"
  exit 91
fi
source "${RUNTIME_HELPER}"
hcpa_runtime_init_paths "${MODULE_NAME}"

DEFAULT_TFREC_SOURCE_DIR="${DEFAULT_TFREC_SOURCE_DIR:-${HOME_ROOT_DIR}/data/all-tfrec}"
if [[ -n "${HOST_TFREC_DIR:-}" ]]; then
  if [[ -d "${HOST_TFREC_DIR}" ]]; then
    HOST_TFREC_DIR="$(hcpa_runtime_resolve_dir_path "${HOST_TFREC_DIR}")"
  fi
else
  if ! HOST_TFREC_DIR="$(hcpa_runtime_stage_shared_tfrec_to_ssd "${DEFAULT_TFREC_SOURCE_DIR}" "all-tfrec")"; then
    echo "[SSD][WARN] stage dos TFRecords falhou; usando ${DEFAULT_TFREC_SOURCE_DIR} diretamente" >&2
    HOST_TFREC_DIR="${DEFAULT_TFREC_SOURCE_DIR}"
  fi
fi
CONT_WORKDIR="/workspace"

INTERNAL_IFACE="${INTERNAL_IFACE:-}"
export INTERNAL_IFACE DOCKER_GPU_RUNTIME
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_TASK_ID}}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_ID=0; fi
if ! [[ "${ARRAY_TASK_MAX}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_MAX="${ARRAY_TASK_ID}"; fi
RUN_ID_OVERRIDE="${RUN_ID_OVERRIDE:-}"
JOB_ID_TAG="${ARRAY_JOB_ID}"
CHECK_VERBOSE="${HCPA_PRECHECK_VERBOSE:-0}"
TARGET_EPOCHS="${TARGET_EPOCHS:-200}"
BASE_SEED="${SEED:-42}"
RUN_START="${RUN_START:-${ARRAY_TASK_ID}}"
RUN_END="${RUN_END:-${ARRAY_TASK_ID}}"
if [[ "${RUN_ID_OVERRIDE}" =~ ^[0-9]+$ ]]; then
  RUN_START="${RUN_ID_OVERRIDE}"
  RUN_END="${RUN_ID_OVERRIDE}"
fi
if ! [[ "${TARGET_EPOCHS}" =~ ^[0-9]+$ ]]; then
  echo "[CHECK][FAIL] TARGET_EPOCHS deve ser inteiro."
  exit 94
fi
if ! [[ "${BASE_SEED}" =~ ^-?[0-9]+$ ]]; then
  echo "[CHECK][FAIL] SEED deve ser inteiro."
  exit 95
fi
if ! [[ "${RUN_START}" =~ ^[0-9]+$ && "${RUN_END}" =~ ^[0-9]+$ ]]; then
  echo "[CHECK][FAIL] RUN_START e RUN_END devem ser inteiros."
  exit 96
fi
if (( RUN_START < 0 || RUN_END > 9 || RUN_START > RUN_END )); then
  echo "[CHECK][FAIL] faixa de runs inválida (${RUN_START}-${RUN_END}). Esperado 0..9."
  exit 97
fi

TRAIN_STATIC_ARGS=(
  --model InceptionV3
  --normalize preprocess
  --batch_size 96
  --epochs "${TARGET_EPOCHS}"
  --lrate 5e-4
  --freeze_epochs 0
  --fine_tune_lr_factor 0.1
  --fine_tune_lr 5e-4
  --scheduler cosine
  --warmup_epochs 5
  --min_lr 1e-6
  --mixup_alpha 0.0
  --cutmix_alpha 0.0
  --label_smoothing 0.0
  --focal_gamma 0.0
  --pos_weight 1.0
  --fundus_crop_ratio 1.0
  --tta_views 1
  --no-freeze_bn
  --use_dali
  --dali_layout NHWC
  --channels_last
  --auc_target 0.95
)

DOCKER_GPU_ARGS=(--gpus all)

extract_batch_size() {
  local prev=""
  for token in "${TRAIN_STATIC_ARGS[@]}"; do
    if [[ "${token}" == "--batch_size" ]]; then
      prev="expect_value"
      continue
    fi
    if [[ "${prev}" == "expect_value" ]]; then
      echo "${token}"
      return 0
    fi
    if [[ "${token}" == --batch_size=* ]]; then
      echo "${token#--batch_size=}"
      return 0
    fi
    prev=""
  done
  return 1
}

TRAIN_BATCH_SIZE="$(extract_batch_size || true)"
if [[ -z "${TRAIN_BATCH_SIZE}" ]]; then
  TRAIN_BATCH_SIZE=96
fi
TRAIN_STATIC_ARGS_STR="$(printf '%s ' "${TRAIN_STATIC_ARGS[@]}")"
TRAIN_STATIC_ARGS_STR="${TRAIN_STATIC_ARGS_STR% }"

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

GPUS_PER_TASK="${SLURM_GPUS_PER_TASK:-0}"
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
TOTAL_GPUS=$(( TASK_COUNT * GPUS_PER_TASK ))
if (( TOTAL_GPUS <= 0 )); then TOTAL_GPUS=1; fi
# Ajusta com base no CUDA_VISIBLE_DEVICES, se presente
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
  if (( ${#_cvd[@]} > 0 )); then
    TOTAL_GPUS=${#_cvd[@]}
  fi
fi
echo "[CHECK][INFO] NODES=${SLURM_JOB_NUM_NODES:-?} NTASKS=${SLURM_NTASKS:-?} TASKS_PER_NODE='${SLURM_TASKS_PER_NODE:-?}' GPUS_PER_TASK=${GPUS_PER_TASK} -> TOTAL_GPUS=${TOTAL_GPUS}"
if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  echo "[CHECK][INFO] TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
else
  echo "[CHECK][INFO] TRAIN_BATCH_SIZE=<não definido nos argumentos>"
fi

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

PARTITION_TAG_RAW="${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}"
if [[ -z "${PARTITION_TAG_RAW}" && -n "${SLURM_JOB_ID:-}" ]]; then
  PARTITION_TAG_RAW="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*Partition=\([^ ]*\).*/\1/p' | head -n1)"
fi
PARTITION_TAG="$(sanitize_tag "${PARTITION_TAG_RAW:-unknown}")"

CLUSTER_TAG_RAW="${SLURM_CLUSTER_NAME:-}"
CLUSTER_TAG="$(sanitize_tag "${CLUSTER_TAG_RAW:-unknown}")"
NODEGROUP_RAW=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  FIRST_SEG="${SLURM_NODELIST%%,*}"
  FIRST_SEG="${FIRST_SEG%%[*}"
  NODEGROUP_RAW="$(printf '%s' "${FIRST_SEG}" | sed -E 's/[0-9]+$//')"
fi
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-nodes}")"

BATCH_TAG_SUFFIX=""
if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  BATCH_TAG_SUFFIX="_bs$(sanitize_tag "${TRAIN_BATCH_SIZE}")"
fi
GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)}"
GPU_TAG="${GPU_TAG:-$(sanitize_tag "${GPU_NAME_RAW:-gpu}")}"
if [[ -z "${GPU_TAG}" || "${GPU_TAG}" == "unknown" ]]; then
  GPU_TAG="gpu"
fi
GPU_DESC="${TOTAL_GPUS}x${GPU_TAG}"
BATCH_TAG="bs$(sanitize_tag "${TRAIN_BATCH_SIZE}")"
if [[ -z "${RESULTS_SUBDIR:-}" ]]; then
  RESULTS_SUBDIR="results/result${JOB_ID_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}"
fi
HOST_RESULTS_DIR="${HOST_RESULTS_DIR:-${HOST_PROJ_DIR}/${RESULTS_SUBDIR}}"
if [[ "${HOST_RESULTS_DIR}" != "${HOST_PROJ_DIR}/"* ]]; then
  echo "[CHECK][FAIL] HOST_RESULTS_DIR deve ficar dentro de ${HOST_PROJ_DIR}"
  exit 93
fi
HOME_RESULTS_DIR="${HOME_RESULTS_DIR:-${HOME_PROJ_DIR}/${RESULTS_SUBDIR}}"
if [[ "${HOME_RESULTS_DIR}" != "${HOME_PROJ_DIR}/"* ]]; then
  echo "[CHECK][FAIL] HOME_RESULTS_DIR deve ficar dentro de ${HOME_PROJ_DIR}"
  exit 93
fi
RUNS_DIR_HOST="${HOST_RESULTS_DIR}"
RUN_DIR_BASE="${HOST_RESULTS_DIR}"
HOST_LOGS_DIR="${HOST_LOGS_DIR:-${HOST_PROJ_DIR}/logs}"
HOME_LOGS_DIR="${HOME_LOGS_DIR:-${HOME_PROJ_DIR}/logs}"

export IMAGE MODULE_NAME HOST_PROJ_DIR HOST_TFREC_DIR CONT_WORKDIR HOST_UID HOST_GID HOST_RESULTS_DIR RUNS_DIR_HOST RUN_DIR_BASE RESULTS_SUBDIR TOTAL_GPUS TRAIN_BATCH_SIZE TRAIN_STATIC_ARGS_STR TARGET_EPOCHS HOST_ROOT_DIR HOME_ROOT_DIR
PROFILER_HELPER="$(hcpa_runtime_resolve_support_file "tools/hcpa_profiler_launcher.sh" "${PROFILER_HELPER:-}")"
CLEAN_TEMP_HELPER="$(hcpa_runtime_resolve_support_file "clean_temp_loggers.py" "${CLEAN_TEMP_HELPER:-}")"
if [[ ! -f "${PROFILER_HELPER}" ]]; then
  echo "[CHECK][FAIL] helper de profiling ausente: ${PROFILER_HELPER}"
  exit 91
fi
if [[ ! -f "${CLEAN_TEMP_HELPER}" ]]; then
  echo "[CHECK][FAIL] helper de limpeza ausente: ${CLEAN_TEMP_HELPER}"
  exit 91
fi
source "${PROFILER_HELPER}"
hcpa_profile_init_env
trap 'exit_code=$?; trap - EXIT; hcpa_runtime_cleanup_and_sync "${exit_code}"; exit "$?"' EXIT

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
    if [[ "${CHECK_VERBOSE}" != "0" ]]; then
      echo "[CHECK][OK] ${human_label}: ${value}"
    fi
  fi
}

check_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "${path}" ]]; then
    if [[ "${CHECK_VERBOSE}" != "0" ]]; then
      echo "[CHECK][OK] Diretório ${label}: ${path}"
    fi
  else
    echo "[CHECK][FAIL] Diretório ${label} ausente: ${path}"
    precheck_ok=false
  fi
}

if command -v docker >/dev/null 2>&1; then
  if [[ "${CHECK_VERBOSE}" != "0" ]]; then
    echo "[CHECK][OK] docker disponível no nó de submissão."
  fi
else
  echo "[CHECK][FAIL] docker indisponível no nó de submissão."
  precheck_ok=false
fi

require_var IMAGE "Imagem Docker"
require_var HOST_PROJ_DIR "Diretório do projeto no host"
require_var HOST_TFREC_DIR "Diretório de TFRecords"
require_var CONT_WORKDIR "Diretório dentro do container"

check_dir "${HOST_PROJ_DIR}" "projeto"
check_dir "${HOST_TFREC_DIR}" "TFRecords"
mkdir -p "${HOST_RESULTS_DIR}" && check_dir "${HOST_RESULTS_DIR}" "resultados (escrita)"
if [[ ! -w "${HOST_RESULTS_DIR}" ]]; then
  echo "[CHECK][FAIL] Diretório de resultados sem permissão de escrita: ${HOST_RESULTS_DIR}"
  precheck_ok=false
else
  if [[ "${CHECK_VERBOSE}" != "0" ]]; then
    echo "[CHECK][OK] Resultados escrevíveis em: ${HOST_RESULTS_DIR}"
  fi
fi
if [[ "${CHECK_VERBOSE}" != "0" ]]; then
  echo "[CHECK][OK] UID/GID locais: ${HOST_UID}:${HOST_GID}"
fi

if [[ "${precheck_ok}" != true ]]; then
  echo "[CHECK] Pré-checagem falhou; abortando job."
  exit 90
fi

echo "[CHECK] Pré-checagem local concluída com sucesso."

mkdir -p "${HOST_LOGS_DIR}"

NNODES="${SLURM_JOB_NUM_NODES:-2}"
readarray -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
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
  if [[ "${CHECK_VERBOSE}" != "0" ]]; then
    echo "[NET] ${host}: iface=${host_iface:-<none>} ip=${host_ip:-<none>}"
  fi
done
NODE_NET_INFO="${NODE_NET_INFO# }"
export NODE_NET_INFO
export MASTER_ADDR
echo "[MASTER] Rendezvous: host=${MASTER_HOST} addr=${MASTER_ADDR} iface=${MASTER_IFACE:-<none>}"

TF_PORT_BASE="${TF_PORT_BASE:-12345}"
TF_WORKER_ENDPOINTS=()
for idx in "${!SLURM_HOSTS[@]}"; do
  host="${SLURM_HOSTS[idx]}"
  endpoint_host="${host}"
  for entry in ${NODE_NET_INFO:-}; do
    IFS=":" read -r entry_host entry_iface entry_ip <<<"${entry}"
    if [[ "${entry_host}" == "${host}" && -n "${entry_ip}" ]]; then
      endpoint_host="${entry_ip}"
      break
    fi
  done
  port=$((TF_PORT_BASE + idx))
  TF_WORKER_ENDPOINTS+=("${endpoint_host}:${port}")
done
TF_WORKER_ADDRESSES="$(IFS=,; printf '%s' "${TF_WORKER_ENDPOINTS[*]}")"
export TF_WORKER_ADDRESSES TF_PORT_BASE

MASTER_PORT="${MASTER_PORT:-29500}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export MASTER_PORT MASTER_ADDR OMP_NUM_THREADS NCCL_DEBUG TF_WORKER_ADDRESSES TF_PORT_BASE

{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_job_id=${ARRAY_JOB_ID}"
  echo "array_task_id=${ARRAY_TASK_ID}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "cluster=${CLUSTER_TAG}"
  echo "partition=${PARTITION_TAG}"
  echo "persistent_project_root=${HOME_PROJ_DIR}"
  echo "execution_project_root=${HOST_PROJ_DIR}"
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
  echo "tf_workers=${TF_WORKER_ADDRESSES}"
  echo "global_batch_size=$(( TRAIN_BATCH_SIZE * TOTAL_GPUS ))"
  echo "run_start=${RUN_START}"
  echo "run_end=${RUN_END}"
  echo "seed_base=${BASE_SEED}"
  echo "target_epochs=${TARGET_EPOCHS}"
  echo "train_args=${TRAIN_STATIC_ARGS_STR}"
} > "${HOST_RESULTS_DIR}/env_manifest.txt"

srun -N ${NNODES} -n ${NNODES} --ntasks-per-node=1 --label bash <<'EOS'
set -euo pipefail

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
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[$(hostname)] Image ${IMAGE} não encontrada; iniciando docker build."
  BUILD_DIR="${HOST_PROJ_DIR}"
  [[ -d "${BUILD_DIR}" ]] || { echo "Diretório de build ausente em ${HOSTNAME}: ${BUILD_DIR}"; exit 128; }
  (
    cd "${BUILD_DIR}"
    docker build --build-arg UID="${HOST_UID}" --build-arg GID="${HOST_GID}" -t "${IMAGE}" .
  )
fi

[[ -d "${HOST_PROJ_DIR}" ]] || { echo "HOST_PROJ_DIR ausente em ${HOSTNAME}: ${HOST_PROJ_DIR}"; exit 2; }
[[ -d "${HOST_TFREC_DIR}" ]] || { echo "HOST_TFREC_DIR ausente em ${HOSTNAME}: ${HOST_TFREC_DIR}"; exit 2; }
EOS

for RUN_ID in $(seq "${RUN_START}" "${RUN_END}"); do
RUN_LABEL="$((RUN_ID - RUN_START + 1))/$((RUN_END - RUN_START + 1))"
RUN_SEED=$((BASE_SEED + RUN_ID))
echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] iniciando..."

RUN_RESULTS_DIR="${HOST_RESULTS_DIR}/run_${RUN_ID}"
if [[ -d "${RUN_RESULTS_DIR}" ]]; then
  rm -rf "${RUN_RESULTS_DIR}"
fi
mkdir -p "${RUN_RESULTS_DIR}"

export RUN_ID
export RUN_SEED
export MASTER_PORT=$((29500 + RUN_ID))
export TF_PORT_BASE=$((12345 + RUN_ID * 8))

TF_WORKER_ENDPOINTS=()
for idx in "${!SLURM_HOSTS[@]}"; do
  host="${SLURM_HOSTS[idx]}"
  endpoint_host="${host}"
  for entry in ${NODE_NET_INFO:-}; do
    IFS=":" read -r entry_host entry_iface entry_ip <<<"${entry}"
    if [[ "${entry_host}" == "${host}" && -n "${entry_ip}" ]]; then
      endpoint_host="${entry_ip}"
      break
    fi
  done
  port=$((TF_PORT_BASE + idx))
  TF_WORKER_ENDPOINTS+=("${endpoint_host}:${port}")
done
TF_WORKER_ADDRESSES="$(IFS=,; printf '%s' "${TF_WORKER_ENDPOINTS[*]}")"
export TF_WORKER_ADDRESSES

srun -N ${NNODES} -n ${NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'EOS'
set -euxo pipefail

NODE_RANK="${SLURM_NODEID}"
WORLD_SIZE="${SLURM_JOB_NUM_NODES}"
CONTAINER_NAME="hcpa_tf_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"
export NODE_RANK WORLD_SIZE CONTAINER_NAME

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
export NCCL_SOCKET_IFNAME

export TF_WORKER_ADDRESSES_LOCAL="${TF_WORKER_ADDRESSES}"
TF_CONFIG_JSON="$(python3 - <<'PY'
import json
import os

workers = [w for w in os.environ.get("TF_WORKER_ADDRESSES_LOCAL", "").split(",") if w]
rank = int(os.environ.get("NODE_RANK", "0"))
cfg = {"cluster": {"worker": workers}, "task": {"type": "worker", "index": rank}}
print(json.dumps(cfg))
PY
)"
DOCKER_TF_CONFIG_ARGS=()
if [[ "${WORLD_SIZE}" =~ ^[0-9]+$ ]] && (( WORLD_SIZE > 1 )); then
	DOCKER_TF_CONFIG_ARGS=(-e TF_CONFIG="${TF_CONFIG_JSON}")
else
	echo "[${LOCAL_NODE}] TF_CONFIG omitido para execução single-worker."
fi

DOCKER_RUNTIME_ARG=()
if [[ -n "${DOCKER_GPU_RUNTIME:-}" ]]; then
  DOCKER_RUNTIME_ARG=(--runtime "${DOCKER_GPU_RUNTIME}")
  set +e
  DOCKER_RUNTIMES="$(docker info --format '{{range .Runtimes}}{{println .Name}}{{end}}' 2>/dev/null)"
  DOCKER_INFO_STATUS=$?
  set -e
  if [[ ${DOCKER_INFO_STATUS} -eq 0 ]]; then
    if ! grep -Fxq "${DOCKER_GPU_RUNTIME}" <<<"${DOCKER_RUNTIMES}"; then
      echo "[${LOCAL_NODE}] Aviso: runtime '${DOCKER_GPU_RUNTIME}' não listado em 'docker info'; forçando uso mesmo assim." >&2
    fi
  else
    echo "[${LOCAL_NODE}] Aviso: não foi possível consultar 'docker info'; forçando uso do runtime '${DOCKER_GPU_RUNTIME}'." >&2
  fi
fi

DOCKER_USERNS_ARGS=()
if docker --version 2>/dev/null | grep -qi podman; then
  DOCKER_USERNS_ARGS=(--userns=keep-id)
fi

  docker run -i "${DOCKER_RUNTIME_ARG[@]}" --rm --name "${CONTAINER_NAME}" \
    "${DOCKER_USERNS_ARGS[@]}" \
    "${DOCKER_GPU_ARGS[@]}" \
    --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --network host \
    --user "${HOST_UID}:${HOST_GID}" \
    -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}/${MODULE_NAME}" \
    -v "${HOME_ROOT_DIR}/tools:${CONT_WORKDIR}/tools:ro" \
    -v "${HOST_TFREC_DIR}:/workspace/data" \
    -w "${CONT_WORKDIR}/${MODULE_NAME}" \
    -e HOME="/workspace" \
    -e XDG_CACHE_HOME="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/cache" \
    -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    -e NCCL_DEBUG="${NCCL_DEBUG}" \
    -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
    -e TMPDIR="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
    -e TMP="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
    -e TEMP="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
    -e MPLCONFIGDIR="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/mplconfig" \
    -e HCPA_SKIP_INTERNAL_RESULTS_CLEANUP=1 \
    -e WORLD_SIZE="${WORLD_SIZE}" \
    -e MASTER_ADDR="${MASTER_ADDR}" \
    -e MASTER_PORT="${MASTER_PORT}" \
    -e NODE_RANK="${NODE_RANK}" \
    -e TF_WORKER_ADDRESSES="${TF_WORKER_ADDRESSES}" \
    -e TF_WORKER_ADDRESSES_LOCAL="${TF_WORKER_ADDRESSES}" \
    -e TRAIN_STATIC_ARGS_STR="${TRAIN_STATIC_ARGS_STR}" \
    -e RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
	    -e RUN_ID="${RUN_ID}" \
	    -e RUN_SEED="${RUN_SEED}" \
	    -e TARGET_EPOCHS="${TARGET_EPOCHS}" \
	    "${DOCKER_TF_CONFIG_ARGS[@]}" \
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
    "${IMAGE}" \
    bash -s <<'INNERSCRIPT'
set -euxo pipefail

echo "[${HOSTNAME}] rank=${NODE_RANK}/${WORLD_SIZE} subindo"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
export TF_CPP_MIN_VLOG_LEVEL=0
export TF_ENABLE_GPU_GC=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-0}"
export DALI_LOG=ERROR
export PYTHONWARNINGS="ignore::UserWarning,ignore::RuntimeWarning"

source /workspace/tools/hcpa_profiler_launcher.sh
hcpa_profile_init_env

results_dir="./${RESULTS_SUBDIR}/run_${RUN_ID}"
local_tmp="${TMPDIR:-/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}}"
mpl_dir="${MPLCONFIGDIR:-${local_tmp}/mplconfig}"
trace_dir="${results_dir}/profiling/tf_traces"
mkdir -p "${results_dir}" "${results_dir}/profiling" "${trace_dir}" "${local_tmp}" "${mpl_dir}"
export TMPDIR="${local_tmp}"; export TMP="${TMPDIR}"; export TEMP="${TMPDIR}"
export MPLCONFIGDIR="${mpl_dir}"
export HCPA_TORCH_PROFILER_DIR="${trace_dir}"
export HCPA_RUNTIME_TRACE_DIR="${trace_dir}"
export HCPA_PROFILE_ARTIFACT_ROOT="${results_dir}/profiling"
export HCPA_PROFILE_ARTIFACT_STEM="tensorflow_opt_node${NODE_RANK}"
export HCPA_PROFILE_ARTIFACT_PREFIX="${HCPA_PROFILE_ARTIFACT_ROOT}/${HCPA_PROFILE_ARTIFACT_STEM}"
hcpa_profile_write_manifest

TF_ARGS=()
if [[ -n "${TRAIN_STATIC_ARGS_STR:-}" ]]; then
  read -r -a TF_ARGS <<< "${TRAIN_STATIC_ARGS_STR}"
fi

python -c 'import tensorflow as tf; print("Visible GPUs:", tf.config.list_physical_devices("GPU"))'

run_results="./${RESULTS_SUBDIR}/run_${RUN_ID}"
export RUN_RESULTS_DIR="${run_results}"

LAUNCH_CMD=(
  python
  dr_hcpa_v2_2024.py
  --tfrec_dir /workspace/data
  --results "${run_results}"
  --exec "${RUN_ID}"
  --seed "${RUN_SEED}"
  --cache_dir "none"
  "${TF_ARGS[@]}"
)
hcpa_profile_wrap_command "${LAUNCH_CMD[@]}"
"${PROFILED_CMD[@]}"
hcpa_profile_postprocess

# Remove caches pesados para caber na quota (25GB)
rm -rf "${run_results}/tfdata_cache" || true
rm -rf "${run_results}/keras_cache" || true

python - <<'PY'
import glob, os, sys
run_dir = os.environ.get("RUN_RESULTS_DIR") or os.environ.get("RUN_RESULTS") or os.path.join("./", os.environ.get("RESULTS_SUBDIR", ""), f"run_{os.environ.get('RUN_ID', '0')}")
target_epochs = int(os.environ.get("TARGET_EPOCHS", "200"))
pattern = os.path.join(run_dir, f"*-{os.environ.get('RUN_ID', '0')}.csv")
csv_files = [p for p in glob.glob(pattern) if os.path.isfile(p)]
if not csv_files:
    sys.exit(f"Nenhum CSV encontrado em {run_dir}")
csv_path = sorted(csv_files)[0]
with open(csv_path, "r", errors="ignore") as fh:
    rows = sum(1 for _ in fh) - 1
if rows < target_epochs:
    sys.exit(f"CSV parcial: {rows}/{target_epochs} épocas em {csv_path}")
print(f"[CHECK] {rows} épocas registradas em {csv_path}")
PY
INNERSCRIPT
EOS
echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] concluído."
done

# Limpa logs/caches volumosos, preservando CSVs/PDFs
python3 "${CLEAN_TEMP_HELPER}" --root "${RUN_DIR_BASE}" --scope results-only --max-print 0 || true

echo "Job complete."
