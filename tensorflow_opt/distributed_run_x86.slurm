#!/usr/bin/env bash
#SBATCH --job-name=tensorflow_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --time=24:00:00
#SBATCH --partition=poti
#SBATCH --nodelist=poti3
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
IMAGE="tf_distributed_x86:latest"

HOST_PROJ_DIR="${SLURM_SUBMIT_DIR}"
HOST_TFREC_DIR="${HOST_TFREC_DIR:-${HOST_PROJ_DIR}/data/all-tfrec}"
CONT_WORKDIR="/workspace"

# Usa tmp local para build/cache Docker (evita encher /usr/NFS)
DOCKER_TMPDIR="${DOCKER_TMPDIR:-/tmp/docker_${SLURM_JOB_ID:-$$}}"
export DOCKER_TMPDIR DOCKER_BUILDKIT=1
mkdir -p "${DOCKER_TMPDIR}"

INTERNAL_IFACE="${INTERNAL_IFACE:-}"
export INTERNAL_IFACE
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_TASK_ID}}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_ID=1; fi
if ! [[ "${ARRAY_TASK_MAX}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_MAX="${ARRAY_TASK_ID}"; fi
RUN_ID=$((ARRAY_TASK_ID-1))
JOB_ID_TAG="${ARRAY_JOB_ID}"
CHECK_VERBOSE="${HCPA_PRECHECK_VERBOSE:-0}"
TARGET_EPOCHS="${TARGET_EPOCHS:-200}"

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
TOTAL_GPUS=$(( TASK_COUNT * GPUS_PER_TASK ))
if (( TOTAL_GPUS <= 0 )); then TOTAL_GPUS=1; fi
# Ajuste com base no CUDA_VISIBLE_DEVICES, se fornecido
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
  if (( ${#_cvd[@]} > 0 )); then
    TOTAL_GPUS=${#_cvd[@]}
  fi
fi
echo "[CHECK][INFO] NODES=${SLURM_JOB_NUM_NODES:-?} NTASKS=${SLURM_NTASKS:-?} TASKS_PER_NODE='${SLURM_TASKS_PER_NODE:-?}' GPUS_PER_TASK=${GPUS_PER_TASK} -> TOTAL_GPUS=${TOTAL_GPUS}"

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

TRAIN_STATIC_ARGS=(
  --model InceptionV3
  --normalize preprocess
  --batch_size 192
  --epochs 200
  --fine_tune_lr 3e-4
  --scheduler cosine
  --warmup_epochs 5
  --min_lr 1e-6
  --mixup_alpha 0.4
  --cutmix_alpha 0.6
  --label_smoothing 0.1
  --focal_gamma 2.0
  --pos_weight 2.0
  --fundus_crop_ratio 0.9
  --fine_tune_schedule 140:-128:unfreeze_bn,170:0:unfreeze_bn
  --use_dali
  --dali_threads 6
  --dali_layout NHWC
  --tta_views 2
  --no-freeze_bn
  --channels_last
  --h2d_uint8
  --recompute_backbone
  --jit_compile
  --auc_target 0.95
)

DOCKER_GPU_ARGS=(--gpus all)
DOCKER_GPU_ARGS_STR="${DOCKER_GPU_ARGS[*]}"

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
RUNS_DIR_HOST="${HOST_RESULTS_DIR}"
RUN_DIR_BASE="${HOST_RESULTS_DIR}"
TRAIN_STATIC_ARGS_STR="$(printf '%s ' "${TRAIN_STATIC_ARGS[@]}")"
TRAIN_STATIC_ARGS_STR="${TRAIN_STATIC_ARGS_STR% }"

export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR CONT_WORKDIR HOST_UID HOST_GID HOST_RESULTS_DIR RUNS_DIR_HOST RUN_DIR_BASE RESULTS_SUBDIR TOTAL_GPUS DOCKER_GPU_ARGS_STR
export TRAIN_STATIC_ARGS_STR TRAIN_BATCH_SIZE TARGET_EPOCHS


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
  [[ "${CHECK_VERBOSE}" != "0" ]] && echo "[CHECK][OK] docker disponível no nó de submissão."
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
  [[ "${CHECK_VERBOSE}" != "0" ]] && echo "[CHECK][OK] Resultados escrevíveis em: ${HOST_RESULTS_DIR}"
fi
[[ "${CHECK_VERBOSE}" != "0" ]] && echo "[CHECK][OK] UID/GID locais: ${HOST_UID}:${HOST_GID}"

if [[ "${precheck_ok}" != true ]]; then
  echo "[CHECK] Pré-checagem falhou; abortando job."
  exit 90
fi

echo "[CHECK] Pré-checagem local concluída com sucesso."

mkdir -p logs

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
  [[ "${CHECK_VERBOSE}" != "0" ]] && echo "[NET] ${host}: iface=${host_iface:-<none>} ip=${host_ip:-<none>}"
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
  echo "nodelist=${SLURM_NODELIST:-}"
  echo "nodes_resolved=${SLURM_HOSTS[*]}"
  echo "nodegroup=${NODEGROUP_TAG}"
  echo "image=${IMAGE}"
  echo "world_size=${TOTAL_GPUS}"
  echo "master_addr=${MASTER_ADDR}"
  echo "master_iface=${MASTER_IFACE}"
  echo "node_net_info=${NODE_NET_INFO}"
  echo "tf_workers=${TF_WORKER_ADDRESSES}"
  echo "global_batch_size=${TRAIN_BATCH_SIZE}"
  echo "run_id=${RUN_ID}"
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
    docker build -t "${IMAGE}" .
  )
fi

# Reconstrói argumentos de GPU, pois arrays não atravessam boundary do srun
DOCKER_GPU_ARGS=()
if [[ -n "${DOCKER_GPU_ARGS_STR:-}" ]]; then
  read -r -a DOCKER_GPU_ARGS <<< "${DOCKER_GPU_ARGS_STR}"
fi

[[ -d "${HOST_PROJ_DIR}" ]] || { echo "HOST_PROJ_DIR ausente em ${HOSTNAME}: ${HOST_PROJ_DIR}"; exit 2; }
[[ -d "${HOST_TFREC_DIR}" ]] || { echo "HOST_TFREC_DIR ausente em ${HOSTNAME}: ${HOST_TFREC_DIR}"; exit 2; }
EOS

RUN_LABEL="${ARRAY_TASK_ID}/${ARRAY_TASK_MAX}"
echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] iniciando..."
export RUN_ID
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

DOCKER_GPU_ARGS=()
if [[ -n "${DOCKER_GPU_ARGS_STR:-}" ]]; then
  read -r -a DOCKER_GPU_ARGS <<< "${DOCKER_GPU_ARGS_STR}"
fi

docker run -i --rm --name "${CONTAINER_NAME}" \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -e DOCKER_TMPDIR="${DOCKER_TMPDIR}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:/workspace/data" \
  -w "${CONT_WORKDIR}" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e TMPDIR="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
  -e TMP="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
  -e TEMP="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
  -e MPLCONFIGDIR="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/mplconfig" \
  -e KERAS_HOME="/workspace/${RESULTS_SUBDIR}/run_${RUN_ID}/keras_cache" \
  -e TRAIN_STATIC_ARGS_STR="${TRAIN_STATIC_ARGS_STR}" \
  -e RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  -e RUN_ID="${RUN_ID}" \
  -e TARGET_EPOCHS="${TARGET_EPOCHS}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e NODE_RANK="${NODE_RANK}" \
    -e TF_WORKER_ADDRESSES="${TF_WORKER_ADDRESSES}" \
    -e TF_WORKER_ADDRESSES_LOCAL="${TF_WORKER_ADDRESSES}" \
    -e TF_CONFIG="${TF_CONFIG_JSON}" \
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

results_dir="./${RESULTS_SUBDIR}/run_${RUN_ID}"
export RUN_RESULTS_DIR="${results_dir}"
local_tmp="${TMPDIR:-/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}}"
mpl_dir="${MPLCONFIGDIR:-${local_tmp}/mplconfig}"
keras_dir="${KERAS_HOME:-/workspace/${RESULTS_SUBDIR}/run_${RUN_ID}/keras_cache}"
mkdir -p "${results_dir}" "${local_tmp}" "${mpl_dir}" "${keras_dir}"
export TMPDIR="${local_tmp}"; export TMP="${TMPDIR}"; export TEMP="${TMPDIR}"
export MPLCONFIGDIR="${mpl_dir}"
export KERAS_HOME="${keras_dir}"

TRAIN_ARGS=()
if [[ -n "${TRAIN_STATIC_ARGS_STR:-}" ]]; then
  read -r -a TRAIN_ARGS <<< "${TRAIN_STATIC_ARGS_STR}"
fi

python -c 'import tensorflow as tf; print("Visible GPUs:", tf.config.list_physical_devices("GPU"))'

python dr_hcpa_v2_2024.py \
  --tfrec_dir /workspace/data \
  --results "./${RESULTS_SUBDIR}/run_${RUN_ID}" \
  --exec "${RUN_ID}" \
  --cache_dir "none" \
  "${TRAIN_ARGS[@]}"

# Remove caches pesados para caber na quota (25GB)
rm -rf "${results_dir}/tfdata_cache" || true
rm -rf "${results_dir}/keras_cache" || true

python - <<'PY'
import glob, os, sys
run_dir = os.environ.get("RUN_RESULTS_DIR") or os.path.join("./", os.environ.get("RESULTS_SUBDIR", ""), f"run_{os.environ.get('RUN_ID', '0')}")
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

# Limpa logs/caches volumosos, preservando CSVs/PDFs
python3 "${HOST_PROJ_DIR}/../clean_temp_loggers.py" --root "${RUN_DIR_BASE}" --scope results-only --max-print 0 || true

echo "Job complete."
