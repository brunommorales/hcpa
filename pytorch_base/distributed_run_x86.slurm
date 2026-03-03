#!/usr/bin/env bash
#SBATCH --job-name=pytorch_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 1 processo por nó (ajuste se quiser +GPUs por nó)
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --time=24:00:00
#SBATCH --partition=tupi
#SBATCH --nodelist=tupi6
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
# Imagem única (já buildada nos dois nós)
IMAGE="distributed_x86:latest"

# Projeto no HOST: usa o diretório onde você rodar o sbatch
HOST_PROJ_DIR="${SLURM_SUBMIT_DIR}"

# Usa tmp local p/ build/cache Docker (evita encher /usr/NFS)
DOCKER_TMPDIR="${DOCKER_TMPDIR:-/tmp/docker_${SLURM_JOB_ID:-$$}}"
export DOCKER_TMPDIR DOCKER_BUILDKIT=1
mkdir -p "${DOCKER_TMPDIR}"

# TFRecords no HOST (ajuste se necessário; precisa existir nos dois nós)
HOST_TFREC_DIR="${HOST_TFREC_DIR:-${HOST_PROJ_DIR}/data/all-tfrec}"

# Diretório de trabalho dentro do container
CONT_WORKDIR="/workspace"

# Interface interna (defina via export INTERNAL_IFACE=eth1 ao chamar sbatch, se necessário)
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

# Estima GPUs totais do job (não confundir com SLURM_GPUS por nó)
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
# Ajuste com CUDA_VISIBLE_DEVICES, se fornecido
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
  if (( ${#_cvd[@]} > 0 )); then
    TOTAL_GPUS=${#_cvd[@]}
  fi
fi
echo "[CHECK][INFO] NODES=${SLURM_JOB_NUM_NODES:-?} NTASKS=${SLURM_NTASKS:-?} TASKS_PER_NODE='${SLURM_TASKS_PER_NODE:-?}' GPUS_PER_TASK=${GPUS_PER_TASK} -> TOTAL_GPUS=${TOTAL_GPUS}"

# Etiquetas (sanitizadas) para identificar a execução
sanitize_tag() {
  local s="${1:-}"
  # normaliza para minúsculas
  s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]')"
  # trata casos nulos
  if [[ -z "${s}" || "${s}" == "(null)" || "${s}" == "null" ]]; then
    printf '%s' 'unknown'
    return
  fi
  # substitui quaisquer chars fora de [a-z0-9._-] por '-'
  s="$(printf '%s' "${s}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  # remove hifens repetidos
  s="$(printf '%s' "${s}" | sed -E 's/-+/-/g')"
  printf '%s' "${s}"
}

PARTITION_TAG_RAW="${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}"
if [[ -z "${PARTITION_TAG_RAW}" && -n "${SLURM_JOB_ID:-}" ]]; then
  PARTITION_TAG_RAW="$(scontrol show job "${SLURM_JOB_ID}" | sed -n 's/.*Partition=\([^ ]*\).*/\1/p' | head -n1)"
fi
PARTITION_TAG="$(sanitize_tag "${PARTITION_TAG_RAW:-unknown}")"

# Metadados adicionais (manifesto)
CLUSTER_TAG_RAW="${SLURM_CLUSTER_NAME:-}"
CLUSTER_TAG="$(sanitize_tag "${CLUSTER_TAG_RAW:-unknown}")"
NODEGROUP_RAW=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  FIRST_SEG="${SLURM_NODELIST%%,*}"
  FIRST_SEG="${FIRST_SEG%%[*}"
  NODEGROUP_RAW="$(printf '%s' "${FIRST_SEG}" | sed -E 's/[0-9]+$//')"
fi
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-nodes}")"

# Argumentos de treino (limpos, sem otimizações avançadas)
TRAIN_STATIC_ARGS=(
  --model InceptionV3
  --normalize preprocess
  --batch_size 96
  --epochs 200
  --lrate 0.003
  --freeze_epochs 3
  --fine_tune_lr_factor 0.01
)

DOCKER_GPU_ARGS=(--gpus all)
DOCKER_GPU_ARGS_STR="${DOCKER_GPU_ARGS[*]}"
export DOCKER_GPU_ARGS_STR

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
RUN_DIR_BASE="${HOST_RESULTS_DIR}"
RUNS_DIR_HOST="${HOST_RESULTS_DIR}"

TRAIN_STATIC_ARGS_STR="$(printf '%s ' "${TRAIN_STATIC_ARGS[@]}")"
TRAIN_STATIC_ARGS_STR="${TRAIN_STATIC_ARGS_STR% }"

export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR CONT_WORKDIR INTERNAL_IFACE HOST_UID HOST_GID HOST_RESULTS_DIR RUNS_DIR_HOST RUN_DIR_BASE RESULTS_SUBDIR TOTAL_GPUS
export TRAIN_STATIC_ARGS_STR TRAIN_BATCH_SIZE
# ===========================================================

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
require_var HOST_PROJ_DIR "Diretório do projeto no host"
require_var HOST_TFREC_DIR "Diretório de TFRecords"
require_var CONT_WORKDIR "Diretório dentro do container"

check_dir "${HOST_PROJ_DIR}" "projeto"
check_dir "${HOST_TFREC_DIR}" "TFRecords"
mkdir -p "${RUNS_DIR_HOST}" && check_dir "${RUNS_DIR_HOST}" "resultados (escrita)"
if [[ ! -w "${RUNS_DIR_HOST}" ]]; then
  echo "[CHECK][FAIL] Diretório de resultados sem permissão de escrita: ${RUNS_DIR_HOST}"
  precheck_ok=false
else
  echo "[CHECK][OK] Resultados escrevíveis em: ${RUNS_DIR_HOST}"
fi
echo "[CHECK][OK] UID/GID locais: ${HOST_UID}:${HOST_GID}"
if [[ -n "${TRAIN_BATCH_SIZE}" ]]; then
  echo "[CHECK][OK] Batch global definido pelos argumentos: ${TRAIN_BATCH_SIZE}"
else
  echo "[CHECK][INFO] Nenhum --batch_size nos argumentos; usando valor padrão do script de treino."
fi

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
  echo "[NET] ${host}: iface=${host_iface:-<none>} ip=${host_ip:-<none>}"
done
NODE_NET_INFO="${NODE_NET_INFO# }"
export NODE_NET_INFO
export MASTER_ADDR
echo "[MASTER] Rendezvous: host=${MASTER_HOST} addr=${MASTER_ADDR} iface=${MASTER_IFACE:-<none>}"
MASTER_PORT="${MASTER_PORT:-29500}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export MASTER_PORT MASTER_ADDR OMP_NUM_THREADS NCCL_DEBUG TORCH_DISTRIBUTED_DEBUG

  # Manifesto do ambiente no diretório de resultados (host)
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
    echo "global_batch_size=${TRAIN_BATCH_SIZE}"
    echo "run_id=${RUN_ID}"
  } > "${HOST_RESULTS_DIR}/env_manifest.txt"

# -------- Sanity checks em TODOS os nós --------
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

[[ -d "${HOST_PROJ_DIR}" ]] || { echo "HOST_PROJ_DIR ausente em ${HOSTNAME}: ${HOST_PROJ_DIR}"; exit 2; }
[[ -d "${HOST_TFREC_DIR}" ]] || { echo "HOST_TFREC_DIR ausente em ${HOSTNAME}: ${HOST_TFREC_DIR}"; exit 2; }
EOS

# -------- Repetições sequenciais (padrão: 10) --------
RUN_LABEL="${ARRAY_TASK_ID}/${ARRAY_TASK_MAX}"

echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] iniciando..."
  export RUN_ID
  # Varia porta por repetição para evitar conflito residual
  export MASTER_PORT=$((29500 + RUN_ID))
  mkdir -p "${RUNS_DIR_HOST}/run_${RUN_ID}"

  # -------- Lança 1 container por nó --------
  srun -N ${NNODES} -n ${NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'EOS'
set -euxo pipefail

DOCKER_GPU_ARGS=()
if [[ -n "${DOCKER_GPU_ARGS_STR:-}" ]]; then
  read -r -a DOCKER_GPU_ARGS <<< "${DOCKER_GPU_ARGS_STR}"
fi

NODE_RANK="${SLURM_NODEID}"
WORLD_SIZE="${SLURM_JOB_NUM_NODES}"
CONTAINER_NAME="hcpa_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"

detect_local_iface() {
  local preferred="${1:-}"
  if [[ -n "${preferred}" ]]; then
    # Tenta a interface preferida; se não existir, faz fallback para auto-detecção
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
  # Auto-detecção: escolhe IP privado se possível, senão último global
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

# Cache local e efêmero para não lotar o filesystem do usuário
CACHE_ROOT="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/cache"

if [[ -n "${NCCL_SOCKET_IFNAME}" ]]; then
  echo "[$(hostname)] NCCL interface ativa: ${NCCL_SOCKET_IFNAME} (ip=${LOCAL_IP:-<unknown>})"
else
  echo "[$(hostname)] NCCL interface não definida; rede padrão será usada."
fi

docker run --rm --name "${CONTAINER_NAME}" \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -e DOCKER_TMPDIR="${DOCKER_TMPDIR}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:/workspace/data" \
  -w "${CONT_WORKDIR}" \
  -e XDG_CACHE_HOME="${CACHE_ROOT}" \
  -e HF_HOME="${CACHE_ROOT}/hf" \
  -e TRANSFORMERS_CACHE="${CACHE_ROOT}/hf" \
  -e TORCH_HOME="${CACHE_ROOT}/torch" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
  -e TMPDIR="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
  -e TMP="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
  -e TEMP="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}" \
  -e MPLCONFIGDIR="/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/mplconfig" \
  -e TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor/${SLURM_NODEID}" \
  -e TRITON_CACHE_DIR="${CACHE_ROOT}/triton/${SLURM_NODEID}" \
  -e TRAIN_STATIC_ARGS_STR="${TRAIN_STATIC_ARGS_STR}" \
  -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e NODE_RANK="${NODE_RANK}" \
  "${IMAGE}" \
  bash -lc "
    echo \"[$(hostname)] rank=${NODE_RANK}/${WORLD_SIZE} subindo\";
      local_tmp=\"/tmp/hcpa_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}\";
      mpl_dir=\"\${local_tmp}/mplconfig\";
      export TMPDIR=\"\${local_tmp}\";
      export TMP=\"\${TMPDIR}\";
      export TEMP=\"\${TMPDIR}\";
      export MPLCONFIGDIR=\"\${mpl_dir}\";
      cache_root=\"\${local_tmp}/cache\";
      export XDG_CACHE_HOME=\"\${cache_root}\";
      export HF_HOME=\"\${cache_root}/hf\";
      export TRANSFORMERS_CACHE=\"\${cache_root}/hf\";
      export TORCH_HOME=\"\${cache_root}/torch\";
      export TORCHINDUCTOR_CACHE_DIR=\"\${cache_root}/torchinductor/${SLURM_NODEID}\";
      export TRITON_CACHE_DIR=\"\${cache_root}/triton/${SLURM_NODEID}\";
      mkdir -p \"./${RESULTS_SUBDIR}/run_${RUN_ID}\" \
               \"./${RESULTS_SUBDIR}/run_${RUN_ID}/torchelastic_logs\" \
               \"\${local_tmp}\" \
               \"\${mpl_dir}\" \
               \"\${cache_root}\" \
               \"\${HF_HOME}\" \
               \"\${TORCH_HOME}\" \
               \"\${TORCHINDUCTOR_CACHE_DIR}\" \
               \"\${TRITON_CACHE_DIR}\";
      python -c 'import torch; print(\"CUDA devices:\", torch.cuda.device_count())'
      read -r -a TRAIN_STATIC_ARGS <<< \"\${TRAIN_STATIC_ARGS_STR}\"
      torchrun \
        --nnodes=${WORLD_SIZE} --nproc_per_node=1 --node_rank=${NODE_RANK} \
        --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        --log-dir=\"./${RESULTS_SUBDIR}/run_${RUN_ID}/torchelastic_logs\" \
        dr_hcpa_v2_2024.py \
          --tfrec_dir /workspace/data \
          --results \"./${RESULTS_SUBDIR}/run_${RUN_ID}\" \
          --exec ${RUN_ID} \
          \"\${TRAIN_STATIC_ARGS[@]}\"
  "
EOS
echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] concluido."

# Limpa logs/caches volumosos, preservando CSVs/PDFs
python3 "${HOST_PROJ_DIR}/../clean_temp_loggers.py" --root "${RUN_DIR_BASE}" --scope results-only --max-print 0 || true

echo "Job complete."
