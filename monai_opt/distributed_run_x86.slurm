#!/usr/bin/env bash
#SBATCH --job-name=monai_opt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --time=24:00:00
#SBATCH --partition=tupi
#SBATCH --nodelist=tupi4
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euxo pipefail

sanitize_tag() {
  local s="${1:-}"
  s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "${s}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  s="$(printf '%s' "${s}" | sed -E 's/-+/-/g' | sed -E 's/^[-.]+|[-.]+$//')"
  if [[ -z "${s}" ]]; then
    printf '%s' "unknown"
  else
    printf '%s' "${s}"
  fi
}

# --- Ajustáveis ---
: "${IMAGE:=monai_opt:latest}"
: "${BATCH_SIZE:=96}"
: "${EPOCHS:=200}"
: "${LR:=3e-4}"
: "${WEIGHT_DECAY:=1e-4}"
: "${IMAGE_SIZE:=299}"
: "${MODEL:=inception_v3}"
: "${NORMALIZE:=inception}"
: "${FUNDUS_CROP:=0.9}"
: "${AUGMENT:=1}"
: "${MIXUP:=0.2}"
: "${CUTMIX:=0.0}"
: "${LABEL_SMOOTH:=0.02}"
: "${EMA_DECAY:=0.999}"
: "${GRAD_CLIP:=1.0}"
: "${GRAD_ACC:=1}"
: "${SEED:=2026}"
: "${USE_DALI:=1}"
: "${USE_COMPILE:=0}"
: "${RUN_EVAL:=1}"
: "${RUN_BENCHMARK:=0}"
: "${BENCH_WARMUP:=20}"
: "${BENCH_STEPS:=200}"

: "${RESULTS_ROOT:=results}"
: "${GPU_NAME_RAW:=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo gpu)}"
GPU_NAME="${GPU_NAME_RAW}"

# GPUs por nó via Slurm; fallback CUDA_VISIBLE_DEVICES; último recurso nvidia-smi
GPU_COUNT="${SLURM_GPUS_ON_NODE:-${GPU_COUNT:-}}"
if [[ -z "${GPU_COUNT:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT=${#_cvd[@]}
  else
    GPU_COUNT=$(nvidia-smi -L | grep -c '^GPU' || true)
  fi
fi
if ! [[ "${GPU_COUNT:-}" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 )); then
  GPU_COUNT=1
fi
NNODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
WORLD_SIZE=$((NNODES * GPU_COUNT))
: "${JOB_TAG:=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"

GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW}")"
if [[ -z "${GPU_TAG}" ]]; then GPU_TAG="gpu"; fi
GPU_DESC="${GPU_COUNT}x${GPU_TAG}"
NODEGROUP_RAW=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  FIRST_SEG="${SLURM_NODELIST%%,*}"
  FIRST_SEG="${FIRST_SEG%%[*}"
  NODEGROUP_RAW="$(printf '%s' "${FIRST_SEG}" | sed -E 's/[0-9]+$//')"
fi
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-node}")"
BATCH_TAG="bs$(sanitize_tag "${BATCH_SIZE}")"
if [[ -z "${RESULTS_SUBDIR:-}" ]]; then
  RESULTS_SUBDIR="${RESULTS_ROOT}/result${JOB_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}"
fi

: "${APP_DIR:=${SLURM_SUBMIT_DIR}}"
HOST_PROJ_DIR="${HOST_PROJ_DIR:-${APP_DIR}}"
CONT_WORKDIR="/workspace"
: "${TFREC_DIR:=${HOST_PROJ_DIR}/data/all-tfrec}"
HOST_TFREC_DIR="${TFREC_DIR}"
CONT_TFREC_DIR="${CONT_WORKDIR}/data/all-tfrec"
: "${RUN_ID:=$(( (${SLURM_ARRAY_TASK_ID:-1}-1) ))}"
if (( RUN_ID < 0 )); then RUN_ID=0; fi
RUN_DIR_BASE="${HOST_PROJ_DIR}/${RESULTS_SUBDIR}"
RUN_DIR_HOST="${RUN_DIR_BASE}/run_${RUN_ID}"
RUN_DIR_CONT="${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
mkdir -p "${RUN_DIR_HOST}" "${RUN_DIR_BASE}" "${HOST_PROJ_DIR}/logs"
: "${HOST_UID:=$(id -u)}"
: "${HOST_GID:=$(id -g)}"

DOCKER_GPU_ARGS="--gpus all"
CACHE_DIR="${CONT_WORKDIR}/.cache"
MPL_DIR="${CACHE_DIR}/matplotlib"
HF_CACHE="${CACHE_DIR}/huggingface"
TORCH_CACHE="${CACHE_DIR}/torch"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Resolução do MASTER_ADDR com suporte a detecção de interface em cada nó (similar ao pytorch_opt)
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

INTERNAL_IFACE="${INTERNAL_IFACE:-}"
export INTERNAL_IFACE

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
export NODE_NET_INFO MASTER_ADDR

echo "[MASTER] Rendezvous: host=${MASTER_HOST} addr=${MASTER_ADDR} iface=${MASTER_IFACE:-<none>}"
MASTER_PORT="${MASTER_PORT:-29500}"

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"
export MASTER_PORT MASTER_ADDR OMP_NUM_THREADS NCCL_DEBUG TORCH_DISTRIBUTED_DEBUG

echo "[monai_opt][${SLURM_JOB_ID:-manual}] RUN_ID=${RUN_ID} BATCH=${BATCH_SIZE} EPOCHS=${EPOCHS} MODEL=${MODEL} DALI=${USE_DALI} COMPILE=${USE_COMPILE} GPU=${GPU_NAME} JOB=${JOB_TAG} NNODES=${NNODES} GPUS_PER_NODE=${GPU_COUNT} WORLD_SIZE=${WORLD_SIZE}"

# Export variables for srun heredoc
export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR CONT_WORKDIR CONT_TFREC_DIR HOST_UID HOST_GID
export CACHE_DIR MPL_DIR HF_CACHE TORCH_CACHE RESULTS_SUBDIR RUN_ID RUN_DIR_CONT
export BATCH_SIZE EPOCHS LR WEIGHT_DECAY IMAGE_SIZE MODEL NORMALIZE FUNDUS_CROP
export MIXUP CUTMIX LABEL_SMOOTH EMA_DECAY GRAD_CLIP GRAD_ACC SEED
export USE_DALI USE_COMPILE AUGMENT RUN_EVAL RUN_BENCHMARK BENCH_WARMUP BENCH_STEPS
export DOCKER_GPU_ARGS NNODES GPU_COUNT JOB_TAG

# -------- Lança 1 container por nó (similar ao pytorch_opt) --------
srun -N ${NNODES} -n ${NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'EOS'
set -euxo pipefail

# Cada nó expande suas próprias variáveis SLURM aqui!
NODE_RANK="${SLURM_NODEID}"

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

LOCAL_IFACE="${INTERNAL_IFACE:-}"
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

if [[ -n "${NCCL_SOCKET_IFNAME}" ]]; then
  echo "[$(hostname)] NCCL interface ativa: ${NCCL_SOCKET_IFNAME} (ip=${LOCAL_IP:-<unknown>})"
else
  echo "[$(hostname)] NCCL interface não definida; rede padrão será usada."
fi

CONTAINER_NAME="monai_opt_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"
CACHE_ROOT="/tmp/monai_opt_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/cache"

docker run --rm --name "${CONTAINER_NAME}" \
  ${DOCKER_GPU_ARGS} \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:${CONT_TFREC_DIR}" \
  -w "${CONT_WORKDIR}" \
  -e HOME="${CONT_WORKDIR}" \
  -e PYTHONUNBUFFERED=1 \
  -e XDG_CACHE_HOME="${CACHE_ROOT}" \
  -e MPLCONFIGDIR="${CACHE_ROOT}/mpl" \
  -e HF_HOME="${CACHE_ROOT}/hf" \
  -e HF_HUB_CACHE="${CACHE_ROOT}/hf" \
  -e TRANSFORMERS_CACHE="${CACHE_ROOT}/hf" \
  -e TORCH_HOME="${CACHE_ROOT}/torch" \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
  -e WORLD_SIZE="${NNODES}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e NODE_RANK="${NODE_RANK}" \
  "${IMAGE}" \
  bash -lc "
    echo \"[$(hostname)] rank=${NODE_RANK}/${NNODES} subindo\";
    local_tmp=\"/tmp/monai_opt_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}\";
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
    mkdir -p \"\${local_tmp}\" \"\${mpl_dir}\" \"\${cache_root}\" \"\${cache_root}/hf\" \"\${cache_root}/torch\";
    mkdir -p '${RUN_DIR_CONT}' '${RUN_DIR_CONT}/tmp' '${RUN_DIR_CONT}/torchelastic_logs';

    python -c 'import torch; print(\"CUDA devices:\", torch.cuda.device_count())'

    # Detecta GPUs disponíveis
    if [[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ]]; then
      IFS=',' read -ra _cvd <<<\"\${CUDA_VISIBLE_DEVICES}\";
      NPROC_PER_NODE=\${#_cvd[@]};
    else
      NPROC_PER_NODE=\$(nvidia-smi -L | grep -c '^GPU' || echo 1);
    fi;

    echo \"[monai_opt] NODE_RANK=${NODE_RANK} NNODES=${NNODES} NPROC_PER_NODE=\${NPROC_PER_NODE} MASTER=${MASTER_ADDR}:${MASTER_PORT}\";

    torchrun \\
      --nnodes=${NNODES} \\
      --nproc_per_node=\${NPROC_PER_NODE} \\
      --node_rank=${NODE_RANK} \\
      --rdzv_backend=c10d \\
      --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \\
      --rdzv_id=${JOB_TAG} \\
      --log-dir='${RUN_DIR_CONT}/torchelastic_logs' \\
      --max_restarts=0 \\
      --monitor_interval=30 \\
      train.py \\
        --results '${RUN_DIR_CONT}' \\
        --tfrec_dir '${CONT_TFREC_DIR}' \\
        --epochs ${EPOCHS} \\
        --batch_size ${BATCH_SIZE} \\
        --learning_rate ${LR} \\
        --weight_decay ${WEIGHT_DECAY} \\
        --image_size ${IMAGE_SIZE} \\
        --model ${MODEL} \\
        --normalize ${NORMALIZE} \\
        --fundus_crop_ratio ${FUNDUS_CROP} \\
        --mixup_alpha ${MIXUP} \\
        --cutmix_alpha ${CUTMIX} \\
        --label_smoothing ${LABEL_SMOOTH} \\
        --ema_decay ${EMA_DECAY} \\
        --grad_clip_norm ${GRAD_CLIP} \\
        --gradient_accumulation ${GRAD_ACC} \\
        --channels_last \\
        --amp \\
        \$([ ${USE_DALI} -eq 1 ] && echo --use_dali || echo --no_dali) \\
        \$([ ${USE_COMPILE} -eq 1 ] || echo --no_compile) \\
        \$([ ${AUGMENT} -eq 1 ] && echo --augment || echo --no_augment) \\
        --seed ${SEED} \\
    | tee '${RUN_DIR_CONT}/train.log';

    if [ ${RUN_EVAL} -eq 1 ]; then
      python3 -u eval.py --results '${RUN_DIR_CONT}' | tee -a '${RUN_DIR_CONT}/train.log';
    fi;

    if [ ${RUN_BENCHMARK} -eq 1 ]; then
      python3 -u benchmark.py \\
        --results '${RUN_DIR_CONT}/benchmark' \\
        --tfrec_dir '${CONT_TFREC_DIR}' \\
        --batch_size ${BATCH_SIZE} \\
        --image_size ${IMAGE_SIZE} \\
        --model ${MODEL} \\
        \$([ ${USE_DALI} -eq 1 ] && echo --use_dali || echo --no_dali) \\
        --warmup_steps ${BENCH_WARMUP} \\
        --measure_steps ${BENCH_STEPS} \\
      | tee -a '${RUN_DIR_CONT}/train.log';
    fi
  "
EOS

echo "[monai_opt] Finalizado run_${RUN_ID}"
