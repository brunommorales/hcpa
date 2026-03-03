#!/usr/bin/env bash
###############################################################################
# autotuner_tf_opt_arm.slurm — Roda o Autotuner com tensorflow_opt na grace2
#
# Baseado em: tensorflow_opt/distributed_run_arm.slurm
# Diferença: substitui "python dr_hcpa_v2_2024.py ..." por
#            "python -m src.main --stack tensorflow --mode opt ..."
#
# Uso:
#   cd ~/projects/hcpa/autotuner
#   sbatch slurm/autotuner_tf_opt_arm.slurm
#
#   # Apenas auditoria (sem treinar):
#   AUDIT_ONLY=1 sbatch slurm/autotuner_tf_opt_arm.slurm
#
#   # Sem autoajuste (apenas logging/monitoramento):
#   ENABLE_TUNING=0 sbatch slurm/autotuner_tf_opt_arm.slurm
###############################################################################

#SBATCH --job-name=autotuner_tf_opt
#SBATCH --nodes=1
#SBATCH --array=1-10%1
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --nodelist=grace2
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euxo pipefail

# ========================== CONFIG ==========================
IMAGE="tf_distributed_arm:latest"
DOCKER_GPU_RUNTIME="${DOCKER_GPU_RUNTIME:-nvidia}"

# Diretórios
AUTOTUNER_DIR="$(cd "${SLURM_SUBMIT_DIR:-$(pwd)}" && pwd)"                 # ~/projects/hcpa/autotuner
# Espera layout: .../hcpa/autotuner (este script) e .../hcpa/tensorflow_opt (irmão)
TF_OPT_DIR="$(cd "${AUTOTUNER_DIR}/../tensorflow_opt" && pwd)"             # ~/projects/hcpa/tensorflow_opt
HOST_PROJ_DIR="${TF_OPT_DIR}"                                             # resolve path absoluto
HOST_TFREC_DIR="${HOST_TFREC_DIR:-${HOST_PROJ_DIR}/data/all-tfrec}"

CONT_WORKDIR="/workspace"

# Autotuner flags
: "${ENABLE_TUNING:=1}"
: "${AUDIT_ONLY:=0}"
: "${MONITOR_INTERVAL:=5.0}"

INTERNAL_IFACE="${INTERNAL_IFACE:-}"
export INTERNAL_IFACE DOCKER_GPU_RUNTIME
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_TASK_ID}}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
if ! [[ "${ARRAY_TASK_ID}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_ID=1; fi
if ! [[ "${ARRAY_TASK_MAX}" =~ ^[0-9]+$ ]]; then ARRAY_TASK_MAX="${ARRAY_TASK_ID}"; fi
RUN_ID=$((ARRAY_TASK_ID-1))

# GPU e resultados
GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || echo unknown)}"
sanitize_tag() {
  printf '%s' "${1:-unknown}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/-+/-/g'
}
GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW}")"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/${GPU_TAG}_${ARRAY_JOB_ID}}"
HOST_RESULTS_DIR="${HOST_PROJ_DIR}/${RESULTS_SUBDIR}"
RUN_RESULTS_DIR="${HOST_RESULTS_DIR}/run_${RUN_ID}"

mkdir -p "${RUN_RESULTS_DIR}" "${HOST_RESULTS_DIR}" "${AUTOTUNER_DIR}/logs" "${HOST_PROJ_DIR}/logs"

# ========================== REDE ==========================
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
      split($4, a, "/"); ip = a[1]
      if (is_private(ip)) { print $2, ip; exit }
      last_if = $2; last_ip = ip
    }
    END { if (NR > 0 && last_if != "" && last_ip != "") print last_if, last_ip }
  '
}
if [[ -n "${pref_iface}" ]]; then
  ip -o -4 addr show dev "${pref_iface}" scope global up | awk '{split($4, a, "/"); print $2, a[1]; exit}'
else
  detect_private
fi
EOS
}

NNODES="${SLURM_JOB_NUM_NODES:-1}"
readarray -t SLURM_HOSTS < <(scontrol show hostnames "$SLURM_NODELIST")
MASTER_HOST="${SLURM_HOSTS[0]}"
MASTER_ADDR="${MASTER_HOST}"
NODE_NET_INFO=""

for host in "${SLURM_HOSTS[@]}"; do
  info="$(detect_iface_ip "${host}" "${INTERNAL_IFACE}")"
  host_iface="" ; host_ip=""
  if [[ -n "${info}" ]]; then
    read -r host_iface host_ip <<<"${info}"
  fi
  if [[ "${host}" == "${MASTER_HOST}" && -n "${host_ip}" ]]; then
    MASTER_ADDR="${host_ip}"
  fi
  NODE_NET_INFO+=" ${host}:${host_iface}:${host_ip}"
done
NODE_NET_INFO="${NODE_NET_INFO# }"
export NODE_NET_INFO MASTER_ADDR

# TF_CONFIG para MultiWorkerMirroredStrategy
TF_PORT_BASE=$((12345 + RUN_ID * 8))
TF_WORKER_ENDPOINTS=()
for idx in "${!SLURM_HOSTS[@]}"; do
  host="${SLURM_HOSTS[idx]}"
  endpoint_host="${host}"
  for entry in ${NODE_NET_INFO:-}; do
    IFS=":" read -r entry_host entry_iface entry_ip <<<"${entry}"
    if [[ "${entry_host}" == "${host}" && -n "${entry_ip}" ]]; then
      endpoint_host="${entry_ip}"; break
    fi
  done
  TF_WORKER_ENDPOINTS+=("${endpoint_host}:$((TF_PORT_BASE + idx))")
done
TF_WORKER_ADDRESSES="$(IFS=,; printf '%s' "${TF_WORKER_ENDPOINTS[*]}")"
export TF_WORKER_ADDRESSES TF_PORT_BASE

MASTER_PORT="${MASTER_PORT:-$((29500 + RUN_ID))}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" MASTER_PORT OMP_NUM_THREADS

DOCKER_GPU_ARGS=(--gpus all)

# Export vars needed inside the pre-check srun; set -u there will fail otherwise.
export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR AUTOTUNER_DIR DOCKER_GPU_RUNTIME

# ========================== PRE-CHECK ==========================
echo "[CHECK] image=${IMAGE} proj=${HOST_PROJ_DIR} tfrec=${HOST_TFREC_DIR} autotuner=${AUTOTUNER_DIR}"
srun -N ${NNODES} -n ${NNODES} --ntasks-per-node=1 --label bash <<'EOS'
set -euo pipefail
echo "[$(hostname)] Docker: $(docker --version)"
echo "[$(hostname)] NVIDIA:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[$(hostname)] Image ${IMAGE} não encontrada; buildando..."
  cd "${HOST_PROJ_DIR}" && docker build -t "${IMAGE}" .
fi
EOS

# ========================== EXEC ==========================
RUN_LABEL="${ARRAY_TASK_ID}/${ARRAY_TASK_MAX}"
echo "============================================================"
echo "[AUTOTUNER] Stack=tensorflow Mode=opt Variant=${HOST_PROJ_DIR}"
echo "[AUTOTUNER] GPU=${GPU_TAG} Run=${RUN_ID} (${RUN_LABEL})"
echo "[AUTOTUNER] Autotuner dir=${AUTOTUNER_DIR}"
echo "[AUTOTUNER] Results=${RUN_RESULTS_DIR}"
echo "============================================================"

# Limpar resultado anterior desta run
[[ -d "${RUN_RESULTS_DIR}" ]] && rm -rf "${RUN_RESULTS_DIR}"
mkdir -p "${RUN_RESULTS_DIR}"

# Construir overrides e flags do autotuner
AUTOTUNER_FLAGS=""
[[ "${ENABLE_TUNING}" == "0" ]] && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --disable-tuning"
[[ "${AUDIT_ONLY}" == "1" ]]    && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --audit-only"

# Overrides baseados nos TRAIN_STATIC_ARGS do distributed_run_arm original
OVERRIDES="tfrec_dir=${HOST_TFREC_DIR}"
OVERRIDES="${OVERRIDES} results=${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
OVERRIDES="${OVERRIDES} exec=${RUN_ID}"
OVERRIDES="${OVERRIDES} cache_dir=none"
OVERRIDES="${OVERRIDES} batch_size=96"
OVERRIDES="${OVERRIDES} model=InceptionV3"
OVERRIDES="${OVERRIDES} normalize=preprocess"
OVERRIDES="${OVERRIDES} freeze_epochs=1"
OVERRIDES="${OVERRIDES} fine_tune_at=-200"
OVERRIDES="${OVERRIDES} fine_tune_lr=2e-4"
OVERRIDES="${OVERRIDES} lrate=0.003"
OVERRIDES="${OVERRIDES} mixup_alpha=0.3"
# ─── Regularisation overrides (SAFE baselines) ────────────────────────────────
# Combinação cutmix_alpha=0.6 + label_smoothing=0.1 + focal_gamma=2.0 + pos_weight=2.0
# causava NaN sistemático em ~10 épocas com LR=0.003 em 2-class desequilibrado.
# Valores seguros são definidos aqui; o autotuner pode aumentá-los progressivamente.
OVERRIDES="${OVERRIDES} cutmix_alpha=0.0"        # desabilitado; autotuner pode ativar até 0.3
OVERRIDES="${OVERRIDES} label_smoothing=0.02"    # suave; 0.1 produzia logits saturados
OVERRIDES="${OVERRIDES} focal_gamma=0.0"         # desabilitado; focal agressivo + LR alto → NaN
OVERRIDES="${OVERRIDES} pos_weight=1.0"          # neutro; deixar autotuner ajustar gradualmente
OVERRIDES="${OVERRIDES} fundus_crop_ratio=0.9"
OVERRIDES="${OVERRIDES} warmup_epochs=5"
OVERRIDES="${OVERRIDES} min_lr=1e-6"
OVERRIDES="${OVERRIDES} tta_views=2"
OVERRIDES="${OVERRIDES} recompute_backbone=true"
OVERRIDES="${OVERRIDES} jit_compile=true"
OVERRIDES="${OVERRIDES} auc_target=0.95"

export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR CONT_WORKDIR HOST_UID HOST_GID
export NODE_NET_INFO MASTER_ADDR MASTER_PORT NCCL_DEBUG OMP_NUM_THREADS
export TF_WORKER_ADDRESSES TF_PORT_BASE
export NNODES RUN_ID RESULTS_SUBDIR AUTOTUNER_DIR
export AUTOTUNER_FLAGS OVERRIDES MONITOR_INTERVAL
export DOCKER_GPU_RUNTIME

srun -N ${NNODES} -n ${NNODES} --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'EOS'
set -euxo pipefail

NODE_RANK="${SLURM_NODEID}"
WORLD_SIZE="${SLURM_JOB_NUM_NODES}"
CONTAINER_NAME="autotuner_tf_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"

# Detectar interface local
LOCAL_IFACE="${INTERNAL_IFACE:-}"
LOCAL_IP=""
LOCAL_NODE="$(hostname)"
for entry in ${NODE_NET_INFO:-}; do
  IFS=":" read -r entry_host entry_iface entry_ip <<<"${entry}"
  if [[ "${entry_host}" == "${LOCAL_NODE}" ]]; then
    [[ -z "${LOCAL_IFACE}" ]] && LOCAL_IFACE="${entry_iface}"
    LOCAL_IP="${entry_ip}"
    break
  fi
done
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${LOCAL_IFACE}}"
export NCCL_SOCKET_IFNAME

# TF_CONFIG para MultiWorkerMirroredStrategy
TF_CONFIG_JSON="$(python3 - <<'PY'
import json, os
workers = [w for w in os.environ.get("TF_WORKER_ADDRESSES", "").split(",") if w]
rank = int(os.environ.get("NODE_RANK", "0"))
cfg = {"cluster": {"worker": workers}, "task": {"type": "worker", "index": rank}}
print(json.dumps(cfg))
PY
)"

# Docker runtime
DOCKER_RUNTIME_ARG=()
if [[ -n "${DOCKER_GPU_RUNTIME:-}" ]]; then
  DOCKER_RUNTIME_ARG=(--runtime "${DOCKER_GPU_RUNTIME}")
fi

DOCKER_GPU_ARGS=(--gpus all)
CACHE_ROOT="/tmp/autotuner_tf_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/cache"
LOCAL_TMP="/tmp/autotuner_tf_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}"

echo "[$(hostname)] rank=${NODE_RANK}/${WORLD_SIZE} NCCL_IF=${NCCL_SOCKET_IFNAME} IP=${LOCAL_IP:-auto}"

docker run -i "${DOCKER_RUNTIME_ARG[@]}" --rm --name "${CONTAINER_NAME}" \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:/workspace/data" \
  -v "${AUTOTUNER_DIR}:/workspace/autotuner:ro" \
  -w "${CONT_WORKDIR}" \
  -e PYTHONPATH="/workspace/autotuner" \
  -e PYTHONUNBUFFERED=1 \
  -e OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
  -e NCCL_DEBUG="${NCCL_DEBUG}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
  -e TMPDIR="${LOCAL_TMP}" \
  -e TMP="${LOCAL_TMP}" \
  -e TEMP="${LOCAL_TMP}" \
  -e MPLCONFIGDIR="${LOCAL_TMP}/mplconfig" \
  -e XDG_CACHE_HOME="${CACHE_ROOT}" \
  -e HF_HOME="${CACHE_ROOT}/hf" \
  -e HCPA_SKIP_INTERNAL_RESULTS_CLEANUP=1 \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e NODE_RANK="${NODE_RANK}" \
  -e TF_WORKER_ADDRESSES="${TF_WORKER_ADDRESSES}" \
  -e TF_CONFIG="${TF_CONFIG_JSON}" \
  -e RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  -e RUN_ID="${RUN_ID}" \
  "${IMAGE}" \
  bash -s <<INNERSCRIPT
set -euxo pipefail

echo "[autotuner] rank=\${NODE_RANK}/\${WORLD_SIZE} subindo em \$(hostname)"

# Variáveis TF
export TF_CPP_MIN_LOG_LEVEL=2
export TF_CPP_MIN_VLOG_LEVEL=0
export TF_ENABLE_GPU_GC=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_NUM_INTEROP_THREADS=0
export DALI_LOG=ERROR
export PYTHONWARNINGS="ignore::UserWarning,ignore::RuntimeWarning"

# Dirs
local_tmp="${LOCAL_TMP}"
mkdir -p "\${local_tmp}" "\${local_tmp}/mplconfig" "${CACHE_ROOT}" "${CACHE_ROOT}/hf"
mkdir -p "${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"

python -c 'import tensorflow as tf; print("GPUs:", tf.config.list_physical_devices("GPU"))'

# ============================================================
# AQUI: autotuner em vez de dr_hcpa_v2_2024.py diretamente
# ============================================================
cd ${CONT_WORKDIR}

python -m src.main \
  --stack tensorflow \
  --mode opt \
  --path ${CONT_WORKDIR} \
  --output "${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}/autotuner_output" \
  --monitor-interval ${MONITOR_INTERVAL} \
  ${AUTOTUNER_FLAGS} \
  --override ${OVERRIDES} \
| tee "${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}/autotuner_full.log"

echo "[autotuner] Finalizado."
INNERSCRIPT
EOS

echo "[RUN ${RUN_LABEL} | run_id ${RUN_ID}] concluído."

# Limpa caches
python3 "${HOST_PROJ_DIR}/../clean_temp_loggers.py" --root "${HOST_RESULTS_DIR}" --scope results-only --max-print 0 2>/dev/null || true

echo "Job complete. Resultados em: ${RUN_RESULTS_DIR}"
