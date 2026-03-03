#!/usr/bin/env bash
###############################################################################
# autotuner_run.slurm — Template SLURM para executar o HCPA Autotuner
#
# Uso:
#   sbatch autotuner_run.slurm
#
#   # Com overrides via variáveis de ambiente:
#   STACK=pytorch MODE=opt VARIANT_DIR=/path/to/pytorch_opt sbatch autotuner_run.slurm
#   BATCH_SIZE=64 EPOCHS=100 LR=1e-4 sbatch autotuner_run.slurm
#   ENABLE_TUNING=0 sbatch autotuner_run.slurm       # só logging, sem autoajuste
#   AUDIT_ONLY=1 sbatch autotuner_run.slurm           # só auditoria, sem treinar
#
# Este script faz EXATAMENTE o que o distributed_run_x86.slurm já faz:
#   - Aloca nós e GPUs via SLURM
#   - Sobe Docker container com a imagem correta
#   - Detecta interface de rede para NCCL/Gloo
#
# A ÚNICA diferença: dentro do container, em vez de chamar
#   torchrun train.py --args...
# ele chama:
#   python -m src.main --stack X --mode Y --path /workspace ...
# que internamente lança o torchrun/train.py como subprocess.
###############################################################################

#SBATCH --job-name=autotuner
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

# ============================================================================
# CONFIGURAÇÃO — ajuste aqui ou passe via variáveis de ambiente
# ============================================================================

# Stack e modo (obrigatórios)
: "${STACK:=monai}"                   # pytorch | tensorflow | monai
: "${MODE:=opt}"                      # base | opt

# Diretório da variante no HOST (onde estão train.py, data/, etc.)
: "${VARIANT_DIR:=${SLURM_SUBMIT_DIR}/../monai_opt}"

# Diretório do autotuner no HOST
: "${AUTOTUNER_DIR:=${SLURM_SUBMIT_DIR}}"

# Imagem Docker (deve corresponder ao stack/mode escolhido)
# Adapte para a imagem correta de cada variante
case "${STACK}_${MODE}" in
  pytorch_base)  IMAGE="${IMAGE:-distributed_x86:latest}" ;;
  pytorch_opt)   IMAGE="${IMAGE:-distributed_x86:latest}" ;;
  tensorflow_*)  IMAGE="${IMAGE:-tf_distributed_x86:latest}" ;;
  monai_base)    IMAGE="${IMAGE:-monai_base:latest}" ;;
  monai_opt)     IMAGE="${IMAGE:-monai_opt:latest}" ;;
  *)             IMAGE="${IMAGE:-distributed_x86:latest}" ;;
esac

# TFRecords no HOST
: "${HOST_TFREC_DIR:=${VARIANT_DIR}/data/all-tfrec}"

# Hiperparâmetros (overrides opcionais — o autotuner usa valores do espaço derivado por padrão)
: "${BATCH_SIZE:=}"
: "${EPOCHS:=}"
: "${LR:=}"
: "${SEED:=2026}"

# Tuning
: "${ENABLE_TUNING:=1}"                # 1=habilitado, 0=desabilitado
: "${AUDIT_ONLY:=0}"                   # 1=apenas auditoria
: "${MONITOR_INTERVAL:=5.0}"           # segundos entre snapshots GPU
: "${RESUME:=0}"                       # 1=retomar de checkpoint

# Container
CONT_WORKDIR="/workspace"
CONT_AUTOTUNER="/workspace/autotuner"
CONT_TFREC_DIR="${CONT_WORKDIR}/data/all-tfrec"

# Docker
DOCKER_GPU_ARGS="--gpus all"
: "${HOST_UID:=$(id -u)}"
: "${HOST_GID:=$(id -g)}"
: "${OMP_NUM_THREADS:=8}"
: "${NCCL_DEBUG:=WARN}"
: "${TORCH_DISTRIBUTED_DEBUG:=OFF}"

# Run ID
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
RUN_ID=$((ARRAY_TASK_ID - 1))
JOB_TAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

# Resultados
: "${GPU_NAME:=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr ' /' '__' | tr -cd '[:alnum:]_' || echo 'unknown')}"
: "${RESULTS_SUBDIR:=results/${GPU_NAME}_${JOB_TAG}}"
RUN_DIR_HOST="${VARIANT_DIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
RUN_DIR_CONT="${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
mkdir -p "${RUN_DIR_HOST}" "${VARIANT_DIR}/logs"

# ============================================================================
# DETECÇÃO DE REDE (copiado dos .slurm originais)
# ============================================================================
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

# GPUs por nó
GPU_COUNT="${SLURM_GPUS_ON_NODE:-${GPU_COUNT:-}}"
if [[ -z "${GPU_COUNT:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT=${#_cvd[@]}
  else
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || echo 1)
  fi
fi
NNODES="${SLURM_NNODES:-1}"
WORLD_SIZE=$((NNODES * GPU_COUNT))

# Master node
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" 2>/dev/null | head -n1 || hostname)
INTERNAL_IFACE="${INTERNAL_IFACE:-}"
read -r _ MASTER_IP < <(detect_iface_ip "${MASTER_NODE}" "${INTERNAL_IFACE}") || MASTER_IP="${MASTER_NODE}"
MASTER_PORT=$((29500 + RUN_ID))
export MASTER_ADDR="${MASTER_IP}" MASTER_PORT

echo "============================================================"
echo "[AUTOTUNER SLURM] Stack=${STACK} Mode=${MODE}"
echo "[AUTOTUNER SLURM] Variant=${VARIANT_DIR}"
echo "[AUTOTUNER SLURM] Nodes=${NNODES} GPUs/node=${GPU_COUNT} World=${WORLD_SIZE}"
echo "[AUTOTUNER SLURM] Master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[AUTOTUNER SLURM] Docker image=${IMAGE}"
echo "[AUTOTUNER SLURM] Run ID=${RUN_ID} (array ${ARRAY_TASK_ID})"
echo "============================================================"

# ============================================================================
# MONTAR OVERRIDES
# ============================================================================
OVERRIDES=""
# Só adiciona overrides para variáveis que foram EXPLICITAMENTE setadas
[[ -n "${BATCH_SIZE}" ]] && OVERRIDES="${OVERRIDES} batch_size=${BATCH_SIZE}"
[[ -n "${EPOCHS}" ]]     && OVERRIDES="${OVERRIDES} epochs=${EPOCHS}"
[[ -n "${LR}" ]]         && OVERRIDES="${OVERRIDES} learning_rate=${LR}"
# Paths sempre são passados
OVERRIDES="${OVERRIDES} tfrec_dir=${CONT_TFREC_DIR}"
OVERRIDES="${OVERRIDES} results_dir=${RUN_DIR_CONT}"
OVERRIDES="${OVERRIDES} results=${RUN_DIR_CONT}"
OVERRIDES="${OVERRIDES} seed=${SEED}"

# Flags do autotuner
AUTOTUNER_FLAGS=""
[[ "${ENABLE_TUNING}" == "0" ]] && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --disable-tuning"
[[ "${AUDIT_ONLY}" == "1" ]]    && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --audit-only"
[[ "${RESUME}" == "1" ]]        && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --resume"

# ============================================================================
# EXECUÇÃO — srun + Docker + Autotuner
# ============================================================================
export DOCKER_GPU_ARGS_STR="${DOCKER_GPU_ARGS}"
export VARIANT_DIR HOST_TFREC_DIR AUTOTUNER_DIR CONT_WORKDIR
export STACK MODE IMAGE NNODES GPU_COUNT WORLD_SIZE
export INTERNAL_IFACE HOST_UID HOST_GID OMP_NUM_THREADS
export NCCL_DEBUG TORCH_DISTRIBUTED_DEBUG
export RUN_ID RUN_DIR_CONT RESULTS_SUBDIR JOB_TAG
export OVERRIDES AUTOTUNER_FLAGS MONITOR_INTERVAL

echo "[RUN ${ARRAY_TASK_ID}] Iniciando..."

srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'SRUNEOF'
set -euxo pipefail

NODE_RANK="${SLURM_NODEID}"
CONTAINER_NAME="autotuner_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"

# Detecta interface local (mesmo código das variantes originais)
detect_local_iface() {
  local preferred="${1:-}"
  if [[ -n "${preferred}" ]]; then
    local out=""
    out=$(ip -o -4 addr show dev "${preferred}" scope global up 2>/dev/null \
          | awk '{split($4, a, "/"); print $2, a[1]; exit}') || true
    [[ -n "${out}" ]] && { echo "${out}"; return 0; }
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
      split($4, a, "/"); ip = a[1]
      if (is_private(ip)) { print $2, ip; exit }
      last_if = $2; last_ip = ip
    }
    END { if (NR > 0 && last_if != "" && last_ip != "") print last_if, last_ip }
  '
}

LOCAL_IFACE="${INTERNAL_IFACE:-}"
LOCAL_IP=""
DETECT_OUTPUT="$(detect_local_iface "${LOCAL_IFACE}")"
if [[ -n "${DETECT_OUTPUT}" ]]; then
  read -r detected_iface detected_ip <<<"${DETECT_OUTPUT}"
  [[ -z "${LOCAL_IFACE}" ]] && LOCAL_IFACE="${detected_iface}"
  [[ -z "${LOCAL_IP}" ]]    && LOCAL_IP="${detected_ip}"
fi
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${LOCAL_IFACE}}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${LOCAL_IFACE}}"
export NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME

CACHE_ROOT="/tmp/autotuner_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/cache"

DOCKER_GPU_ARGS=()
if [[ -n "${DOCKER_GPU_ARGS_STR:-}" ]]; then
  read -r -a DOCKER_GPU_ARGS <<< "${DOCKER_GPU_ARGS_STR}"
fi

echo "[$(hostname)] rank=${NODE_RANK}/${NNODES} NCCL_IF=${NCCL_SOCKET_IFNAME} IP=${LOCAL_IP:-auto}"

docker run --rm --name "${CONTAINER_NAME}" \
  "${DOCKER_GPU_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${VARIANT_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:${CONT_WORKDIR}/data/all-tfrec" \
  -v "${AUTOTUNER_DIR}:${CONT_WORKDIR}/autotuner:ro" \
  -w "${CONT_WORKDIR}" \
  -e HOME="${CONT_WORKDIR}" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="${CONT_WORKDIR}/autotuner" \
  -e XDG_CACHE_HOME="${CACHE_ROOT}" \
  -e MPLCONFIGDIR="${CACHE_ROOT}/mpl" \
  -e HF_HOME="${CACHE_ROOT}/hf" \
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
    echo '[autotuner] Iniciando dentro do container';
    echo '[autotuner] CUDA devices:';
    python -c 'import torch; print(\"  GPUs:\", torch.cuda.device_count())' 2>/dev/null || \
    python -c 'import tensorflow as tf; print(\"  GPUs:\", len(tf.config.list_physical_devices(\"GPU\")))' 2>/dev/null || \
    echo '  (detecção via nvidia-smi será feita pelo autotuner)';

    local_tmp='/tmp/autotuner_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}';
    mkdir -p \"\${local_tmp}\" '${CACHE_ROOT}' '${CACHE_ROOT}/hf' '${CACHE_ROOT}/torch';
    mkdir -p '${RUN_DIR_CONT}' '${RUN_DIR_CONT}/autotuner_output';

    # ======================================================================
    # AQUI ESTÁ A MUDANÇA: em vez de torchrun direto, chamamos o autotuner
    # ======================================================================
    cd ${CONT_WORKDIR};
    python -m src.main \
      --stack ${STACK} \
      --mode ${MODE} \
      --path ${CONT_WORKDIR} \
      --output '${RUN_DIR_CONT}/autotuner_output' \
      --monitor-interval ${MONITOR_INTERVAL} \
      ${AUTOTUNER_FLAGS} \
      --override ${OVERRIDES} \
    | tee '${RUN_DIR_CONT}/autotuner_full.log';

    echo '[autotuner] Finalizado.';
  "
SRUNEOF

echo "[RUN ${ARRAY_TASK_ID}] Concluído."
echo "Resultados em: ${RUN_DIR_HOST}/autotuner_output/"
echo "Job complete."
