#!/usr/bin/env bash
###############################################################################
# Autotuner — MONAI BASE — x86 (tupi6, RTX4090)
# Usa imagem: monai_base:latest
###############################################################################

#SBATCH --job-name=autotuner_monai_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --array=1-10%1
#SBATCH --time=06:00:00
#SBATCH --partition=tupi
#SBATCH --nodelist=tupi6
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euxo pipefail

IMAGE="monai_base:latest"
DOCKER_GPU_RUNTIME="${DOCKER_GPU_RUNTIME:-nvidia}"

AUTOTUNER_DIR="${SLURM_SUBMIT_DIR}"
VARIANT_DIR="${AUTOTUNER_DIR}/../monai_base"
HOST_PROJ_DIR="$(cd "${VARIANT_DIR}" && pwd)"
HOST_TFREC_DIR="${HOST_TFREC_DIR:-${HOST_PROJ_DIR}/data/all-tfrec}"
CONT_WORKDIR="/workspace"

: "${ENABLE_TUNING:=1}"
: "${AUDIT_ONLY:=0}"
: "${MONITOR_INTERVAL:=5.0}"

HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-1}"
ARRAY_TASK_MAX="${SLURM_ARRAY_TASK_MAX:-${ARRAY_TASK_ID}}"
ARRAY_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
RUN_ID=$((ARRAY_TASK_ID-1))

GPU_NAME_RAW="${GPU_NAME_RAW:-$(srun -N1 -n1 -w "${SLURM_NODELIST}" nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || echo unknown)}"
sanitize_tag(){ printf '%s' "${1:-unknown}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/-+/-/g'; }
GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW}")"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/${GPU_TAG}_${ARRAY_JOB_ID}}"
HOST_RESULTS_DIR="${HOST_PROJ_DIR}/${RESULTS_SUBDIR}"
RUN_RESULTS_DIR="${HOST_RESULTS_DIR}/run_${RUN_ID}"
mkdir -p "${RUN_RESULTS_DIR}" "${HOST_RESULTS_DIR}" "${AUTOTUNER_DIR}/logs" "${HOST_PROJ_DIR}/logs"

export IMAGE HOST_PROJ_DIR HOST_TFREC_DIR AUTOTUNER_DIR DOCKER_GPU_RUNTIME

echo "[CHECK] image=${IMAGE} proj=${HOST_PROJ_DIR} tfrec=${HOST_TFREC_DIR} autotuner=${AUTOTUNER_DIR}"
srun -N1 -n1 --ntasks-per-node=1 --label bash <<'EOS'
set -euo pipefail
docker --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Image ${IMAGE} não encontrada; buildando..."
  cd "${HOST_PROJ_DIR}" && docker build -t "${IMAGE}" .
fi
EOS

AUTOTUNER_FLAGS=""
[[ "${ENABLE_TUNING}" == "0" ]] && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --disable-tuning"
[[ "${AUDIT_ONLY}" == "1" ]]    && AUTOTUNER_FLAGS="${AUTOTUNER_FLAGS} --audit-only"

OVERRIDES=""
OVERRIDES="${OVERRIDES} tfrec_dir=${HOST_TFREC_DIR}"
OVERRIDES="${OVERRIDES} results_dir=${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
OVERRIDES="${OVERRIDES} results=${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
OVERRIDES="${OVERRIDES} exec=${RUN_ID}"
OVERRIDES="${OVERRIDES} batch_size=96"
OVERRIDES="${OVERRIDES} model_name=inception_v3"
OVERRIDES="${OVERRIDES} normalize=inception"
OVERRIDES="${OVERRIDES} epochs=200"

RUN_LABEL="${ARRAY_TASK_ID}/${ARRAY_TASK_MAX}"
echo "============================================================"
echo "[AUTOTUNER] Stack=monai Mode=base Variant=${HOST_PROJ_DIR}"
echo "[AUTOTUNER] GPU=${GPU_TAG} Run=${RUN_ID} (${RUN_LABEL})"
echo "============================================================"

srun -N1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 --label bash <<'EOS'
set -euxo pipefail

CONTAINER_NAME="autotuner_monaibase_${SLURM_JOB_ID}_${SLURM_NODEID}_r${RUN_ID}"
CACHE_ROOT="/tmp/autotuner_monaibase_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}/cache"
LOCAL_TMP="/tmp/autotuner_monaibase_${SLURM_JOB_ID}_r${RUN_ID}_${SLURM_NODEID}"

docker run -i --rm --name "${CONTAINER_NAME}" \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:/workspace/data" \
  -v "${AUTOTUNER_DIR}:/workspace/autotuner:ro" \
  -w "${CONT_WORKDIR}" \
  -e PYTHONPATH="/workspace/autotuner" \
  -e PYTHONUNBUFFERED=1 \
  -e XDG_CACHE_HOME="${CACHE_ROOT}" \
  -e HF_HOME="${CACHE_ROOT}/hf" \
  -e TMPDIR="${LOCAL_TMP}" -e TMP="${LOCAL_TMP}" -e TEMP="${LOCAL_TMP}" \
  "${IMAGE}" \
  bash -s <<INNERSCRIPT
set -euxo pipefail
mkdir -p "${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
python -m src.main \
  --stack monai \
  --mode base \
  --path ${CONT_WORKDIR} \
  --output "${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}/autotuner_output" \
  --monitor-interval ${MONITOR_INTERVAL} \
  ${AUTOTUNER_FLAGS} \
  --override ${OVERRIDES} \
| tee "${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}/autotuner_full.log"
INNERSCRIPT
EOS

echo "Job complete. Resultados em: ${RUN_RESULTS_DIR}"
