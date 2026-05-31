#!/bin/bash
set -euo pipefail

# Caminhos absolutos
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${ROOT}/monai_opt.sif"
TFREC_DIR_HOST="${ROOT}/data/all-tfrec"

# Configs
BATCH_SIZE=96
EPOCHS=200
LR=3e-4
IMG_SIZE=299
MODEL="efficientnet_b3"
NUM_RUNS=10
USE_DALI_FLAG="--use_dali"   # remova se a imagem não tiver DALI

sanitize_tag() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//' | cut -c1-80
}

JOB_ID_TAG="$(sanitize_tag "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}")"
NODE_NAME_RAW="${SLURM_NODELIST:-$(hostname)}"
NODEGROUP_RAW="${NODE_NAME_RAW%%,*}"
NODEGROUP_RAW="${NODEGROUP_RAW%%[*}"
NODEGROUP_RAW="$(printf '%s' "${NODEGROUP_RAW}" | sed -E 's/[0-9]+$//')"
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-nodes}")"
GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)}"
GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW:-gpu}")"
if [[ -z "${GPU_TAG}" || "${GPU_TAG}" == "unknown" ]]; then
  GPU_TAG="gpu"
fi
GPU_COUNT="${SLURM_GPUS_PER_TASK:-${CUDA_VISIBLE_DEVICES:+$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}}"
GPU_COUNT="${GPU_COUNT:-1}"
GPU_DESC="${GPU_COUNT}x${GPU_TAG}"
BATCH_TAG="bs$(sanitize_tag "${BATCH_SIZE}")"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results/result${JOB_ID_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}}"
BASE_RESULTS_HOST="${ROOT}/${RESULTS_SUBDIR}"

for i in $(seq 0 $((NUM_RUNS-1)))
do
  RUN_DIR_HOST="${BASE_RESULTS_HOST}/run_${i}"
  RUN_DIR_CONT="/workspace/${RESULTS_SUBDIR}/run_${i}"
  TFREC_DIR_CONT="/workspace/data/all-tfrec"

  echo "======================================"
  echo "Iniciando treinamento: run_${i}"
  echo "Salvando em: ${RUN_DIR_HOST}"
  echo "======================================"

  mkdir -p "${RUN_DIR_HOST}"

  singularity exec --nv \
    -B "${TFREC_DIR_HOST}:${TFREC_DIR_CONT}" \
    -B "${RUN_DIR_HOST}:${RUN_DIR_CONT}" \
    -B "${ROOT}:/workspace" \
    "${CONTAINER}" \
    bash -lc "cd /workspace && \
      python3 train.py \
        --tfrec_dir ${TFREC_DIR_CONT} \
        --results ${RUN_DIR_CONT} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --image_size ${IMG_SIZE} \
        --model ${MODEL} \
        --log_every 50 \
        ${USE_DALI_FLAG}"

  echo "Treinamento run_${i} finalizado."
done

echo "Todos os treinamentos foram concluídos."
