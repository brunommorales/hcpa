#!/bin/bash
set -euo pipefail

# Caminhos absolutos
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTAINER="${ROOT}/monai_opt.sif"
TFREC_DIR_HOST="${ROOT}/data/all-tfrec"
BASE_RESULTS_HOST="${ROOT}/results/h200_opt"

# Configs
BATCH_SIZE=96
EPOCHS=200
LR=3e-4
IMG_SIZE=299
MODEL="efficientnet_b3"
NUM_RUNS=10
USE_DALI_FLAG="--use_dali"   # remova se a imagem não tiver DALI

for i in $(seq 0 $((NUM_RUNS-1)))
do
  RUN_DIR_HOST="${BASE_RESULTS_HOST}/run_${i}"
  RUN_DIR_CONT="/workspace/results/h200_opt/run_${i}"
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
