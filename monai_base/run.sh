#!/bin/bash
set -euo pipefail

# =============================================================================
# run.sh - Executa 10 treinamentos baseline (monai_base) no Grid com Apptainer
# =============================================================================

# Container Apptainer/Singularity
CONTAINER="${CONTAINER:-./hcpa.sif}"

# Caminhos host/contêiner
HOST_DIR="$(cd "$(dirname "$0")" && pwd)"
CONT_DIR="/workspace"

TFREC_DIR_HOST="${TFREC_DIR_HOST:-${HOST_DIR}/data/all-tfrec}"
TFREC_DIR_CONT="${CONT_DIR}/data/all-tfrec"

GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null | tr ' ' '_' | tr '/' '_' | tr -cd '[:alnum:]_')}"
GPU_TAG="${GPU_TAG:-${GPU_NAME_RAW:-gpu}}"
JOB_ID_TAG="${JOB_ID_TAG:-${SLURM_JOB_ID:-${OAR_JOB_ID:-manual}}}"

BASE_RESULTS_HOST="${BASE_RESULTS_HOST:-${HOST_DIR}/results/${GPU_TAG}+${JOB_ID_TAG}/runs}"
BASE_RESULTS_CONT="${CONT_DIR}/results/${GPU_TAG}+${JOB_ID_TAG}/runs"

# Hiperparâmetros baseline (sem otimizações)
BATCH_SIZE=${BATCH_SIZE:-96}
EPOCHS=${EPOCHS:-200}
LR=${LR:-3e-4}
IMG_SIZE=${IMG_SIZE:-299}
MODEL=${MODEL:-inception_v3}

# Validações
if [ ! -f "$CONTAINER" ]; then
  echo "❌ Container não encontrado: $CONTAINER" >&2
  echo "   Build com: apptainer build hcpa.sif Apptainer.def" >&2
  exit 1
fi
if [ ! -d "$TFREC_DIR_HOST" ]; then
  echo "❌ TFRecords não encontrados em $TFREC_DIR_HOST" >&2
  exit 1
fi

mkdir -p "$BASE_RESULTS_HOST"

echo "=============================================="
echo " monai_base - Baseline (sem otimizações)"
echo "=============================================="
echo " Container: $CONTAINER"
echo " TFRecords: $TFREC_DIR_HOST"
echo " Results:   $BASE_RESULTS_HOST"
echo " Epochs:    $EPOCHS"
echo " Batch:     $BATCH_SIZE"
echo " LR:        $LR"
echo " Model:     $MODEL"
echo "=============================================="
echo ""

for i in $(seq 0 9); do
  RUN_DIR_HOST="${BASE_RESULTS_HOST}/${i}"
  RUN_DIR_CONT="${BASE_RESULTS_CONT}/${i}"

  echo "=== run_${i} -> ${RUN_DIR_HOST} ==="
  mkdir -p "${RUN_DIR_HOST}"

  singularity exec --nv \
    -B "${HOST_DIR}:${CONT_DIR}" \
    "$CONTAINER" \
    env PYTHONPATH=${CONT_DIR} \
    bash -lc "cd ${CONT_DIR} && \
      LIBCUDA_DIR=/usr/local/cuda/compat/lib; \
      TMP_LIBCUDA=/tmp/libcuda_fix; \
      if [ -f \"\${LIBCUDA_DIR}/libcuda.so.1\" ]; then \
        mkdir -p \"\${TMP_LIBCUDA}\"; \
        ln -sf \"\${LIBCUDA_DIR}/libcuda.so.1\" \"\${TMP_LIBCUDA}/libcuda.so\"; \
        export LD_LIBRARY_PATH=\"\${TMP_LIBCUDA}:\${LD_LIBRARY_PATH:-}\"; \
      fi; \
      python3 train.py \
        --tfrec_dir ${TFREC_DIR_CONT} \
        --results ${RUN_DIR_CONT} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${EPOCHS} \
        --learning_rate ${LR} \
        --image_size ${IMG_SIZE} \
        --model ${MODEL} \
        --no_compile \
        --mixup_alpha 0.0 \
        --cutmix_alpha 0.0 \
        --label_smoothing 0.0 \
        --ema_decay 0.0 \
        --scheduler none"

  echo "✓ run_${i} finalizado."
done

echo ""
echo "=============================================="
echo " Todos os 10 treinamentos baseline concluídos!"
echo " Resultados em: $BASE_RESULTS_HOST"
echo "=============================================="
