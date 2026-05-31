#!/usr/bin/env bash

# ============================================================================
# COMPARATIVE TRAINING SCRIPT - pytorch_opt vs hybrid_simple vs hybrid_token_reduction
# ============================================================================
#
# Este script treina os 3 modelos sequencialmente com protocolo JUSTO:
# - Mesmo backbone: Inception V3
# - Mesmo preprocessing: 299x299 imagens
# - Mesma augmentação: DALI pipeline
# - Mesmos hyperparameters
# ============================================================================

set -euxo pipefail

# Configuração
PROJ_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
TFREC_DIR="${TFREC_DIR:-${PROJ_DIR}/data/all-tfrec}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EPOCHS="${EPOCHS:-200}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
IMG_SIZE="${IMG_SIZE:-299}"

echo "=========================================="
echo "COMPARATIVE TRAINING - 3 Modelos"
echo "=========================================="
echo "TFRecords: ${TFREC_DIR}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "=========================================="

if [[ ! -d "${TFREC_DIR}" ]]; then
  echo "Erro: TFREC_DIR não existe: ${TFREC_DIR}"
  exit 1
fi

# ============================================================================
# 1. PyTorch Opt (Baseline)
# ============================================================================
echo ""
echo "=========================================="
echo "[1/3] Treinando pytorch_opt (CNN puro)"
echo "=========================================="

cd "${PROJ_DIR}/pytorch_opt"
mkdir -p logs

if [[ -f "train.slurm" ]]; then
  # Se houver SLURM script, usa ele com variáveis
  BATCH_SIZE=${BATCH_SIZE} \
  EPOCHS=${EPOCHS} \
  LEARNING_RATE=${LEARNING_RATE} \
  sbatch train.slurm
  echo "SLURM job submetido para pytorch_opt"
else
  # Senão, roda direto
  python3 dr_hcpa_v2_2024.py \
    --tfrec_dir "${TFREC_DIR}" \
    --results "results/comparative_$(date +%Y%m%d_%H%M%S)" \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lrate ${LEARNING_RATE} \
    --model inception_v3 \
    --normalize preprocess \
    --enable_dali \
    --enable_amp \
    --enable_cosine \
    --enable_ema \
    --seed 42
fi

echo "pytorch_opt completo!"

# ============================================================================
# 2. Hybrid Simple
# ============================================================================
echo ""
echo "=========================================="
echo "[2/3] Treinando hybrid_simple (CNN + Transformador)"
echo "=========================================="

cd "${PROJ_DIR}/hybrid_simple"
mkdir -p logs

BATCH_SIZE=${BATCH_SIZE} \
EPOCHS=${EPOCHS} \
LEARNING_RATE=${LEARNING_RATE} \
ENABLE_AMP=1 \
ENABLE_EMA=1 \
ENABLE_DALI=1 \
bash train.slurm

echo "hybrid_simple completo!"

# ============================================================================
# 3. Hybrid Token Reduction
# ============================================================================
echo ""
echo "=========================================="
echo "[3/3] Treinando hybrid_token_reduction (CNN + Transformer + Token Reduction)"
echo "=========================================="

cd "${PROJ_DIR}/hybrid_token_reduction"
mkdir -p logs

BATCH_SIZE=${BATCH_SIZE} \
EPOCHS=${EPOCHS} \
LEARNING_RATE=${LEARNING_RATE} \
KEEP_RATIO=0.5 \
ENABLE_AMP=1 \
ENABLE_EMA=1 \
ENABLE_DALI=1 \
bash train.slurm

echo "hybrid_token_reduction completo!"

# ============================================================================
# ANÁLISE COMPARATIVA
# ============================================================================
echo ""
echo "=========================================="
echo "TREINAMENTO COMPARATIVO CONCLUÍDO"
echo "=========================================="
echo ""
echo "Resultados salvos em:"
echo "  1. ${PROJ_DIR}/pytorch_opt/results/"
echo "  2. ${PROJ_DIR}/hybrid_simple/results/"
echo "  3. ${PROJ_DIR}/hybrid_token_reduction/results/"
echo ""
echo "Para comparar métricas, execute:"
echo "  python3 compare_results.py"
echo "=========================================="
