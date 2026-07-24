#!/usr/bin/env bash
###############################################################################
# tools/lr_sweep.sh — mini-varredura de learning rate para a receita comum.
#
# Por que: tensorflow_opt rodava com LR efetivo 2e-4 (fine_tune_lr, pois
# freeze_epochs=1) e pytorch_opt com 5e-4. Ao unificar a receita, precisamos de
# UM LR. Copiar o de um para o outro pode quebrar a convergência e invalidar a
# comparação clínica — então mede-se.
#
# Faz 1 run curto por (framework x LR). Escolhe-se o LR com a melhor val_auc de
# pico nos DOIS frameworks (ou o melhor par, documentando).
#
# Resultados vão para <approach>/results/lrsweep_<lr>_g5k_hydra/run_0/ e NÃO
# sobrescrevem a produção (via RESULTS_TAG).
#
# Custo: 6 runs x SWEEP_EPOCHS. Com 60 épocas ≈ 1,5 h de GPU no total.
#
# ATENÇÃO — cota Grid5000: isto consome GPU·h. De dia a cota vale; rode na
# janela noturna/fim de semana, ou com um job <= 1 h.
#
# Uso:
#   bash tools/lr_sweep.sh                       # 5e-4, 1e-3, 3e-3 x 2 frameworks
#   SWEEP_LRS="1e-4 5e-4" SWEEP_EPOCHS=40 bash tools/lr_sweep.sh
#   SWEEP_APPROACHES=tensorflow_opt bash tools/lr_sweep.sh
###############################################################################
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

SWEEP_LRS="${SWEEP_LRS:-5e-4 1e-3 3e-3}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-60}"
SWEEP_APPROACHES="${SWEEP_APPROACHES:-tensorflow_opt pytorch_opt}"

echo "=== mini-varredura de LR ==="
echo "  LRs        : ${SWEEP_LRS}"
echo "  épocas     : ${SWEEP_EPOCHS}"
echo "  abordagens : ${SWEEP_APPROACHES}"
echo "  1 run por combinação (RUN_START=RUN_END=0)"
echo

for approach in ${SWEEP_APPROACHES}; do
  for lr in ${SWEEP_LRS}; do
    tag="lrsweep_${lr}"
    echo "--- ${approach}  lr=${lr}  -> results/${tag}_g5k_hydra/run_0"
    HCPA_LRATE="${lr}" \
    HCPA_EPOCHS="${SWEEP_EPOCHS}" \
    RESULTS_TAG="${tag}" \
    RUN_START=0 RUN_END=0 \
      bash "${ROOT}/${approach}/run_g5k_hydra.sh"
  done
done

echo
echo "=== varredura concluída. Para resumir: ==="
echo "  python3 ${ROOT}/tools/lr_sweep_report.py"
