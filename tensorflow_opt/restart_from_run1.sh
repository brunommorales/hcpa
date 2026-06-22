#!/usr/bin/env bash
# restart_from_run1.sh
# Mata o processo atual do tensorflow_opt e reinicia a partir do RUN 1.
# Executar no flyon APOS run_0 completar (CSV salvo).
set -uo pipefail

LOG="$HOME/.g5k_hcpa/tensorflow_opt/full_run.log"

echo "" >> "${LOG}"
echo "=========================================================" >> "${LOG}"
echo "[restart] iniciando restart from RUN 1 em $(date -Iseconds)" >> "${LOG}"
echo "=========================================================" >> "${LOG}"

# Mata o processo full_run.sh e run_g5k_hydra.sh do tensorflow_opt
pkill -TERM -f ".g5k_hcpa/tensorflow_opt/full_run.sh" 2>/dev/null || true
pkill -TERM -f "tensorflow_opt/run_g5k_hydra.sh"     2>/dev/null || true
sleep 5
pkill -KILL -f ".g5k_hcpa/tensorflow_opt/full_run.sh" 2>/dev/null || true
pkill -KILL -f "tensorflow_opt/run_g5k_hydra.sh"     2>/dev/null || true
sleep 3

echo "[restart] processo morto, iniciando RUN_START=1 RUN_END=9 com XLA cache fix" >> "${LOG}"

cd /home/bmorales/projects/hcpa/tensorflow_opt
export JOB_LABEL="tensorflow_opt"
export RUN_START=1
export RUN_END=9
nohup bash run_g5k_hydra.sh >> "${LOG}" 2>&1 &
echo "[restart] novo processo PID=$!" >> "${LOG}"
echo "[restart] done" >> "${LOG}"
