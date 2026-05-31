#!/usr/bin/env bash
###############################################################################
# g5k_node_state.sh - Helper para localizar o node/job preparado pelo
#                     g5k_setup_node.sh.
#
# Sourceable por outros scripts. Exporta:
#   G5K_NODE, OAR_JOB_ID, PROJECT_DIR, JOB_END_EPOCH
# Em modo CLI imprime "OAR_JOB_ID NODE" na stdout.
###############################################################################

set -euo pipefail

STATE_DIR_BASE="${STATE_DIR:-$HOME/.g5k_hcpa}"
JOB_LABEL="${JOB_LABEL:-}"
if [[ -n "${JOB_LABEL}" ]]; then
  STATE_DIR="${STATE_DIR_BASE}/${JOB_LABEL}"
else
  STATE_DIR="${STATE_DIR_BASE}"
fi
STATE_FILE="${STATE_FILE:-$STATE_DIR/state.env}"

g5k_load_state() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    echo "[g5k] estado ausente: ${STATE_FILE}" >&2
    echo "[g5k] rode tools/g5k_setup_node.sh no flyon antes." >&2
    return 91
  fi
  # shellcheck disable=SC1090
  source "${STATE_FILE}"
  : "${OAR_JOB_ID:?OAR_JOB_ID nao definido em ${STATE_FILE}}"
  : "${G5K_NODE:?G5K_NODE nao definido em ${STATE_FILE}}"
  : "${PROJECT_DIR:?PROJECT_DIR nao definido em ${STATE_FILE}}"

  if command -v oarstat >/dev/null 2>&1; then
    local state
    state="$(oarstat -j "${OAR_JOB_ID}" -s 2>/dev/null | awk -F': *' '{print $2}' | tr -d ' ' || true)"
    if [[ -n "${state}" && "${state}" != "Running" ]]; then
      echo "[g5k] AVISO: job ${OAR_JOB_ID} esta em state=${state}; rode g5k_setup_node.sh novamente" >&2
    fi
  fi

  if [[ -n "${JOB_END_EPOCH:-}" ]]; then
    local now remaining
    now="$(date +%s)"
    remaining=$(( JOB_END_EPOCH - now ))
    if (( remaining <= 0 )); then
      echo "[g5k] AVISO: walltime do job ${OAR_JOB_ID} ja expirou" >&2
    elif (( remaining < 1800 )); then
      echo "[g5k] AVISO: restam apenas $(( remaining / 60 )) minutos de walltime" >&2
    fi
  fi

  export OAR_JOB_ID G5K_NODE PROJECT_DIR JOB_END_EPOCH
}

if [[ "${BASH_SOURCE[0]:-$0}" == "${0}" ]]; then
  g5k_load_state
  echo "OAR_JOB_ID=${OAR_JOB_ID}"
  echo "G5K_NODE=${G5K_NODE}"
  echo "PROJECT_DIR=${PROJECT_DIR}"
  if [[ -n "${JOB_END_EPOCH:-}" ]]; then
    echo "JOB_END=$(date -d "@${JOB_END_EPOCH}" 2>/dev/null || echo "${JOB_END_EPOCH}")"
  fi
fi
