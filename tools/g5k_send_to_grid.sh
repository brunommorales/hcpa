#!/usr/bin/env bash
###############################################################################
# g5k_send_to_grid.sh - Envia codigo do HCPA + dataset all-tfrec do GPPD para
#                        o frontend Lyon do Grid5000.
#
# Rodar a partir do GPPD (maquina local). Faz rsync via SSH ProxyJump.
#
# Uso:
#   bash tools/g5k_send_to_grid.sh                 # envia tudo (codigo + data)
#   DRY_RUN=1 bash tools/g5k_send_to_grid.sh       # so simula
#   NO_DATA=1 bash tools/g5k_send_to_grid.sh       # nao envia data/all-tfrec
#   ONLY_DATA=1 bash tools/g5k_send_to_grid.sh     # so envia data/all-tfrec
#
# Variaveis:
#   SRC_ROOT       default /home/users/bmmorales/projects/hcpa
#   DST_ROOT       default /home/bmorales/projects/hcpa
#   GRID_USER      default bmorales
#   ACCESS_HOST    default access.grid5000.fr
#   SITE_HOST      default lyon.grid5000.fr
#   FRONTEND_HOST  default flyon.lyon.grid5000.fr
###############################################################################

set -euo pipefail

SRC_ROOT="${SRC_ROOT:-/home/users/bmmorales/projects/hcpa}"
DST_ROOT="${DST_ROOT:-/home/bmorales/projects/hcpa}"
GRID_USER="${GRID_USER:-bmorales}"
ACCESS_HOST="${ACCESS_HOST:-access.grid5000.fr}"
SITE_HOST="${SITE_HOST:-lyon.grid5000.fr}"
FRONTEND_HOST="${FRONTEND_HOST:-flyon.lyon.grid5000.fr}"
DRY_RUN="${DRY_RUN:-0}"
NO_DATA="${NO_DATA:-0}"
ONLY_DATA="${ONLY_DATA:-0}"

SSH_JUMP="ssh -J ${GRID_USER}@${ACCESS_HOST},${GRID_USER}@${SITE_HOST}"
DST="${GRID_USER}@${FRONTEND_HOST}:${DST_ROOT}/"

DRY_FLAG=""
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_FLAG="-n"
  echo "[send] DRY_RUN=1 (apenas simulacao)"
fi

CODE_INCLUDES=(
  '/'
  'tools/***'
  'clean_temp_loggers.py'
  'hybrid_shared/***'
  'hybrid_simple/***'
  'hybrid_token_reduction/***'
  'hybrid_token_reduction_opt/***'
  'vit_pure/***'
  'pytorch_base/***'
  'pytorch_opt/***'
  'tensorflow_base/***'
  'tensorflow_opt/***'
)

COMMON_EXCLUDES=(
  '**/results/***'
  '**/logs/***'
  '**/.venv/***'
  '**/env/***'
  '**/env_arm/***'
  '**/env_graph/***'
  '**/.env/***'
  '**/__pycache__/***'
  '**/tmp/***'
  '**/.cache/***'
  '**/.keras/***'
  '**/hcpa.sif'
  '**/*.sif'
)

rsync_args=( -avz --progress --partial ${DRY_FLAG} --prune-empty-dirs -e "${SSH_JUMP}" )
for e in "${COMMON_EXCLUDES[@]}"; do rsync_args+=( --exclude="${e}" ); done

send_code() {
  echo "[send] CODIGO: ${SRC_ROOT} -> ${DST}"
  local -a inc=()
  for i in "${CODE_INCLUDES[@]}"; do inc+=( --include="${i}" ); done
  rsync "${rsync_args[@]}" "${inc[@]}" --exclude='*' "${SRC_ROOT}/" "${DST}"
}

send_data() {
  echo "[send] DATA: ${SRC_ROOT}/data/all-tfrec/ -> ${DST}data/all-tfrec/"
  # Garante que o diretorio data/ exista no destino
  ${SSH_JUMP} "${GRID_USER}@${FRONTEND_HOST}" "mkdir -p '${DST_ROOT}/data'"
  rsync -avz --progress --partial ${DRY_FLAG} -e "${SSH_JUMP}" \
    "${SRC_ROOT}/data/all-tfrec/" \
    "${GRID_USER}@${FRONTEND_HOST}:${DST_ROOT}/data/all-tfrec/"
}

if [[ "${ONLY_DATA}" == "1" ]]; then
  send_data
elif [[ "${NO_DATA}" == "1" ]]; then
  send_code
else
  send_code
  send_data
fi

echo "[send] concluido."
