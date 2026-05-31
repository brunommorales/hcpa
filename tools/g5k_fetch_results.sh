#!/usr/bin/env bash
###############################################################################
# g5k_fetch_results.sh - Puxa do frontend Lyon (G5K) para o GPPD os
#                        resultados de cada abordagem, em sua respectiva
#                        pasta results/, mantendo apenas CSVs e PDFs.
#
# Rodar a partir do GPPD (maquina local).
#
# Uso:
#   bash tools/g5k_fetch_results.sh                          # todas as abordagens
#   APPROACHES="pytorch_opt vit_pure" \
#     bash tools/g5k_fetch_results.sh                        # subset
#   DRY_RUN=1 bash tools/g5k_fetch_results.sh                # simulacao
#   KEEP_ALL=1 bash tools/g5k_fetch_results.sh               # nao filtrar (puxa tudo de results/)
#   PURGE_REMOTE=1 bash tools/g5k_fetch_results.sh           # apaga do G5K apos confirmar no GPPD
#
# Variaveis:
#   APPROACHES   default: todas as 6 abordagens
#   SRC_ROOT     default /home/bmorales/projects/hcpa            (no grid5000)
#   DST_ROOT     default /home/users/bmmorales/projects/hcpa     (no GPPD)
#   GRID_USER    default bmorales
#   ACCESS_HOST  default access.grid5000.fr
#   SITE_HOST    default lyon.grid5000.fr
#   FRONTEND_HOST default flyon.lyon.grid5000.fr
#   PURGE_REMOTE default 0. Se 1, usa rsync --remove-source-files: cada arquivo
#                so e apagado do G5K APOS ser transferido com sucesso para o
#                GPPD (verificacao atomica por arquivo). Depois remove dirs
#                vazios remotos. Libera quota do home no G5K. Ignorado se DRY_RUN=1.
###############################################################################

set -euo pipefail

APPROACHES_DEFAULT="pytorch_opt tensorflow_opt vit_pure hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt"
APPROACHES="${APPROACHES:-$APPROACHES_DEFAULT}"
SRC_ROOT="${SRC_ROOT:-/home/bmorales/projects/hcpa}"
DST_ROOT="${DST_ROOT:-/home/users/bmmorales/projects/hcpa}"
GRID_USER="${GRID_USER:-bmorales}"
ACCESS_HOST="${ACCESS_HOST:-access.grid5000.fr}"
SITE_HOST="${SITE_HOST:-lyon.grid5000.fr}"
FRONTEND_HOST="${FRONTEND_HOST:-flyon.lyon.grid5000.fr}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_ALL="${KEEP_ALL:-0}"
PURGE_REMOTE="${PURGE_REMOTE:-0}"

SSH_JUMP="ssh -J ${GRID_USER}@${ACCESS_HOST},${GRID_USER}@${SITE_HOST}"

DRY_FLAG=""
if [[ "${DRY_RUN}" == "1" ]]; then
  DRY_FLAG="-n"
  echo "[fetch] DRY_RUN=1 (apenas simulacao)"
fi

log()  { printf '[fetch][%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }

# Por padrao, mantemos apenas csv/pdf e tambem env_manifest.txt/run_manifest.txt
# (descobertas/contexto sao uteis). Se KEEP_ALL=1, copia tudo.
filter_args=()
if [[ "${KEEP_ALL}" != "1" ]]; then
  filter_args+=(
    --include='*/'
    --include='*.csv'
    --include='*.pdf'
    --include='*.txt'
    --include='*.json'
    --include='*.png'
    --exclude='*'
  )
fi

fetch_one() {
  local approach="$1"
  local src="${GRID_USER}@${FRONTEND_HOST}:${SRC_ROOT}/${approach}/results/"
  local dst="${DST_ROOT}/${approach}/results/"

  if [[ ! -d "${DST_ROOT}/${approach}" ]]; then
    log "[${approach}] AVISO: ${DST_ROOT}/${approach} nao existe no GPPD; pulando."
    return 0
  fi
  mkdir -p "${dst}"

  # --remove-source-files: rsync so apaga da origem (G5K) os arquivos que
  # foram efetivamente transferidos com sucesso. E a "verificacao antes de
  # apagar" pedida: nada some do G5K sem antes chegar no GPPD.
  local purge_flag=""
  if [[ "${PURGE_REMOTE}" == "1" && "${DRY_RUN}" != "1" ]]; then
    purge_flag="--remove-source-files"
    log "[${approach}] PURGE_REMOTE=1: apos transferir, apaga do G5K"
  fi

  log "[${approach}] rsync ${src} -> ${dst}"
  if rsync -avz --progress --partial ${DRY_FLAG} ${purge_flag} \
       -e "${SSH_JUMP}" \
       "${filter_args[@]}" \
       "${src}" "${dst}"; then
    if [[ -n "${purge_flag}" ]]; then
      # rsync ja removeu os arquivos transferidos; agora limpa dirs vazios no G5K
      ${SSH_JUMP} "${GRID_USER}@${FRONTEND_HOST}" \
        "find '${SRC_ROOT}/${approach}/results' -mindepth 1 -type d -empty -delete 2>/dev/null || true" \
        && log "[${approach}] dirs vazios removidos no G5K" \
        || log "[${approach}] aviso: limpeza de dirs vazios no G5K falhou"
    fi
  else
    log "[${approach}] rsync retornou erro (nao critico se diretorio vazio)"
  fi
}

for approach in ${APPROACHES}; do
  fetch_one "${approach}"
done

log "concluido. Resultados em ${DST_ROOT}/<approach>/results/"
