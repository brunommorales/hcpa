#!/usr/bin/env bash
###############################################################################
# g5k_target.sh - Presets dos alvos G5K do estudo HCPA (bronze-acessiveis) e
#                 dispatcher unico para send / setup / run / fetch.
#
# Alvos (todos prod=NO, acessiveis a conta bronze; H100 e production-only e ficou
# de fora):
#   lyon-hydra     -> GH200 480GB   | Lyon  | ARM aarch64 | exotic | deploy ubuntugh2404-arm64-big
#   lille-chuc     -> A100-SXM4-40G | Lille | x86_64      | -      | deploy debiannvopen11-big
#   lille-chicoree -> H200 NVL 140G | Lille | x86_64      | exotic | deploy debiannvopen11-big
#
# === No GPPD (este box) ===
#   bash tools/g5k_target.sh <alvo> send        # envia codigo + dados (tfrec+idx) ao site
#   bash tools/g5k_target.sh <alvo> send-code   # so codigo (sem dataset)
#   bash tools/g5k_target.sh <alvo> fetch       # puxa results (csv+pdf) do site -> GPPD
#   bash tools/g5k_target.sh <alvo> fetch-purge # idem, e apaga do G5K o que transferiu
#
# === No frontend do site (flyon / flille) ===
#   WALLTIME=08:00:00 NIGHT_JOB=1 bash tools/g5k_target.sh <alvo> setup
#                                              # oarsub + kadeploy + build .sif das 9 abordagens
#   bash tools/g5k_target.sh <alvo> run <approach>   # roda 0..9 de UMA abordagem
#   bash tools/g5k_target.sh <alvo> run-all          # roda as 9 abordagens, 0..9
#
# Lembretes bronze: a janela e noturna/FDS (hora de Paris). Use NIGHT_JOB=1 e
# WALLTIME que caiba na janela. Ver memory reference-grid5000-bronze.
###############################################################################

set -euo pipefail

PROJECT_DIR_LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${PROJECT_DIR_LOCAL}/tools"

ALL_APPROACHES="pytorch_base pytorch_opt tensorflow_base tensorflow_opt vit_pure hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt retfound_green"

TARGET="${1:-}"
ACTION="${2:-}"

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}"; exit "${1:-1}"; }
[[ -z "${TARGET}" || -z "${ACTION}" ]] && usage 1

# ---- presets por alvo ----
case "${TARGET}" in
  lyon-hydra)
    export G5K_SITE=lyon   G5K_CLUSTER=hydra
    export FRONTEND_HOST=flyon.lyon.grid5000.fr
    export DEPLOY_IMAGE=ubuntugh2404-arm64-big
    export EXOTIC=1
    ;;
  lille-chuc)
    export G5K_SITE=lille  G5K_CLUSTER=chuc
    export FRONTEND_HOST=flille.lille.grid5000.fr
    export DEPLOY_IMAGE=debiannvopen11-big
    export EXOTIC=0
    ;;
  lille-chicoree)
    export G5K_SITE=lille  G5K_CLUSTER=chicoree
    export FRONTEND_HOST=flille.lille.grid5000.fr
    export DEPLOY_IMAGE=debiannvopen11-big
    export EXOTIC=1
    ;;
  nantes-ecotaxe)
    export G5K_SITE=nantes  G5K_CLUSTER=ecotaxe
    export FRONTEND_HOST=fnantes.nantes.grid5000.fr
    export DEPLOY_IMAGE=debiannvopen11-big
    export EXOTIC=1
    ;;
  *)
    echo "[g5k_target] alvo desconhecido: ${TARGET}" >&2
    echo "[g5k_target] use: lyon-hydra | lille-chuc | lille-chicoree | nantes-ecotaxe" >&2
    exit 2
    ;;
esac

# jump unico pelo access (vazio = sem hop extra) — funciona p/ todos os sites
export SITE_HOST=""
export GRID_USER="${GRID_USER:-bmorales}"

echo "[g5k_target] alvo=${TARGET} site=${G5K_SITE} cluster=${G5K_CLUSTER} deploy=${DEPLOY_IMAGE} exotic=${EXOTIC} frontend=${FRONTEND_HOST}"

case "${ACTION}" in
  send)
    bash "${TOOLS_DIR}/g5k_send_to_grid.sh"
    ;;
  send-code)
    NO_DATA=1 bash "${TOOLS_DIR}/g5k_send_to_grid.sh"
    ;;
  fetch)
    bash "${TOOLS_DIR}/g5k_fetch_results.sh"
    ;;
  fetch-purge)
    PURGE_REMOTE=1 bash "${TOOLS_DIR}/g5k_fetch_results.sh"
    ;;
  setup)
    bash "${TOOLS_DIR}/g5k_setup_node.sh"
    ;;
  run)
    approach="${3:-}"
    [[ -z "${approach}" ]] && { echo "[g5k_target] uso: run <approach>" >&2; exit 1; }
    bash "${PROJECT_DIR_LOCAL}/${approach}/run_g5k_hydra.sh"
    ;;
  run-all)
    for a in ${ALL_APPROACHES}; do
      echo "============================================================"
      echo "[g5k_target] === ${a} (${TARGET}) ==="
      echo "============================================================"
      bash "${PROJECT_DIR_LOCAL}/${a}/run_g5k_hydra.sh"
    done
    ;;
  *)
    echo "[g5k_target] acao desconhecida: ${ACTION}" >&2
    usage 1
    ;;
esac
