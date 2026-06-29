#!/usr/bin/env bash
###############################################################################
# g5k_multi_run.sh - Orquestra varios jobs HCPA em paralelo em nodes diferentes
#                    do cluster hydra (Lyon).
#
# Rodar no flyon. Exemplo:
#   bash tools/g5k_multi_run.sh \
#     vit_pure:hydra-1 \
#     tensorflow_opt:hydra-2 \
#     hybrid_simple:hydra-4
#
# Cada par "<approach>:<host>" vira:
#   1) JOB_LABEL=<approach> + setup_node (aloca node inteiro, kadeploy, build .sif)
#   2) JOB_LABEL=<approach> + <approach>/run_g5k_hydra.sh (roda 0..9)
#
# Vars de ambiente (aplicam a TODOS os pares; cada par herda):
#   WALLTIME=02:30:00            # ajustar conforme janela bronze restante
#   RUN_START=0 RUN_END=9
#   TARGET_EPOCHS=200
#
# Vars por par (sobrescrever no comando):
#   APPROACH_OVERRIDES_<APPROACH>="EPOCHS=50 BATCH_SIZE=64"   (opcional)
#
# State e logs:
#   ~/.g5k_hcpa/<approach>/state.env
#   ~/.g5k_hcpa/<approach>/full_run.log
#   ~/.g5k_hcpa/<approach>/logs/*
###############################################################################

set -uo pipefail

if [[ $# -lt 1 ]]; then
  cat >&2 <<EOF
uso: $0 <approach>:<host> [<approach>:<host> ...]

exemplos:
  $0 pytorch_opt:hydra-1
  $0 vit_pure:hydra-1 tensorflow_opt:hydra-2 hybrid_simple:hydra-4

approaches validas: pytorch_opt tensorflow_opt vit_pure
                    hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt
hosts: hydra-1 .. hydra-4 (ou *.lyon.grid5000.fr)
EOF
  exit 2
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/hcpa}"
WALLTIME="${WALLTIME:-02:30:00}"
NIGHT_JOB="${NIGHT_JOB:-0}"
RUN_START="${RUN_START:-0}"
RUN_END="${RUN_END:-9}"
STATE_DIR_BASE="${STATE_DIR:-$HOME/.g5k_hcpa}"

VALID_APPROACHES=(pytorch_opt tensorflow_opt vit_pure hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt)
is_valid_approach() {
  local a x
  a="$1"
  for x in "${VALID_APPROACHES[@]}"; do [[ "$x" == "$a" ]] && return 0; done
  return 1
}

PAIRS=("$@")
declare -a LABELS HOSTS APPROACHES_LIST

for spec in "${PAIRS[@]}"; do
  IFS=':' read -r approach host <<<"${spec}"
  if [[ -z "${approach}" || -z "${host}" ]]; then
    echo "[multi] formato invalido: ${spec} (esperado <approach>:<host>)" >&2
    exit 2
  fi
  if ! is_valid_approach "${approach}"; then
    echo "[multi] approach desconhecida: ${approach}" >&2
    exit 2
  fi
  APPROACHES_LIST+=("${approach}")
  HOSTS+=("${host}")
  LABELS+=("${approach}")
done

echo "[multi] PROJECT_DIR=${PROJECT_DIR}"
echo "[multi] WALLTIME=${WALLTIME} NIGHT_JOB=${NIGHT_JOB} RUN_START=${RUN_START} RUN_END=${RUN_END}"
echo "[multi] pares:"
for i in "${!APPROACHES_LIST[@]}"; do
  printf '  %d) approach=%s host=%s label=%s\n' "$((i+1))" "${APPROACHES_LIST[i]}" "${HOSTS[i]}" "${LABELS[i]}"
done

cd "${PROJECT_DIR}"

# Cria um wrapper-por-par e dispara em background.
for i in "${!APPROACHES_LIST[@]}"; do
  approach="${APPROACHES_LIST[i]}"
  host="${HOSTS[i]}"
  label="${LABELS[i]}"
  state_dir="${STATE_DIR_BASE}/${label}"
  mkdir -p "${state_dir}/logs"
  wrapper="${state_dir}/full_run.sh"
  logf="${state_dir}/full_run.log"

  # Permite overrides por approach via env APPROACH_OVERRIDES_<APPROACH>
  overrides_var="APPROACH_OVERRIDES_$(echo "${approach}" | tr '[:lower:]' '[:upper:]')"
  overrides="${!overrides_var:-}"

  cat > "${wrapper}" <<EOWRAP
#!/usr/bin/env bash
set -uo pipefail
cd "${PROJECT_DIR}"
exec >> "${logf}" 2>&1
echo "========================================================="
echo "[multi-${label}] inicio em \$(date -Iseconds)"
echo "  approach=${approach}  host=${host}  walltime=${WALLTIME}"
echo "========================================================="
export JOB_LABEL="${label}"
export WALLTIME="${WALLTIME}"
export NIGHT_JOB="${NIGHT_JOB}"
export G5K_HOST="${host}"
export APPROACHES="${approach}"
# overrides per-approach (e.g. EPOCHS=50 BATCH_SIZE=64)
${overrides}
if ! bash tools/g5k_setup_node.sh; then
  echo "[multi-${label}] setup_node FALHOU"
  exit 1
fi
echo "========================================================="
echo "[multi-${label}] setup ok, iniciando run em \$(date -Iseconds)"
echo "========================================================="
cd "${PROJECT_DIR}/${approach}"
export RUN_START="${RUN_START}"
export RUN_END="${RUN_END}"
bash run_g5k_hydra.sh
rc=\$?
echo "========================================================="
echo "[multi-${label}] FIM rc=\$rc em \$(date -Iseconds)"
echo "========================================================="
exit \$rc
EOWRAP
  chmod +x "${wrapper}"

  echo "[multi] disparando: ${wrapper} -> log=${logf}"
  nohup bash "${wrapper}" </dev/null >/dev/null 2>&1 &
  disown 2>/dev/null || true
  pids[i]=$!
done

sleep 2
echo
echo "[multi] processos lancados:"
pgrep -af 'full_run.sh' | head
echo
echo "[multi] acompanhe com:"
echo "  oarstat -u"
for label in "${LABELS[@]}"; do
  echo "  tail -F ${STATE_DIR_BASE}/${label}/full_run.log"
done
