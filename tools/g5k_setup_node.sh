#!/usr/bin/env bash
###############################################################################
# g5k_setup_node.sh - Prepara um node hydra (Lyon) para rodar HCPA em Singularity
#
# Rodar a partir do flyon (frontend Lyon):
#   bash tools/g5k_setup_node.sh
#
# Por padrao:
#   1) submete oarsub -t exotic -t deploy em hydra (1 node, walltime configuravel)
#   2) aguarda o job entrar em Running
#   3) extrai o node alocado (uniq)
#   4) kadeploy ubuntugh2404-arm64-big -k
#   5) ssh root@node instala singularity-container
#   6) para cada Apptainer.def das abordagens, builda o .sif no node
#   7) grava ~/.g5k_hcpa/state.env com OAR_JOB_ID, G5K_NODE e expiracao para
#      os scripts run_g5k_hydra.sh detectarem o node automaticamente
#
# Variaveis de ambiente uteis:
#   WALLTIME=08:00:00           # tempo do job OAR
#   PROJECT_DIR=$HOME/projects/hcpa
#   DEPLOY_IMAGE=ubuntugh2404-arm64-big
#   APPROACHES="pytorch_opt tensorflow_opt vit_pure hybrid_simple \
#               hybrid_token_reduction hybrid_token_reduction_opt"
#   SKIP_BUILD=1                # nao builda .sif (so prepara node)
#   SKIP_KADEPLOY=1             # reusa kadeploy anterior (so checa)
#   REUSE_JOB=<id>              # nao submete um novo oarsub; usa esse job_id
#   FORCE_REBUILD_SIF=1         # rebuild mesmo se hcpa.sif ja existir
###############################################################################

set -euo pipefail

WALLTIME="${WALLTIME:-08:00:00}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/projects/hcpa}"
DEPLOY_IMAGE="${DEPLOY_IMAGE:-ubuntugh2404-arm64-big}"
G5K_CLUSTER="${G5K_CLUSTER:-hydra}"
G5K_HOST="${G5K_HOST:-}"    # ex: hydra-2 ou hydra-2.lyon.grid5000.fr (vazio = qualquer node do cluster)
G5K_QUEUE="${G5K_QUEUE:-default}"       # default; bronze user-class enforca horario, nao a fila
G5K_RESOURCES="${G5K_RESOURCES:-host=1}" # whole-node por padrao (gpus inteiras)
APPROACHES_DEFAULT="pytorch_opt tensorflow_opt vit_pure hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt"
APPROACHES="${APPROACHES:-$APPROACHES_DEFAULT}"
STATE_DIR_BASE="${STATE_DIR:-$HOME/.g5k_hcpa}"
JOB_LABEL="${JOB_LABEL:-}"   # vazio = comportamento legado (state.env na raiz)
if [[ -n "${JOB_LABEL}" ]]; then
  STATE_DIR="${STATE_DIR_BASE}/${JOB_LABEL}"
else
  STATE_DIR="${STATE_DIR_BASE}"
fi
STATE_FILE="${STATE_DIR}/state.env"
LOG_DIR="${LOG_DIR:-$STATE_DIR/logs}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SKIP_KADEPLOY="${SKIP_KADEPLOY:-0}"
FORCE_REBUILD_SIF="${FORCE_REBUILD_SIF:-0}"
REUSE_JOB="${REUSE_JOB:-}"
NIGHT_JOB="${NIGHT_JOB:-0}"   # 1 = adiciona -t night ao oarsub (agenda para a janela noturna)

mkdir -p "${STATE_DIR}" "${LOG_DIR}"

log()  { printf '[setup][%s] %s\n' "$(date +'%H:%M:%S')" "$*"; }
fail() { printf '[setup][FAIL] %s\n' "$*" >&2; exit 1; }

require_cmd() {
  local c
  for c in "$@"; do
    command -v "${c}" >/dev/null 2>&1 || fail "comando ausente: ${c}"
  done
}

require_cmd oarsub oarstat kadeploy3 ssh

# ----------- 1) submete OAR job (ou reusa) -----------
if [[ -n "${REUSE_JOB}" ]]; then
  OAR_JOB_ID="${REUSE_JOB}"
  log "reusando job existente OAR_JOB_ID=${OAR_JOB_ID}"
else
  prop="cluster='${G5K_CLUSTER}'"
  if [[ -n "${G5K_HOST}" ]]; then
    case "${G5K_HOST}" in
      *.grid5000.fr) host_full="${G5K_HOST}" ;;
      *)             host_full="${G5K_HOST}.lyon.grid5000.fr" ;;
    esac
    prop="${prop} AND host='${host_full}'"
    log "submetendo oarsub com host fixo: ${host_full}"
  else
    log "submetendo oarsub para qualquer node do cluster ${G5K_CLUSTER}"
  fi
  night_flag=""
  [[ "${NIGHT_JOB}" == "1" ]] && night_flag="-t night"
  log "props: ${prop} | walltime=${WALLTIME} | queue=${G5K_QUEUE} | resources=${G5K_RESOURCES} | night=${NIGHT_JOB}"
  # shellcheck disable=SC2086
  submit_out="$(oarsub \
    -q "${G5K_QUEUE}" \
    -t exotic -t deploy \
    ${night_flag} \
    -l "${G5K_RESOURCES},walltime=${WALLTIME}" \
    -p "${prop}" \
    -O "${LOG_DIR}/oar.%jobid%.out" \
    -E "${LOG_DIR}/oar.%jobid%.err" \
    "sleep $(( $(echo "${WALLTIME}" | awk -F: '{print ($1*3600)+($2*60)+$3}') ))" \
    2>&1)" || fail "oarsub falhou: ${submit_out}"
  echo "${submit_out}"
  OAR_JOB_ID="$(echo "${submit_out}" | sed -n 's/^OAR_JOB_ID=\([0-9][0-9]*\).*/\1/p' | head -1)"
  [[ -n "${OAR_JOB_ID}" ]] || fail "nao foi possivel extrair OAR_JOB_ID da saida do oarsub"
  log "OAR_JOB_ID=${OAR_JOB_ID}"
fi

# ----------- 2) aguarda job entrar em Running -----------
log "aguardando job ${OAR_JOB_ID} entrar em Running (timeout 20min)..."
deadline=$(( $(date +%s) + 1200 ))
while :; do
  # oarstat -j N -s -> "N: Running" (separador ':', nao '=')
  state="$(oarstat -j "${OAR_JOB_ID}" -s 2>/dev/null | awk -F': *' '{print $2}' | tr -d ' ')"
  if [[ "${state}" == "Running" ]]; then
    log "job ${OAR_JOB_ID} esta Running"
    break
  fi
  if (( $(date +%s) >= deadline )); then
    fail "timeout aguardando job ${OAR_JOB_ID} virar Running (state=${state:-?})"
  fi
  log "state=${state:-Unknown}; aguardando..."
  sleep 10
done

# ----------- 3) extrai node alocado (uniq) -----------
RAW="${STATE_DIR}/hydra_nodes_job${OAR_JOB_ID}.txt"
UNIQ="${STATE_DIR}/hydra_nodes_job${OAR_JOB_ID}.uniq.txt"
# oarstat -fj imprime 'assigned_hostnames = host1, host2'
NODE_LIST="$(oarstat -fj "${OAR_JOB_ID}" 2>/dev/null \
  | awk -F'=' '/^[[:space:]]*assigned_hostnames[[:space:]]*=/{print $2}' \
  | tr ',' '\n' | sed 's/^[[:space:]]*//; s/[[:space:]]*$//' \
  | grep -v '^$' || true)"
if [[ -z "${NODE_LIST}" ]]; then
  fail "nao foi possivel extrair assigned_hostnames de oarstat -fj ${OAR_JOB_ID}"
fi
printf '%s\n' "${NODE_LIST}" > "${RAW}"
sort -u "${RAW}" > "${UNIQ}"

n_nodes="$(wc -l < "${UNIQ}")"
[[ "${n_nodes}" -eq 1 ]] || { cat "${UNIQ}"; fail "esperado 1 node unico em ${UNIQ}; recebi ${n_nodes}"; }
G5K_NODE="$(head -n 1 "${UNIQ}")"
log "node alocado: ${G5K_NODE}"

# ----------- 4) kadeploy (idempotente: pula se ssh root@node ja funciona) -----------
node_already_deployed() {
  ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=8 \
      "root@${G5K_NODE}" "true" 2>/dev/null
}

if [[ "${SKIP_KADEPLOY}" == "1" ]]; then
  log "SKIP_KADEPLOY=1; verificando se root@${G5K_NODE} responde..."
  node_already_deployed || fail "ssh root@${G5K_NODE} falhou; kadeploy parece necessario"
elif node_already_deployed; then
  log "node ${G5K_NODE} ja deployado (ssh root respondeu); pulando kadeploy"
else
  log "kadeploy3 -f ${UNIQ} -e ${DEPLOY_IMAGE} -k (pode levar ~10min)"
  kadeploy3 -f "${UNIQ}" -e "${DEPLOY_IMAGE}" -k 2>&1 | tee -a "${LOG_DIR}/kadeploy.${OAR_JOB_ID}.log"
  if ! ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 \
         "root@${G5K_NODE}" "hostname && uname -m && nvidia-smi --query-gpu=name --format=csv,noheader"; then
    fail "ssh root@${G5K_NODE} ainda falha apos kadeploy"
  fi
fi

# ----------- 5) instala singularity-container -----------
log "instalando singularity-container em root@${G5K_NODE}..."
ssh -o StrictHostKeyChecking=no "root@${G5K_NODE}" "bash -s" <<'EOSSH'
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
if command -v singularity >/dev/null 2>&1; then
  echo "[node] singularity ja instalado: $(singularity --version)"
  exit 0
fi
# desabilita repos quebrados conhecidos (mellanox costuma 404 em imagens novas)
for f in /etc/apt/sources.list.d/mellanox.list /etc/apt/sources.list.d/mlnx_ofed.list; do
  if [[ -f "$f" ]]; then
    echo "[node] desabilitando $f (repo quebrado conhecido)"
    mv "$f" "${f}.disabled" || true
  fi
done
# remove qualquer linha mellanox/mlnx remanescente nas sources principais
sed -i '/mellanox\|mlnx_ofed/d' /etc/apt/sources.list 2>/dev/null || true

# update tolerante: se algum repo cair, tenta seguir com o cache parcial
apt-get update -y || apt-get update -y --allow-unauthenticated || true

if ! apt-get install -y singularity-container; then
  echo "[node] falha no install padrao; tentando com cache atual" >&2
  apt-get install -y --no-install-recommends singularity-container \
    || { echo "[node] singularity install falhou DEFINITIVAMENTE" >&2; exit 91; }
fi
singularity --version
EOSSH

# ----------- 6) builda .sif para cada approach -----------
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
LOGIN_USER="${USER:-bmorales}"

build_sif_for() {
  local approach="$1"
  local apath="${PROJECT_DIR}/${approach}"
  local sif="${apath}/hcpa.sif"
  local def="${apath}/Apptainer.def"

  if [[ ! -f "${def}" ]]; then
    log "[${approach}] SEM Apptainer.def em ${def}; pulando build"
    return 0
  fi

  ssh -o StrictHostKeyChecking=no "root@${G5K_NODE}" "test -f '${def}'" \
    || fail "[${approach}] node nao enxerga ${def} (rsync do projeto antes?)"

  if [[ "${FORCE_REBUILD_SIF}" != "1" ]]; then
    if ssh -o StrictHostKeyChecking=no "root@${G5K_NODE}" "test -f '${sif}'"; then
      log "[${approach}] hcpa.sif ja existe; pulando (use FORCE_REBUILD_SIF=1 para forcar)"
      return 0
    fi
  fi

  log "[${approach}] singularity build hcpa.sif Apptainer.def (loga em ${LOG_DIR}/build_${approach}.log)"
  ssh -o StrictHostKeyChecking=no "root@${G5K_NODE}" \
    "cd '${apath}' && singularity build --force hcpa.sif Apptainer.def && chown ${HOST_UID}:${HOST_GID} hcpa.sif" \
    2>&1 | tee -a "${LOG_DIR}/build_${approach}.log"
  ssh -o StrictHostKeyChecking=no "${LOGIN_USER}@${G5K_NODE}" "ls -lh '${sif}'" || true
}

if [[ "${SKIP_BUILD}" == "1" ]]; then
  log "SKIP_BUILD=1; pulando build das imagens"
else
  for approach in ${APPROACHES}; do
    build_sif_for "${approach}"
  done
fi

# ----------- 7) grava state.env -----------
EPOCH_START="$(date +%s)"
WALL_SECS="$(echo "${WALLTIME}" | awk -F: '{print ($1*3600)+($2*60)+$3}')"
EPOCH_END=$(( EPOCH_START + WALL_SECS ))

cat > "${STATE_FILE}" <<EOF
# Gerado por g5k_setup_node.sh em $(date -Iseconds)
OAR_JOB_ID=${OAR_JOB_ID}
G5K_NODE=${G5K_NODE}
G5K_NODE_FILE=${UNIQ}
PROJECT_DIR=${PROJECT_DIR}
DEPLOY_IMAGE=${DEPLOY_IMAGE}
JOB_START_EPOCH=${EPOCH_START}
JOB_END_EPOCH=${EPOCH_END}
WALLTIME=${WALLTIME}
LOGIN_USER=${LOGIN_USER}
HOST_UID=${HOST_UID}
HOST_GID=${HOST_GID}
EOF

log "state gravado em ${STATE_FILE}"
log "PRONTO. Para rodar uma abordagem:"
log "  cd ${PROJECT_DIR}/pytorch_opt && bash run_g5k_hydra.sh"
log ""
log "Resumo:"
log "  OAR_JOB_ID = ${OAR_JOB_ID}"
log "  Node       = ${G5K_NODE}"
log "  Expira em  = $(date -d "@${EPOCH_END}" 2>/dev/null || echo "+${WALL_SECS}s")"
