#!/usr/bin/env bash
###############################################################################
# g5k_run_common.sh - Funcoes compartilhadas por run_g5k_hydra.sh de cada
#                     abordagem do HCPA.
#
# Como usar (em uma run_g5k_hydra.sh):
#   APPROACH=pytorch_opt
#   ENTRY=dr_hcpa_v2_2024.py
#   RESULTS_FLAG=--results       # ou --results_dir
#   EXEC_FLAG=--exec              # ou --exec_id
#   SEED_FLAG=--seed              # vazio se a entry nao aceita --seed
#   TRAIN_STATIC_ARGS=( ... )     # args fixos (sem run-id e sem path)
#   source "$(dirname "$0")/../tools/g5k_run_common.sh"
#   g5k_run_all
#
# Variaveis de ambiente reconhecidas:
#   RUN_START / RUN_END           default 0 / 9
#   BASE_SEED                     default 42
#   PROJECT_DIR                   default lido do state.env
#   STATE_FILE                    default ~/.g5k_hcpa/state.env
#   SIF_NAME                      default hcpa.sif
#   DATA_REL                      default data/all-tfrec-v2 (split val* por paciente)
#   DATASET_NAME                  default all-tfrec-v2
#   CONT_WORKDIR                  default /workspace
#   GPU_TAG                       default extraido por nvidia-smi
###############################################################################

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./g5k_node_state.sh
source "${TOOLS_DIR}/g5k_node_state.sh"

: "${APPROACH:?defina APPROACH antes de chamar g5k_run_all}"
: "${ENTRY:?defina ENTRY (script python a rodar)}"
: "${RESULTS_FLAG:?defina RESULTS_FLAG (--results ou --results_dir)}"
: "${EXEC_FLAG:?defina EXEC_FLAG (--exec ou --exec_id)}"
SEED_FLAG="${SEED_FLAG:-}"           # vazio se a entry nao aceita seed
RUN_START="${RUN_START:-0}"
RUN_END="${RUN_END:-9}"
BASE_SEED="${BASE_SEED:-42}"
SIF_NAME="${SIF_NAME:-hcpa.sif}"
# all-tfrec-v2 = split POR PACIENTE com val* disjunto do test* (tools/make_val_split.py).
# NAO usar data/all-tfrec (legado): nao tem shards val*, o que dispara o fallback
# val->test e reintroduz o vazamento P1 (best-checkpoint selecionado no teste).
DATA_REL="${DATA_REL:-data/all-tfrec-v2}"
DATASET_NAME="${DATASET_NAME:-all-tfrec-v2}"
CONT_WORKDIR="${CONT_WORKDIR:-/workspace}"
GPU_TAG_OVERRIDE="${GPU_TAG:-}"

# O estudo e SINGLE-GPU por padrao. Em nos com varias GPUs (chuc tem 4x A100), o
# TensorFlow faz `if len(gpus) > 1: MirroredStrategy` e treinaria em 4 GPUs com
# batch global 4x96, enquanto o pytorch_* (sem torchrun) usaria 1 — e o NVML
# mediria a energia de UMA GPU. Isso invalidaria a comparacao. Em hydra (1
# GPU/no) o bug era invisivel. Fixar a visibilidade e o que torna o experimento
# identico entre clusters. HCPA_VISIBLE_GPUS=0,1 para o estudo de escalabilidade
# (weak scaling): TF usa MirroredStrategy automaticamente (len(gpus)>1); PyTorch
# precisa de TORCHRUN_NPROC (abaixo) — sem ele so usaria a GPU 0.
HCPA_VISIBLE_GPUS="${HCPA_VISIBLE_GPUS:-0}"

# TORCHRUN_NPROC: vazio (default) = `python3 ENTRY ...` (comportamento legado,
# inalterado). Se setado (ex.: 2), lanca via `torchrun --standalone
# --nproc_per_node=N ENTRY ...` — DDP multi-GPU num node so (pytorch_opt/
# pytorch_base). Nao afeta TensorFlow (MirroredStrategy nao usa torchrun).
TORCHRUN_NPROC="${TORCHRUN_NPROC:-}"

# HCPA_NUMA_NODE: vazio (default) = sem pin (legado, inalterado). Se setado
# (0 ou 1), o processo INTEIRO (singularity exec + tudo dentro, incl. workers
# de decode/augment da CPU) roda preso a esse NUMA node via `numactl
# --cpunodebind=N --membind=N`. Existe para rodar 2 abordagens EM PARALELO no
# mesmo node multi-GPU (ex.: pytorch_opt na GPU0 + tensorflow_opt na GPU1, fase
# 1-GPU do estudo de escalabilidade) sem contencao de CPU/memoria entre elas:
# `nvidia-smi topo -m` mostra GPU0<->cores 0-31,64-95 (NUMA0) e GPU1<->cores
# 32-63,96-127 (NUMA1) no grouille; sem pin, o scheduler do Linux pode espalhar
# os workers de decode de UM framework pelos cores do NUMA node do OUTRO,
# competindo por CPU e adicionando latencia de memoria cross-socket (~3.2x mais
# lenta que local, medido: node distance 32 vs 10). Energia/util de GPU (NVML)
# JA SAO isoladas por hardware por device — o pin nao afeta essas metricas,
# so a limpeza do lado CPU/decode/augment.
HCPA_NUMA_NODE="${HCPA_NUMA_NODE:-}"

# carrega state.env (define G5K_NODE, OAR_JOB_ID, PROJECT_DIR)
g5k_load_state

APPROACH_DIR="${PROJECT_DIR}/${APPROACH}"
SIF_PATH="${APPROACH_DIR}/${SIF_NAME}"
TFREC_DIR="${PROJECT_DIR}/${DATA_REL}"
# Diretorio de dados efetivamente montado no container. Por padrao = NFS, mas
# g5k_precheck faz staging para o disco local do node (/tmp NVMe) e sobrescreve
# esta var. Ler do NFS a cada epoch deixa a GPU faminta (I/O-bound); o dataset
# (~1.6GB) cabe folgado no NVMe local e no page cache de 480GB de RAM.
TFREC_DIR_EFFECTIVE="${TFREC_DIR}"
# STAGE_DATA_LOCAL=0 desliga o staging (volta a ler do NFS).
STAGE_DATA_LOCAL="${STAGE_DATA_LOCAL:-1}"
RESULTS_DIR="${APPROACH_DIR}/results"
LOG_DIR="${APPROACH_DIR}/logs"

log()  { printf '[%s][%s] %s\n' "${APPROACH}" "$(date +'%H:%M:%S')" "$*"; }
fail() { printf '[%s][FAIL] %s\n' "${APPROACH}" "$*" >&2; exit 1; }

sanitize_tag() {
  local s="${1:-}"
  s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]')"
  if [[ -z "${s}" || "${s}" == "(null)" || "${s}" == "null" ]]; then
    printf '%s' 'unknown'; return
  fi
  s="$(printf '%s' "${s}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  s="$(printf '%s' "${s}" | sed -E 's/-+/-/g')"
  printf '%s' "${s}"
}

G5K_REMOTE_USER="${G5K_REMOTE_USER:-root}"
LOGIN_USER="${LOGIN_USER:-${USER:-bmorales}}"
HOST_UID="${HOST_UID:-$(id -u 2>/dev/null || echo 1000)}"
HOST_GID="${HOST_GID:-$(id -g 2>/dev/null || echo 1000)}"

g5k_ssh() {
  ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
      "${G5K_REMOTE_USER}@${G5K_NODE}" "$@"
}

g5k_precheck() {
  log "checando node ${G5K_NODE}..."
  g5k_ssh "test -f '${SIF_PATH}'" \
    || fail "imagem ausente em ${G5K_NODE}:${SIF_PATH}. Rode tools/g5k_setup_node.sh com APPROACHES='${APPROACH}'"
  g5k_ssh "test -d '${TFREC_DIR}'" \
    || fail "dataset ausente em ${G5K_NODE}:${TFREC_DIR}. Faca o rsync de ${DATA_REL}."
  # GUARDA ANTI-VAZAMENTO (P1): sem shards val*, o codigo cai no fallback
  # val->valid->test (resolve_split_files) e o best-checkpoint passa a ser
  # selecionado NO PROPRIO TESTE. Falhar alto em vez de vazar em silencio.
  g5k_ssh "ls '${TFREC_DIR}'/val*.tfrec >/dev/null 2>&1" \
    || fail "dataset ${DATA_REL} nao tem split val* -> usaria val==test (vazamento P1). Use DATA_REL=data/all-tfrec-v2."
  g5k_ssh "nvidia-smi --query-gpu=name --format=csv,noheader | head -1" \
    || fail "nvidia-smi falhou em ${G5K_NODE}"

  if [[ -z "${GPU_TAG_OVERRIDE}" ]]; then
    GPU_NAME_RAW="$(g5k_ssh "nvidia-smi --query-gpu=name --format=csv,noheader | head -1" 2>/dev/null || echo "gpu")"
    GPU_TAG_OVERRIDE="$(sanitize_tag "${GPU_NAME_RAW}")"
  fi
  export GPU_TAG="${GPU_TAG_OVERRIDE}"
  # RESULTS_TAG nomeia a pasta de resultados. Por padrão é o GPU_TAG (produção).
  # A varredura de LR o sobrepõe (ex.: RESULTS_TAG=lrsweep_1e-3) para não
  # sobrescrever os runs de produção.
  export RESULTS_TAG="${RESULTS_TAG:-${GPU_TAG}}"

  mkdir -p "${LOG_DIR}"
  g5k_ssh "mkdir -p '${RESULTS_DIR}' '${LOG_DIR}'"

  g5k_stage_data
}

# Copia o dataset do NFS para o disco local do node (/tmp NVMe) uma vez por job.
# Mirror do que o distributed_run_arm.slurm fazia com stage_shared_tfrec_to_ssd:
# ler TFRecords do NFS a cada epoch tornava a GPU I/O-bound (~750 img/s, util
# <40%); servindo do NVMe local + page cache a GPU fica compute-bound (~4000).
g5k_stage_data() {
  if [[ "${STAGE_DATA_LOCAL}" != "1" ]]; then
    log "staging local desativado (STAGE_DATA_LOCAL=0); lendo dataset do NFS"
    TFREC_DIR_EFFECTIVE="${TFREC_DIR}"
    return 0
  fi
  local local_dir="/tmp/hcpa_data_${OAR_JOB_ID}/${DATASET_NAME}"
  log "staging dataset NFS -> ${G5K_NODE}:${local_dir} (uma vez por job)..."
  if g5k_ssh "mkdir -p '${local_dir}' && rsync -a --ignore-existing '${TFREC_DIR}/' '${local_dir}/' && test -n \"\$(ls -A '${local_dir}' 2>/dev/null)\""; then
    TFREC_DIR_EFFECTIVE="${local_dir}"
    local nfiles
    nfiles="$(g5k_ssh "ls '${local_dir}' | wc -l" 2>/dev/null || echo '?')"
    log "staging ok: ${nfiles} arquivos em disco local; container lera de ${local_dir}"
  else
    log "staging FALHOU; fallback para NFS ${TFREC_DIR}"
    TFREC_DIR_EFFECTIVE="${TFREC_DIR}"
  fi
}

g5k_run_one() {
  local run_id="$1"
  local run_seed=$(( BASE_SEED + run_id ))
  # cluster real derivado do hostname (chuc-7.lille... -> chuc; hydra-1.lyon... -> hydra),
  # senao os resultados de OUTROS clusters (ex: chuc/A100) sairiam rotulados "hydra".
  local cl_suffix="${G5K_CLUSTER:-$(printf '%s' "${G5K_NODE}" | sed -E 's/-[0-9]+\..*//')}"
  [[ -n "${cl_suffix}" ]] || cl_suffix="hydra"
  local run_results_subdir="results/${RESULTS_TAG}_g5k_${cl_suffix}/run_${run_id}"
  local run_results_host="${APPROACH_DIR}/${run_results_subdir}"
  local run_results_cont="${CONT_WORKDIR}/${run_results_subdir}"
  local run_log="${LOG_DIR}/g5k_${APPROACH}_job${OAR_JOB_ID}_run${run_id}.log"

  log "RUN ${run_id} seed=${run_seed} -> ${run_results_host}"
  g5k_ssh "mkdir -p '${run_results_host}'"

  # monta argumentos do treino (sem reescrever os do chamador)
  local -a run_args=(
    "${RESULTS_FLAG}" "${run_results_cont}"
    "${EXEC_FLAG}" "${run_id}"
  )
  if [[ -n "${SEED_FLAG}" ]]; then
    run_args+=("${SEED_FLAG}" "${run_seed}")
  fi
  run_args+=( --tfrec_dir "${CONT_WORKDIR}/${DATA_REL}" )
  if [[ "${TRAIN_ACCEPTS_DATASET:-1}" == "1" ]]; then
    run_args+=( --dataset "${DATASET_NAME}" )
  fi
  run_args+=( "${TRAIN_STATIC_ARGS[@]}" )
  # EXTRA_TRAIN_ARGS: appenda DEPOIS de TRAIN_STATIC_ARGS. argparse e last-wins
  # para flags repetidas (ex.: --epochs), entao isto sobrescreve valores fixos
  # do script sem precisar editar TRAIN_STATIC_ARGS. Uso: smoke test rapido
  # (EXTRA_TRAIN_ARGS="--epochs 2"), ablacoes pontuais, etc. Vazio = no-op.
  if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local -a _extra=( ${EXTRA_TRAIN_ARGS} )
    run_args+=( "${_extra[@]}" )
  fi

  # Constroi argv como uma string segura. Cada arg %q-escapado.
  local printed_args
  printed_args="$(printf ' %q' "${run_args[@]}")"

  # Coleta --env extras de vars opcionais exportadas pela abordagem (ex: TF_*).
  # As HCPA_* precisam estar aqui: sem isso a telemetria e o profiler de kernel
  # nunca veem as manoplas (elas ficariam só no host, fora do container).
  local extra_env_block=""
  for _evar in TF_GPU_ALLOCATOR TF_FORCE_GPU_ALLOW_GROWTH TF_ENABLE_GPU_GC \
               TF_NUM_INTEROP_THREADS TF_CPP_MIN_LOG_LEVEL DALI_LOG \
               PT_DISABLE_COMPILE HCPA_COMPILE_MODE HCPA_CACHE_IN_MEMORY \
               HCPA_ENERGY_ALL_VISIBLE_GPUS \
               HCPA_GPU_SAMPLE_MS HCPA_PROFILE_EPOCHS HCPA_PROFILE_DIR; do
    local _eval="${!_evar:-}"
    [[ -n "${_eval}" ]] && extra_env_block+="  --env ${_evar}=${_eval} \\"$'\n'
  done

  # Singularity exec com bind do projeto e dos tmp/caches em /tmp do node
  local tmp_root="/tmp/hcpa_${OAR_JOB_ID}_${APPROACH}_r${run_id}"
  local cache_root="${tmp_root}/cache"
  # Cache do torch.compile (TorchInductor/Triton) PERSISTENTE entre runs da MESMA
  # abordagem no MESMO job (nao leva _r${run_id} no nome, nao e removido pelo
  # cleanup por-run). Sem isso, cada um dos 10 runs recompilava do zero: no
  # smoke com DDP (2 ranks compilando em paralelo, cache Triton frio) o 1o
  # compile passou de 10min. So o 1o run de cada approach paga o custo; os
  # demais 9 reusam os kernels (mesma config: mesmo modelo/batch/GPU, so muda
  # a seed). Some sozinho quando o job termina (node e ephemeral).
  local torch_cache_root="/tmp/hcpa_${OAR_JOB_ID}_${APPROACH}_torchcache"
  local remote_script_dir="${APPROACH_DIR}/.g5k_run_scripts"
  local remote_script="${remote_script_dir}/run_${run_id}.sh"

  mkdir -p "${remote_script_dir}" 2>/dev/null || true
  # Prefixo do lancamento: torchrun (DDP multi-GPU, --standalone = rendezvous
  # local, sem precisar de MASTER_ADDR/PORT) se TORCHRUN_NPROC setado, senao
  # python3 direto (legado, single-processo). So troca o prefixo — o resto da
  # linha (aspas do entry path + printed_args) fica EXATAMENTE como antes.
  local py_launch_prefix="python3"
  if [[ -n "${TORCHRUN_NPROC}" ]]; then
    py_launch_prefix="torchrun --standalone --nproc_per_node=${TORCHRUN_NPROC}"
  fi
  # Pin de NUMA (ver comentario de HCPA_NUMA_NODE acima). Prende o processo
  # INTEIRO (singularity + tudo dentro) ao node de memoria/cpu local da GPU.
  local numa_prefix=""
  if [[ -n "${HCPA_NUMA_NODE}" ]]; then
    numa_prefix="numactl --cpunodebind=${HCPA_NUMA_NODE} --membind=${HCPA_NUMA_NODE} "
  fi
  # Escreve o script local (no NFS compartilhado, ja visivel no node)
  cat > "${remote_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
mkdir -p '${tmp_root}/mplconfig' '${cache_root}/hf' '${cache_root}/torch' '${torch_cache_root}/torchinductor' '${torch_cache_root}/triton'
cd '${APPROACH_DIR}'
exec ${numa_prefix}singularity exec --nv \\
  --bind '${APPROACH_DIR}:${CONT_WORKDIR}' \\
  --bind '${PROJECT_DIR}:/hcpa:ro' \\
  --bind '${PROJECT_DIR}/tools:${CONT_WORKDIR}/tools:ro' \\
  --bind '${PROJECT_DIR}/hybrid_shared:${CONT_WORKDIR}/hybrid_shared:ro' \\
  --bind '${TFREC_DIR_EFFECTIVE}:${CONT_WORKDIR}/${DATA_REL}:ro' \\
  --bind '${tmp_root}:/tmp/hcpa' \\
  --bind '${torch_cache_root}:/tmp/hcpa_persist' \\
  --env CUDA_VISIBLE_DEVICES=${HCPA_VISIBLE_GPUS} \\
  --env KERAS_BACKEND=${KERAS_BACKEND:-torch} \\
  --env PYTHONPATH=${CONT_WORKDIR}:/hcpa \\
  --env PYTHONUNBUFFERED=1 \\
  --env HOME=${CONT_WORKDIR} \\
  --env TMPDIR=/tmp/hcpa \\
  --env TMP=/tmp/hcpa \\
  --env TEMP=/tmp/hcpa \\
  --env MPLCONFIGDIR=/tmp/hcpa/mplconfig \\
  --env XDG_CACHE_HOME=/tmp/hcpa/cache \\
  --env HF_HOME=/tmp/hcpa/cache/hf \\
  --env TRANSFORMERS_CACHE=/tmp/hcpa/cache/hf \\
  --env TORCH_HOME=/tmp/hcpa/cache/torch \\
  --env TORCHINDUCTOR_CACHE_DIR=/tmp/hcpa_persist/torchinductor \\
  --env TRITON_CACHE_DIR=/tmp/hcpa_persist/triton \\
  --env NCCL_DEBUG=${NCCL_DEBUG:-WARN} \\
  --env CUDA_LAUNCH_BLOCKING=0 \\
  --env TRITON_LIBCUDA_PATH=/.singularity.d/libs \\
  --env LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/compat/lib.real:/usr/local/cuda/lib64 \\
${extra_env_block}  '${SIF_PATH}' \\
  bash -c '[ -f /etc/shinit_v2 ] && source /etc/shinit_v2 >/dev/null 2>&1; exec ${py_launch_prefix} '\''${CONT_WORKDIR}/${ENTRY}'\''${printed_args}'
EOF
  chmod +x "${remote_script}"
  # Executa no node (nao usa -lc pra evitar dump de /etc/profile do Ubuntu)
  g5k_ssh "bash '${remote_script}'" 2>&1 | tee "${run_log}"

  # limpa loggers temporarios (tensorboard, wandb, etc)
  local clean_helper="${PROJECT_DIR}/clean_temp_loggers.py"
  if g5k_ssh "test -f '${clean_helper}'"; then
    g5k_ssh "python3 '${clean_helper}' --root '${run_results_host}' --scope results-only --max-print 0" \
      || log "clean_temp_loggers falhou (nao critico)"
  fi
  # WHITELIST final: mantem APENAS *.csv e *.pdf no run. Remove checkpoints/
  # (best.ckpt/last.ckpt do TF, ~460MB/run), mplconfig/, *.index, json/txt e
  # qualquer cache que o clean_temp_loggers nao tenha pego. Evita estourar a
  # quota do home no G5K (.sif ja consome ~25GB).
  # Preserva *.json: sao os traces de kernel do CUPTI (kprof-*/epoch_N.json).
  # Sem esta excecao a limpeza apagava exatamente o que o piano roll consome.
  g5k_ssh "find '${run_results_host}' -type f ! -name '*.csv' ! -name '*.pdf' ! -name '*.json' -delete 2>/dev/null; find '${run_results_host}' -mindepth 1 -type d -empty -delete 2>/dev/null" \
    || log "whitelist purge falhou (nao critico)"
  # garante que os resultados ficam acessiveis para o usuario via NFS
  g5k_ssh "chown -R ${HOST_UID}:${HOST_GID} '${run_results_host}' 2>/dev/null || true" || true
  # remove tmp do run (cache enorme, nao precisa preservar)
  g5k_ssh "rm -rf '${tmp_root}'" || true
}

g5k_run_all() {
  g5k_precheck
  log "iniciando faixa runs=${RUN_START}..${RUN_END} no node=${G5K_NODE} job=${OAR_JOB_ID}"
  for rid in $(seq "${RUN_START}" "${RUN_END}"); do
    g5k_run_one "${rid}"
  done
  log "concluido. Resultados em ${G5K_NODE}:${RESULTS_DIR}/${RESULTS_TAG}_g5k_hydra/run_*"
}
