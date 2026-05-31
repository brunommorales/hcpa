#!/usr/bin/env bash

hcpa_runtime_resolve_dir_path() {
  local dir_path="$1"
  if [[ -d "${dir_path}" ]]; then
    cd "${dir_path}" && pwd
  else
    printf '%s' "${dir_path}"
  fi
}

hcpa_runtime_resolve_fast_storage_base() {
  local explicit_base="${SSD_BASE_DIR:-}"
  local candidates=()

  if [[ -n "${explicit_base}" ]]; then
    candidates+=("${explicit_base}")
  fi
  candidates+=(/ssd/bmmorales /scratch/bmmorales)

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  if [[ -n "${explicit_base}" ]]; then
    printf '%s' "${explicit_base}"
  else
    printf '%s' '/scratch/bmmorales'
  fi
}

_hcpa_runtime_run_with_lock() {
  local lock_path="$1"
  shift
  mkdir -p "$(dirname "${lock_path}")"
  if command -v flock >/dev/null 2>&1; then
    exec 9>"${lock_path}"
    flock 9
    "$@"
    local status=$?
    flock -u 9 || true
    exec 9>&-
    return "${status}"
  fi
  "$@"
}

hcpa_runtime_stage_module_to_ssd() {
  local module_name="$1"
  local source_dir="$2"
  local ssd_base
  ssd_base="$(hcpa_runtime_resolve_fast_storage_base)"
  local target_dir="${ssd_base}/${module_name}"
  local lock_path="${ssd_base}/.stage_${module_name}.lock"

  if [[ ! -d "${source_dir}" ]]; then
    echo "[CHECK][FAIL] módulo de origem ausente para stage: ${source_dir}" >&2
    return 92
  fi

  if ! mkdir -p "${ssd_base}" 2>/dev/null; then
    echo "[SSD][WARN] sem permissão para preparar ${ssd_base}; usando ${source_dir} diretamente" >&2
    cd "${source_dir}" && pwd
    return 0
  fi
  echo "[SSD] sincronizando módulo ${module_name}: ${source_dir} -> ${target_dir}" >&2
  _hcpa_runtime_run_with_lock "${lock_path}" \
    rsync -a --delete \
      --exclude '.git/' \
      --exclude '.venv/' \
      --exclude '__pycache__/' \
      --exclude '*.pyc' \
      --exclude '.cache/' \
      --exclude 'tmp/' \
      --exclude 'logs/' \
      --exclude 'results/' \
      --exclude 'data/' \
      "${source_dir}/" "${target_dir}/" || return $?

  cd "${target_dir}" && pwd
}

hcpa_runtime_stage_shared_tfrec_to_ssd() {
  local source_dir="${1:-${HOME_ROOT_DIR:-${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}}/data/all-tfrec}"
  source_dir="${source_dir%/}"
  local stage_name="${2:-$(basename "${source_dir}")}"
  local ssd_base
  ssd_base="$(hcpa_runtime_resolve_fast_storage_base)"
  local stage_slug
  stage_slug="$(printf '%s' "${stage_name}" | sed -E 's/[^a-zA-Z0-9._-]+/_/g')"
  if [[ -z "${stage_slug}" ]]; then
    stage_slug="all-tfrec"
  fi
  local target_dir="${ssd_base}/data/${stage_slug}"
  local lock_path="${ssd_base}/.stage_${stage_slug}.lock"

  if [[ ! -d "${source_dir}" ]]; then
    echo "[CHECK][FAIL] TFRecords de origem ausentes para stage: ${source_dir}" >&2
    return 92
  fi

  if ! mkdir -p "${ssd_base}/data" 2>/dev/null; then
    echo "[SSD][WARN] sem permissão para preparar ${ssd_base}/data; usando ${source_dir} diretamente" >&2
    cd "${source_dir}" && pwd
    return 0
  fi
  echo "[SSD] sincronizando TFRecords: ${source_dir} -> ${target_dir}" >&2
  _hcpa_runtime_run_with_lock "${lock_path}" \
    rsync -a --delete --partial "${source_dir}/" "${target_dir}/" || return $?

  cd "${target_dir}" && pwd
}

hcpa_runtime_resolve_home_module_dir() {
  local module_name="$1"
  local persistent_root="${PERSISTENT_ROOT_DIR:-${HOME:-/home/users/bmmorales}/projects/hcpa}"
  local candidate="${HOME_PROJ_DIR:-}"
  if [[ -n "${candidate}" ]]; then
    :
  elif [[ -d "${persistent_root}/${module_name}" ]]; then
    candidate="${persistent_root}/${module_name}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" && "$(basename "${SLURM_SUBMIT_DIR}")" == "${module_name}" ]]; then
    candidate="${SLURM_SUBMIT_DIR}"
  else
    candidate="${persistent_root}/${module_name}"
  fi
  cd "${candidate}" && pwd
}

hcpa_runtime_resolve_execution_module_dir() {
  local module_name="$1"
  local fallback_dir="$2"
  local candidate="${EXEC_PROJ_DIR:-${HOST_PROJ_DIR:-}}"
  if [[ -n "${candidate}" ]]; then
    :
  elif [[ -d "$(hcpa_runtime_resolve_fast_storage_base)" || "${REQUIRE_SSD:-1}" == "1" ]]; then
    candidate="$(hcpa_runtime_stage_module_to_ssd "${module_name}" "${fallback_dir}")" || {
      echo "[SSD][WARN] stage do módulo falhou; usando ${fallback_dir}" >&2
      candidate="${fallback_dir}"
    }
  else
    candidate="${fallback_dir}"
  fi
  cd "${candidate}" && pwd
}

hcpa_runtime_init_paths() {
  local module_name="$1"
  HOME_PROJ_DIR="$(hcpa_runtime_resolve_home_module_dir "${module_name}")" || return $?
  HOST_PROJ_DIR="$(hcpa_runtime_resolve_execution_module_dir "${module_name}" "${HOME_PROJ_DIR}")" || return $?
  HOME_ROOT_DIR="$(cd "${HOME_PROJ_DIR}/.." && pwd)"
  HOST_ROOT_DIR="$(cd "${HOST_PROJ_DIR}/.." && pwd)"

  # Garante que ${HOST_ROOT_DIR}/tools sempre exista e contenha os helpers atuais,
  # mesmo quando o stage_hcpa_node.sh nao copiou (ou copiou versao antiga).
  if [[ "${HOST_ROOT_DIR}" != "${HOME_ROOT_DIR}" && -d "${HOME_ROOT_DIR}/tools" ]]; then
    local _tools_target="${HOST_ROOT_DIR}/tools"
    local _tools_lock="${HOST_ROOT_DIR}/.stage_tools.lock"
    if mkdir -p "${_tools_target}" 2>/dev/null; then
      _hcpa_runtime_run_with_lock "${_tools_lock}" \
        rsync -a --delete \
          --exclude '__pycache__/' \
          --exclude '*.pyc' \
          "${HOME_ROOT_DIR}/tools/" "${_tools_target}/" >&2 || true
    fi
  fi
}

hcpa_runtime_resolve_support_file() {
  local relative_path="$1"
  local explicit_path="$2"
  local candidate
  local candidates=()
  if [[ -n "${explicit_path}" ]]; then
    candidates+=("${explicit_path}")
  else
    candidates+=("${HOST_ROOT_DIR}/${relative_path}")
    candidates+=("${HOME_ROOT_DIR}/${relative_path}")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      cd "$(dirname "${candidate}")" && printf '%s/%s' "$(pwd)" "$(basename "${candidate}")"
      return 0
    fi
  done
  printf '%s' "${candidates[0]}"
}

hcpa_runtime_maybe_run_final_sync() {
  if [[ "${FINAL_SYNC:-0}" == "1" ]]; then
    echo "Executando sync final (FINAL_SYNC=1)."
    sync
  else
    echo "Pulando sync final (FINAL_SYNC=${FINAL_SYNC:-0}); usando apenas rsync para persistência."
  fi
}

hcpa_runtime_sync_results_to_home() {
  if [[ -z "${HOST_RESULTS_DIR:-}" || -z "${HOME_RESULTS_DIR:-}" ]]; then
    return 0
  fi
  if [[ "${HOST_RESULTS_DIR}" == "${HOME_RESULTS_DIR}" ]]; then
    echo "Resultados já estão em armazenamento persistente."
    hcpa_runtime_maybe_run_final_sync
    return 0
  fi
  if ! command -v rsync >/dev/null 2>&1; then
    echo "Erro: rsync não está disponível no nó."
    return 1
  fi
  mkdir -p "${HOME_RESULTS_DIR}"
  echo "Sincronizando artefatos finais de ${HOST_RESULTS_DIR} para ${HOME_RESULTS_DIR} (*.csv, *.pdf)"
  rsync -a --prune-empty-dirs \
    --include '*/' \
    --include '*.csv' \
    --include '*.pdf' \
    --exclude '*' \
    "${HOST_RESULTS_DIR}/" "${HOME_RESULTS_DIR}/"
  if [[ "${SYNC_LOGS_TO_HOME:-0}" == "1" && -d "${HOST_LOGS_DIR:-}" ]]; then
    mkdir -p "${HOME_LOGS_DIR}"
    echo "Sincronizando logs de ${HOST_LOGS_DIR} para ${HOME_LOGS_DIR}"
    rsync -a "${HOST_LOGS_DIR}/" "${HOME_LOGS_DIR}/"
  fi
  hcpa_runtime_maybe_run_final_sync
}

hcpa_runtime_cleanup_and_sync() {
  local exit_code="${1:-0}"
  local sync_exit_code=0
  local cleanup_helper="${CLEAN_TEMP_HELPER:-}"
  set +e
  if [[ -d "${HOST_RESULTS_DIR:-}" ]]; then
    if [[ -z "${cleanup_helper}" && -n "${HOME_ROOT_DIR:-}" && -f "${HOME_ROOT_DIR}/clean_temp_loggers.py" ]]; then
      cleanup_helper="${HOME_ROOT_DIR}/clean_temp_loggers.py"
    fi
    if [[ -f "${cleanup_helper}" ]]; then
      python3 "${cleanup_helper}" --root "${HOST_RESULTS_DIR}" --scope results-only --max-print 0 || true
    fi
    hcpa_runtime_sync_results_to_home
    sync_exit_code=$?
  fi
  if (( exit_code == 0 && sync_exit_code != 0 )); then
    return "${sync_exit_code}"
  fi
  return "${exit_code}"
}
