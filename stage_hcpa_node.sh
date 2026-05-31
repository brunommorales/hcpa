#!/usr/bin/env bash
#SBATCH --job-name=hcpa_stage
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --partition=grace
#SBATCH --nodelist=grace1
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err
#SBATCH --chdir=/home/users/bmmorales/projects/hcpa

set -euo pipefail

HOME_ROOT_DIR="${HOME_ROOT_DIR:-/home/users/bmmorales/projects/hcpa}"
STORAGE="${STORAGE:-auto}"       # auto | ssd | scratch
DEST_ROOT="${DEST_ROOT:-}"       # caminho absoluto opcional

TARGETS=(
  tools
  hybrid_shared
  hybrid_simple
  hybrid_token_reduction
  hybrid_token_reduction_opt
  pytorch_base
  pytorch_opt
  tensorflow_base
  tensorflow_opt
  vit_pure
  retfound_green
  data/all-tfrec
)

resolve_dest_root() {
  local owner ssd_path scratch_path
  owner="${USER:-bmmorales}"
  ssd_path="/ssd/${owner}"
  scratch_path="/scratch/${owner}"

  if [[ -n "${DEST_ROOT}" ]]; then
    printf '%s' "${DEST_ROOT%/}"
    return 0
  fi

  case "${STORAGE}" in
    auto)
      if [[ -d "${ssd_path}" ]]; then
        printf '%s' "${ssd_path}"
      elif [[ -d "${scratch_path}" ]]; then
        printf '%s' "${scratch_path}"
      else
        echo "[FAIL] nenhum destino existe: ${ssd_path} ou ${scratch_path}" >&2
        return 2
      fi
      ;;
    ssd)
      if [[ -d "${ssd_path}" ]]; then
        printf '%s' "${ssd_path}"
      else
        echo "[FAIL] destino SSD nao existe: ${ssd_path}" >&2
        return 2
      fi
      ;;
    scratch)
      if [[ -d "${scratch_path}" ]]; then
        printf '%s' "${scratch_path}"
      else
        echo "[FAIL] destino scratch nao existe: ${scratch_path}" >&2
        return 2
      fi
      ;;
    *)
      echo "[FAIL] STORAGE invalido: ${STORAGE}. Use auto, ssd ou scratch." >&2
      return 2
      ;;
  esac
}

copy_one() {
  local src_root="$1"
  local dst_root="$2"
  local rel="$3"
  local src="${src_root}/${rel}"
  local dst="${dst_root}/${rel}"

  if [[ ! -d "${src}" ]]; then
    echo "[FAIL] origem nao existe: ${src}" >&2
    return 92
  fi

  if [[ "${rel}" == "data/all-tfrec" ]]; then
    if [[ ! -d "${dst_root}/data" ]]; then
      mkdir -p "${dst_root}/data"
    fi
    echo "[COPY] rsync -a --delete --partial ${src}/ -> ${dst}/"
    rsync -a --delete --partial "${src}/" "${dst}/"
    echo "[DONE] alvo copiado: ${rel}"
    return 0
  fi

  if [[ -d "${dst}" ]]; then
    echo "[COPY] cp -r ${src}/. ${dst}/"
    cp -r "${src}/." "${dst}/"
  else
    echo "[COPY] cp -r ${src} ${dst}"
    cp -r "${src}" "${dst}"
  fi

  echo "[DONE] alvo copiado: ${rel}"
}

verify_targets() {
  local dst_root="$1"
  local rel

  for rel in "${TARGETS[@]}"; do
    if [[ ! -d "${dst_root}/${rel}" ]]; then
      echo "[FAIL] alvo ausente apos copia: ${dst_root}/${rel}" >&2
      return 95
    fi
  done
}

main() {
  local src_root
  local dst_root
  local start_ts
  local end_ts
  local duration_s

  src_root="$(cd "${HOME_ROOT_DIR}" && pwd)"
  dst_root="$(resolve_dest_root)"
  start_ts="$(date +%s)"

  if [[ ! -d "${dst_root}" ]]; then
    echo "[FAIL] destino raiz nao existe: ${dst_root}" >&2
    echo "[FAIL] este script nao cria diretorios."
    exit 94
  fi

  echo "[INFO] job_id=${SLURM_JOB_ID:-no_slurm}"
  echo "[INFO] node=$(hostname)"
  echo "[INFO] source_root=${src_root}"
  echo "[INFO] dest_root=${dst_root}"

  local rel
  for rel in "${TARGETS[@]}"; do
    copy_one "${src_root}" "${dst_root}" "${rel}"
  done

  verify_targets "${dst_root}"

  end_ts="$(date +%s)"
  duration_s=$((end_ts - start_ts))
  echo "[DONE] todos os alvos foram copiados para ${dst_root}"
  echo "[DONE] finalizado em $(date '+%Y-%m-%d %H:%M:%S') (duracao=${duration_s}s)"
}

main "$@"
