#!/usr/bin/env bash
set -euo pipefail

# Build all images that ainda não foram provados na Tupi.
# Padrão: usa o nó tupi4 e partição tupi; ajustável via vars.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARTITION="${PARTITION:-tupi}"
NODELIST="${NODELIST:-${1:-tupi4}}"
TIME_LIMIT="${TIME_LIMIT:-00:40:00}"
ACCOUNT="${ACCOUNT:-}"
SRUN_EXTRA="${SRUN_EXTRA:-}"

if [[ -z "${NODELIST}" ]]; then
  echo "Use: NODELIST=tupi4 [PARTITION=tupi] [TIME_LIMIT=00:40:00] $0" >&2
  exit 1
fi

if ! command -v srun >/dev/null 2>&1; then
  echo "[ERRO] srun não encontrado no PATH." >&2
  exit 1
fi

# Ordem garante base TF antes das demais imagens.
TARGETS=(
  tensorflow_base
  pytorch_base
  tensorflow_opt
  pytorch_opt
  monai_base
  monai_opt
)

echo "[INFO] Nó(s): ${NODELIST} | partição: ${PARTITION} | tempo: ${TIME_LIMIT}"

for dir in "${TARGETS[@]}"; do
  prep="${ROOT_DIR}/${dir}/prepare.sh"
  if [[ ! -x "${prep}" ]]; then
    echo "[WARN] ${prep} não encontrado ou sem permissão de execução; pulando."
    continue
  fi
  echo "[INFO] Construindo ${dir} em ${NODELIST}..."
  (
    cd "${ROOT_DIR}/${dir}"
    NODELIST="${NODELIST}" \
    PARTITION="${PARTITION}" \
    TIME_LIMIT="${TIME_LIMIT}" \
    ACCOUNT="${ACCOUNT}" \
    SRUN_EXTRA="${SRUN_EXTRA}" \
    ./prepare.sh
  )
done

echo "[INFO] Build concluído para todos os alvos."
