#!/usr/bin/env bash
###############################################################################
# tools/sync_shared_libs.sh
#
# Copia as bibliotecas compartilhadas da RAIZ para cada pasta de abordagem.
#
# Por que a cópia existe: dentro do container o mount é a pasta da abordagem, e
# `_SELF_DIR` (o diretório do próprio script) é o primeiro caminho do sys.path.
# Ter o módulo ao lado do script é o que torna o import robusto ao mount.
#
# Por que este script existe: já aconteceu de editar só a raiz e as 9 cópias
# ficarem para trás — os runs continuam usando a versão velha, em silêncio.
# Rode SEMPRE depois de mexer em gpu_energy.py ou gpu_kernel_profile.py, e
# antes de tools/g5k_send_to_grid.sh.
#
#   bash tools/sync_shared_libs.sh          # copia
#   bash tools/sync_shared_libs.sh --check  # só verifica (rc=1 se divergir)
###############################################################################
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIBS=(gpu_energy.py gpu_kernel_profile.py)
APPROACHES=(pytorch_base pytorch_opt tensorflow_base tensorflow_opt
            hybrid_simple hybrid_token_reduction hybrid_token_reduction_opt
            retfound_green vit_pure)

CHECK=0
[[ "${1:-}" == "--check" ]] && CHECK=1

rc=0
for lib in "${LIBS[@]}"; do
  src="${ROOT}/${lib}"
  [[ -f "${src}" ]] || { echo "FALTA na raiz: ${lib}"; exit 1; }
  for a in "${APPROACHES[@]}"; do
    dst="${ROOT}/${a}/${lib}"
    if [[ -f "${dst}" ]] && cmp -s "${src}" "${dst}"; then
      continue
    fi
    if [[ "${CHECK}" == "1" ]]; then
      echo "DIVERGE: ${a}/${lib}"
      rc=1
    else
      cp "${src}" "${dst}"
      echo "sync -> ${a}/${lib}"
    fi
  done
done

if [[ "${CHECK}" == "1" ]]; then
  [[ "${rc}" == "0" ]] && echo "todas as cópias em dia"
  exit "${rc}"
fi
echo "ok: ${#LIBS[@]} libs x ${#APPROACHES[@]} abordagens"
