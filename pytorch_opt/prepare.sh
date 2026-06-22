#!/usr/bin/env bash
set -euo pipefail

# Build the PyTorch image on a list of nodes (Grid5000 / Slurm).
# Usage: NODELIST=nodeA,nodeB [PARTITION=tupi] [TIME_LIMIT=00:20:00] ./prepare.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_X86="distributed_x86:latest"
IMAGE_ARM="distributed_arm:latest"
BASE_X86="${BASE_X86:-}"
BASE_ARM="${BASE_ARM:-}"
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

NODELIST="${NODELIST:-${1:-}}"
if [[ -z "${NODELIST}" ]]; then
  echo "Use: NODELIST=node1,node2 [PARTITION=...] [TIME_LIMIT=00:20:00] ./prepare.sh" >&2
  exit 1
fi
TIME_LIMIT="${TIME_LIMIT:-00:20:00}"
PARTITION="${PARTITION:-}"
ACCOUNT="${ACCOUNT:-}"
SRUN_EXTRA="${SRUN_EXTRA:-}"

IFS=',' read -ra NODES <<<"${NODELIST}"
NTASKS="${#NODES[@]}"

SRUN_OPTS=(--nodelist "${NODELIST}" --nodes "${NTASKS}" --ntasks "${NTASKS}" --ntasks-per-node 1 --time "${TIME_LIMIT}")
[[ -n "${PARTITION}" ]] && SRUN_OPTS+=(--partition "${PARTITION}")
[[ -n "${ACCOUNT}" ]] && SRUN_OPTS+=(--account "${ACCOUNT}")
if [[ -n "${SRUN_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  SRUN_OPTS+=(${SRUN_EXTRA})
fi

echo "[INFO] Building images on: ${NODELIST} (tasks=${NTASKS} time=${TIME_LIMIT}${PARTITION:+ part=${PARTITION}})"

export IMAGE_X86 IMAGE_ARM BASE_X86 BASE_ARM HOST_UID HOST_GID BUILD_DIR="${SCRIPT_DIR}"
srun "${SRUN_OPTS[@]}" bash -s <<'EOS'
set -euo pipefail
arch="$(uname -m)"
case "${arch}" in
  x86_64) image="${IMAGE_X86}"; base="${BASE_X86}";;
  aarch64|arm64) image="${IMAGE_ARM}"; base="${BASE_ARM}";;
  *) echo "[ERROR] unsupported arch ${arch}" >&2; exit 1;;
esac

command -v docker >/dev/null 2>&1 || { echo "[ERROR] docker not available on $(hostname)" >&2; exit 2; }

cd "${BUILD_DIR}"

if docker image inspect "${image}" >/dev/null 2>&1; then
  echo "[SKIP] $(hostname) already has ${image}"
  exit 0
fi

args=(--build-arg UID="${HOST_UID}" --build-arg GID="${HOST_GID}")
if [[ -n "${base}" ]]; then
  args+=(--build-arg BASE_IMAGE="${base}")
fi

echo "[BUILD] $(hostname) arch=${arch} -> ${image} (base=${base:-<default>})"
DOCKER_BUILDKIT=1 docker build "${args[@]}" -t "${image}" .
EOS
