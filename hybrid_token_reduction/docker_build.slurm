#!/usr/bin/env bash
#SBATCH --job-name=build_hybrid_token_reduction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=poti
#SBATCH --nodelist=poti5
#SBATCH --output=/home/users/bmmorales/projects/hcpa/hybrid_token_reduction/logs/%x_%j_%N.out
#SBATCH --error=/home/users/bmmorales/projects/hcpa/hybrid_token_reduction/logs/%x_%j_%N.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/users/bmmorales/projects/hcpa/hybrid_token_reduction}"
IMAGE_TAG="${IMAGE_TAG:-hybrid_token_reduction:latest}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
HOST_UID="${HOST_UID:-$(id -u)}"
HOST_GID="${HOST_GID:-$(id -g)}"
BUILD_ARGS="${BUILD_ARGS:-} --build-arg UID=${HOST_UID} --build-arg GID=${HOST_GID}"
NNODES="${SLURM_JOB_NUM_NODES:-1}"

echo "Iniciando build do Docker"
echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Projeto: ${PROJECT_DIR}"
echo "Imagem: ${IMAGE_TAG}"
echo "Force rebuild: ${FORCE_REBUILD}"
echo "Nós alocados: ${NNODES}"
echo "Nodelist: ${SLURM_NODELIST:-$(hostname)}"
echo "Data: $(date)"
echo ""

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "Erro: diretório do projeto não existe: ${PROJECT_DIR}"
  exit 1
fi

if [[ ! -f "${PROJECT_DIR}/Dockerfile" ]]; then
  echo "Erro: Dockerfile não encontrado em ${PROJECT_DIR}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: docker não está disponível no nó de submissão."
  exit 1
fi

echo "Hosts alocados:"
scontrol show hostnames "${SLURM_NODELIST}"
echo "Hostname reportado por tarefa:"
srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label hostname

export PROJECT_DIR IMAGE_TAG FORCE_REBUILD BUILD_ARGS

srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label bash -lc '
  set -euo pipefail
  cd "${PROJECT_DIR}"
  read -r -a EXTRA_BUILD_ARGS <<<"${BUILD_ARGS:-}"
  echo "[INFO] $(hostname): verificando imagem ${IMAGE_TAG}"
  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    echo "[INFO] $(hostname): rebuild forçado de ${IMAGE_TAG}"
    docker build "${EXTRA_BUILD_ARGS[@]}" -t "${IMAGE_TAG}" .
  elif docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "[INFO] $(hostname): imagem já presente, pulando build"
  else
    echo "[INFO] $(hostname): construindo imagem ${IMAGE_TAG}"
    docker build "${EXTRA_BUILD_ARGS[@]}" -t "${IMAGE_TAG}" .
  fi
'

srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label bash -lc '
  set -euo pipefail
  if docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
    echo "[VERIFY] $(hostname): OK imagem ${IMAGE_TAG}"
  else
    echo "[VERIFY] $(hostname): MISSING imagem ${IMAGE_TAG}"
    exit 2
  fi
'

echo "Build concluído com sucesso."
