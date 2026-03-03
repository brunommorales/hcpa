#!/usr/bin/env bash
#SBATCH --job-name=build_docker_distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1          # 1 processo por nó (ajuste se quiser +GPUs por nó)
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=poti
#SBATCH --nodelist=poti1,poti2,poti3,poti4
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euo pipefail
# Imprime informações sobre o job
echo "Iniciando o job de build do Docker"
echo "Job ID: $SLURM_JOB_ID"
echo "Executando no nó: $(hostname)"
echo "Data: $(date)"
echo ""

# 1. Mudar para o diretório do projeto
# O 'cd' é essencial para o 'docker build .' encontrar o Dockerfile
echo "Mudando para o diretório /home/users/bmmorales/projects/hcpa/pytorch_opt"
cd /home/users/bmmorales/projects/hcpa/pytorch_opt

# Verifica se o diretório existe e foi acessado
if [ $? -ne 0 ]; then
  echo "Erro: Não foi possível acessar o diretório."
  echo "Verifique se o caminho está correto."
  exit 1
fi

echo "Diretório atual: $(pwd)"
echo ""

# 2. Executar docker build em TODOS os nós alocados, em paralelo
IMAGE_TAG="hcpa_nvidia"
NNODES="${SLURM_JOB_NUM_NODES:-1}"

echo "Iniciando build paralelo em ${NNODES} nó(s): ${SLURM_NODELIST}"

# Sanity check: lista os hosts alocados e o hostname reportado por cada tarefa
echo "Hosts alocados:" 
scontrol show hostnames "${SLURM_NODELIST}"
echo "Hostname reportado por tarefa (srun):"
srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label hostname

# Executa 1 tarefa por nó; em cada nó faz o build se a imagem não existir
srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label bash -lc \
  "cd /home/users/bmmorales/projects/hcpa/pytorch_opt && \
   echo '[INFO]' \"\$(hostname)\": verificando imagem ${IMAGE_TAG} && \
   if docker image inspect ${IMAGE_TAG} >/dev/null 2>&1; then \
      echo '[INFO]' \"\$(hostname)\": imagem já presente, pulando build; \
   else \
      echo '[INFO]' \"\$(hostname)\": construindo imagem ${IMAGE_TAG}; \
      docker build -t ${IMAGE_TAG} .; \
   fi"

# Verificação final: garante que a imagem existe em todos os nós
srun -N "${NNODES}" -n "${NNODES}" --ntasks-per-node=1 --label bash -lc \
  "if docker image inspect ${IMAGE_TAG} >/dev/null 2>&1; then \
      echo '[VERIFY]' \"\$(hostname)\": OK imagem ${IMAGE_TAG}; \
    else \
      echo '[VERIFY]' \"\$(hostname)\": MISSING imagem ${IMAGE_TAG}; exit 2; \
    fi"

echo "Build do Docker concluído (verifique logs por nó acima)."

echo "Job finalizado."
