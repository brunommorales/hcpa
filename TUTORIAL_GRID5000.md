# Tutorial: Executando Jobs no Grid'5000

Este tutorial explica como migrar seus jobs SLURM para o Grid'5000, que usa **OAR** como gerenciador de recursos.

## ⚠️ IMPORTANTE: Arquitetura dos Clusters

| Cluster | CPU | Arquitetura | GPU | Memória |
|---------|-----|-------------|-----|---------|
| **hydra** | NVIDIA/ARM Grace | **aarch64 (ARM)** | 1x H200 (96 GiB) | 480 GiB |
| **sirius** | AMD EPYC 7742 | **x86_64** | 8x A100 (40 GiB) | 1 TiB |
| **gemini** | Intel Xeon E5-2698 | **x86_64** | 8x V100 (32 GiB) | 512 GiB |

> **⚡ O Hydra é ARM, não x86!** Use os scripts `*_arm.oar` para Hydra e `*_x86.oar` para Sirius/Gemini.

## 📋 Índice

1. [Conexão ao Grid'5000](#1-conexão-ao-grid5000)
2. [Reserva de Recursos (OAR)](#2-reserva-de-recursos-oar)
3. [Build do Docker no Nó](#3-build-do-docker-no-nó)
4. [Execução dos Jobs](#4-execução-dos-jobs)
5. [Scripts Prontos para Uso](#5-scripts-prontos-para-uso)
6. [Executar Todas as Abordagens HCPA](#6-executar-todas-as-abordagens-hcpa)

---

## 1. Conexão ao Grid'5000

### 1.1 Acesso ao Frontend

```bash
# Conectar ao site Lyon (onde está o Hydra)
ssh <seu_usuario>@access.grid5000.fr

# Uma vez dentro, conectar ao site específico
ssh lyon
```

### 1.2 Verificar Recursos Disponíveis no Hydra

```bash
# Ver status dos nós no cluster Hydra
oarstat -f

# Ver nós disponíveis com GPU
oarnodes -p host,gpu,gpu_model | grep hydra

# Informações detalhadas do cluster
oarstat -u        # seus jobs
oarstat -g hydra  # jobs no hydra
```

---

## 2. Reserva de Recursos (OAR)

### 2.1 Equivalência SLURM → OAR

| SLURM | OAR | Descrição |
|-------|-----|-----------|
| `sbatch script.slurm` | `oarsub -S script.oar` | Submeter job |
| `srun` | `oarsh` / execução direta | Executar comando |
| `scancel <jobid>` | `oardel <jobid>` | Cancelar job |
| `squeue` | `oarstat` | Ver fila |
| `--nodes=N` | `-l nodes=N` | Número de nós |
| `--time=HH:MM:SS` | `-l walltime=HH:MM:SS` | Tempo máximo |
| `--gpus=N` | `-l gpu=N` ou `-t gpu` | Solicitar GPUs |
| `--partition=xxx` | `-q xxx` ou `-p cluster='xxx'` | Cluster/partição |

### 2.2 Reserva Interativa (Recomendado para Testes)

```bash
# Reservar 1 nó do Hydra com GPU por 2 horas
oarsub -I -q default -p "cluster='hydra'" -l nodes=1,walltime=2:00:00 -t exotic

# Reservar múltiplos nós com GPU
oarsub -I -q default -p "cluster='hydra'" -l nodes=2,walltime=4:00:00 -t exotic

# Com tipo de GPU específico (se disponível)
oarsub -I -p "cluster='hydra' and gpu_model='A100'" -l nodes=1,walltime=2:00:00 -t exotic
```

> **Nota:** O flag `-t exotic` é necessário para recursos especiais como GPUs no G5K.

### 2.3 Reserva em Batch (Para Jobs Longos)

```bash
# Submeter script OAR
oarsub -S ./run_training.oar

# Ou diretamente na linha de comando
oarsub -p "cluster='hydra'" -l nodes=2,walltime=24:00:00 -t exotic "./meu_script.sh"
```

---

## 3. Build do Docker no Nó

### 3.1 Após Obter a Reserva

Uma vez dentro do nó (após `oarsub -I`), você pode usar Docker:

```bash
# Verificar se Docker está disponível
docker --version

# No G5K, geralmente precisamos usar o ambiente kadeploy ou 
# o modo de acesso privilegiado para Docker

# Se Docker não estiver disponível diretamente, usar:
# 1. Deploy de ambiente com Docker
kadeploy3 -f $OAR_NODE_FILE -e debian11-x64-nfs -k

# 2. Ou usar Singularity (mais comum no G5K)
```

### 3.2 Opção A: Usando Docker (se disponível)

```bash
# Navegar até o diretório do projeto
cd ~/projects/hcpa/tensorflow_base

# Build da imagem Docker
docker build -t tf_distributed_x86:latest .

# Verificar se a imagem foi criada
docker images | grep tf_distributed
```

### 3.3 Opção B: Usando Singularity (Recomendado no G5K)

O Grid'5000 suporta melhor Singularity. Converta seu Dockerfile:

```bash
# Converter Dockerfile para Singularity (no frontend)
# Primeiro, criar arquivo singularity.def (já existe no seu projeto)

# Build da imagem Singularity
singularity build --fakeroot tf_distributed.sif singularity.def

# Ou converter diretamente de uma imagem Docker
singularity build tf_distributed.sif docker://nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
```

---

## 4. Execução dos Jobs

### 4.1 Execução com Docker

```bash
# Dentro do nó reservado
docker run --gpus all --rm \
    -v $(pwd):/workspace \
    -v $(pwd)/data/all-tfrec:/workspace/data/all-tfrec \
    --network host \
    --ipc=host \
    tf_distributed_x86:latest \
    python3 /workspace/dr_hcpa_v2_2024.py --batch_size 96 --epochs 100
```

### 4.2 Execução com Singularity

```bash
# Execução básica com GPU
singularity exec --nv \
    --bind $(pwd):/workspace \
    --bind $(pwd)/data/all-tfrec:/workspace/data/all-tfrec \
    tf_distributed.sif \
    python3 /workspace/dr_hcpa_v2_2024.py --batch_size 96 --epochs 100
```

### 4.3 Execução Distribuída Multi-Nó

Para treinamento distribuído, você precisa coordenar múltiplos nós:

```bash
# Obter lista de nós alocados
cat $OAR_NODE_FILE

# Definir variáveis
MASTER_ADDR=$(head -1 $OAR_NODE_FILE)
MASTER_PORT=12345
WORLD_SIZE=$(wc -l < $OAR_NODE_FILE)

# Executar em cada nó usando oarsh
for i in $(seq 0 $((WORLD_SIZE-1))); do
    NODE=$(sed -n "$((i+1))p" $OAR_NODE_FILE)
    oarsh $NODE "docker run --gpus all --rm \
        -v /home/$USER/projects/hcpa:/workspace \
        --network host \
        -e MASTER_ADDR=$MASTER_ADDR \
        -e MASTER_PORT=$MASTER_PORT \
        -e WORLD_SIZE=$WORLD_SIZE \
        -e RANK=$i \
        tf_distributed_x86:latest \
        python3 /workspace/dr_hcpa_v2_2024.py" &
done
wait
```

---

## 5. Scripts Prontos para Uso

### 5.1 Scripts OAR por Projeto e Arquitetura

| Projeto | ARM (Hydra) | x86 (Sirius/Gemini) |
|---------|-------------|---------------------|
| tensorflow_base | `run_g5k_arm.oar` | `run_g5k_x86.oar` |
| pytorch_base | `run_g5k_arm.oar` | `run_g5k_x86.oar` |
| tensorflow_opt | `run_g5k_arm.oar` | `run_g5k_x86.oar` |
| pytorch_opt | `run_g5k_arm.oar` | `run_g5k_x86.oar` |

### 5.2 Comandos Rápidos

```bash
# === CONEXÃO ===
ssh <user>@access.grid5000.fr
ssh lyon

# === RESERVA RÁPIDA ARM (Hydra - H200) ===
oarsub -I -p "cluster='hydra'" -l nodes=1,walltime=2:00:00 -t exotic

# === RESERVA RÁPIDA x86 (Sirius - A100) ===
oarsub -I -p "cluster='sirius'" -l nodes=1,walltime=2:00:00 -t exotic

# === RESERVA RÁPIDA x86 (Gemini - V100) ===
oarsub -I -p "cluster='gemini'" -l nodes=1,walltime=2:00:00 -t exotic

# === SUBMETER JOB BATCH ===
cd ~/projects/hcpa/tensorflow_base
oarsub -S ./run_g5k_arm.oar    # Para Hydra (ARM)
oarsub -S ./run_g5k_x86.oar    # Para Sirius/Gemini (x86)

# === MONITORAR ===
oarstat -u                    # Seus jobs
watch -n5 oarstat -u          # Monitorar em tempo real

# === CANCELAR ===
oardel <JOB_ID>
```

---

## 6. Executar Todas as Abordagens HCPA

### 6.1 Script Mestre

Use o script `submit_all_g5k.sh` para submeter todos os jobs de uma vez:

```bash
cd ~/projects/hcpa

# Submeter apenas jobs ARM (Hydra)
./submit_all_g5k.sh arm

# Submeter apenas jobs x86 (Sirius/Gemini)
./submit_all_g5k.sh x86

# Submeter TODOS (ARM + x86)
./submit_all_g5k.sh all
```

### 6.2 Submissão Manual por Projeto

```bash
# === tensorflow_base ===
cd ~/projects/hcpa/tensorflow_base
oarsub -S ./run_g5k_arm.oar

# === pytorch_base ===
cd ~/projects/hcpa/pytorch_base
oarsub -S ./run_g5k_arm.oar

# === tensorflow_opt ===
cd ~/projects/hcpa/tensorflow_opt
oarsub -S ./run_g5k_arm.oar

# === pytorch_opt ===
cd ~/projects/hcpa/pytorch_opt
oarsub -S ./run_g5k_arm.oar

```

### 6.3 Verificar Status de Todos os Jobs

```bash
# Ver todos os seus jobs
oarstat -u

# Cancelar todos os seus jobs
oardel $(oarstat -u | grep -v Job_id | awk '{print $1}')
```

---

## 📝 Diferenças Importantes SLURM vs OAR

1. **Variáveis de Ambiente:**
   - SLURM: `$SLURM_JOB_ID`, `$SLURM_NODELIST`
   - OAR: `$OAR_JOB_ID`, `$OAR_NODE_FILE`

2. **Execução Remota:**
   - SLURM: `srun`
   - OAR: `oarsh` ou execução direta no script

3. **Arrays de Jobs:**
   - SLURM: `--array=1-10`
   - OAR: Usar loops ou `oarsub` múltiplas vezes

4. **GPUs:**
   - No G5K, usar `-t exotic` para acessar recursos especiais

---

## 🔗 Links Úteis

- [Documentação OAR](https://oar.imag.fr/docs/latest/)
- [Grid'5000 Getting Started](https://www.grid5000.fr/w/Getting_Started)
- [Grid'5000 Docker](https://www.grid5000.fr/w/Docker)
- [Cluster Hydra](https://www.grid5000.fr/w/Lyon:Hardware#hydra)

---

## ⚠️ Troubleshooting

### Docker não disponível
```bash
# Fazer deploy de ambiente com Docker
kadeploy3 -f $OAR_NODE_FILE -e debian11-x64-nfs -k
# Após o deploy, conectar novamente ao nó
```

### GPU não detectada
```bash
# Verificar se CUDA está acessível
nvidia-smi

# Se não funcionar, verificar se reservou com -t exotic
```

### Conexão entre nós falha
```bash
# Usar a rede interna do G5K
# Verificar interfaces disponíveis
ip addr show

# Geralmente usar a interface eth0 ou ib0 (InfiniBand)
```
