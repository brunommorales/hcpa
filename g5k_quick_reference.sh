#!/bin/bash
###############################################################################
# Script de Ajuda Rápida - Grid'5000 Hydra
# 
# Este script mostra comandos úteis para trabalhar no G5K
# Execute: bash g5k_quick_reference.sh
###############################################################################

echo "=============================================="
echo "   GRID'5000 - QUICK REFERENCE (Hydra/Lyon)  "
echo "=============================================="
echo ""

echo "📡 CONEXÃO:"
echo "  ssh <user>@access.grid5000.fr"
echo "  ssh lyon"
echo ""

echo "📋 RESERVA INTERATIVA (debug/testes):"
echo "  # 1 nó por 2 horas:"
echo "  oarsub -I -p \"cluster='hydra'\" -l nodes=1,walltime=2:00:00 -t exotic"
echo ""
echo "  # 2 nós por 4 horas:"
echo "  oarsub -I -p \"cluster='hydra'\" -l nodes=2,walltime=4:00:00 -t exotic"
echo ""

echo "📤 SUBMISSÃO BATCH:"
echo "  oarsub -S ./run_g5k.oar"
echo ""
echo "  # Ou com parâmetros inline:"
echo "  oarsub -p \"cluster='hydra'\" -l nodes=1,walltime=4:00:00 -t exotic './meu_script.sh'"
echo ""

echo "📊 MONITORAMENTO:"
echo "  oarstat -u              # Seus jobs"
echo "  oarstat -f              # Todos os jobs"
echo "  watch -n5 oarstat -u    # Monitorar em tempo real"
echo ""

echo "❌ CANCELAR JOB:"
echo "  oardel <JOB_ID>"
echo ""

echo "🔧 NO NÓ ALOCADO:"
echo "  # Ver GPUs:"
echo "  nvidia-smi"
echo ""
echo "  # Build Docker:"
echo "  cd ~/projects/hcpa/tensorflow_base"
echo "  docker build -t tf_distributed_x86:latest ."
echo ""
echo "  # Executar:"
echo "  docker run --gpus all --rm -v \$(pwd):/workspace --network host \\"
echo "      tf_distributed_x86:latest python3 /workspace/dr_hcpa_v2_2024.py"
echo ""

echo "📁 VARIÁVEIS OAR IMPORTANTES:"
echo "  \$OAR_JOB_ID     - ID do job"
echo "  \$OAR_NODE_FILE  - Arquivo com lista de nós alocados"
echo "  \$OAR_WORKDIR    - Diretório de trabalho"
echo ""

echo "🔗 DOCUMENTAÇÃO:"
echo "  https://www.grid5000.fr/w/Getting_Started"
echo "  https://www.grid5000.fr/w/Lyon:Hardware#hydra"
echo "  https://www.grid5000.fr/w/Docker"
echo ""

echo "=============================================="
