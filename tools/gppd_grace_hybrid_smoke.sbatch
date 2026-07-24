#!/usr/bin/env bash
# SMOKE: compara transformer_dim=256 (NOVO, gargalo) vs 2048 (ANTIGO) no
# hybrid_token_reduction_opt, single-GPU L40S (grace), 2 epocas, freeze=0.
# Mede throughput/memoria reais a partir do CSV de resultados.
#SBATCH --job-name=hyb_smoke
#SBATCH --partition=grace
#SBATCH --nodelist=grace1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=/home/users/bmmorales/projects/hcpa/logs/%x_%j.out
#SBATCH --error=/home/users/bmmorales/projects/hcpa/logs/%x_%j.err
set -uo pipefail

PROJ=/home/users/bmmorales/projects/hcpa
APPROACH=hybrid_token_reduction_opt
IMAGE="${IMAGE:-hybrid_token_reduction_opt_distributed_arm:latest}"
DATA="$PROJ/data/all-tfrec-v2"
EPOCHS="${EPOCHS:-2}"
BATCH="${BATCH:-96}"
TMP="/tmp/hcpa_smoke_${SLURM_JOB_ID:-manual}"
mkdir -p "$TMP" "$PROJ/logs"

echo "[$(hostname)] arch=$(uname -m) image=$IMAGE data=$DATA epochs=$EPOCHS"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "IMAGE MISSING: $IMAGE"
  echo "--- imagens hybrid/arm/pytorch disponiveis ---"
  docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep -iE "hybrid|arm|pytorch|torch" || docker images | head
  exit 3
fi

run_dim () {
  local DIM=$1
  local RES="./results_smoke_dim${DIM}/run_0"
  echo "############################################################"
  echo "### transformer_dim=$DIM  -> $APPROACH/$RES"
  echo "############################################################"
  rm -rf "$PROJ/$APPROACH/results_smoke_dim${DIM}"
  mkdir -p "$PROJ/$APPROACH/results_smoke_dim${DIM}/run_0"
  # shellcheck disable=SC2086
  docker run --rm --name "hyb_smoke_${SLURM_JOB_ID:-m}_d${DIM}" \
    --gpus all --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --user "$(id -u):$(id -g)" \
    -v "$PROJ:/workspace" \
    -v "$DATA:/workspace/data_v2:ro" \
    -v "$TMP:/tmp/hcpa" \
    -w "/workspace/$APPROACH" \
    -e HOME=/tmp/hcpa -e TMPDIR=/tmp/hcpa -e MPLCONFIGDIR=/tmp/hcpa/mpl \
    -e XDG_CACHE_HOME=/tmp/hcpa/cache -e TORCH_HOME=/tmp/hcpa/cache/torch \
    -e HF_HOME=/tmp/hcpa/cache/hf -e PYTHONPATH=/workspace \
    -e KERAS_BACKEND=torch -e PYTHONUNBUFFERED=1 -e HCPA_GPU_SAMPLE_MS=200 \
    "$IMAGE" \
    bash -lc "mkdir -p /tmp/hcpa/mpl /tmp/hcpa/cache; \
      nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
      python3 train.py \
        --tfrec_dir /workspace/data_v2 --dataset all-tfrec-v2 \
        --results_dir $RES --exec_id 0 --seed 42 \
        --batch_size $BATCH --epochs $EPOCHS --img_size 299 \
        --backbone inception_v3 --transformer_dim $DIM \
        --num_transformer_layers 4 --num_heads 4 --keep_ratio 0.5 \
        --freeze_backbone_epochs 0 --pos_weight 1.0 --amp_dtype auto \
        --mixup_alpha 0.0 --cutmix_alpha 0.0 \
        --enable_flash_attention --enable_amp --disable_ema --disable_dali \
        --disable_cosine --disable_compile"
  echo "[$(hostname)] dim=$DIM rc=$?"
}

run_dim 256
run_dim 2048
echo "[$(hostname)] SMOKE DONE"