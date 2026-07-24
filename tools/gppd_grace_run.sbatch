#!/usr/bin/env bash
# GPPD (SLURM, nodes grace ARM + L40S) — roda pytorch_opt OU tensorflow_opt
# single-GPU via docker, com a receita _opt (espelha tools/common_recipe.sh) e o
# dataset all-tfrec-v2 (val* disjunto). Autossuficiente: NAO depende do helper
# hcpa_ssd_runtime.sh (que nao existe no repo).
#
# Uso:
#   sbatch --job-name=pt_opt  --nodelist=grace1 --array=0-0 \
#          --export=ALL,APPROACH=pytorch_opt,IMAGE=distributed_arm:latest,HCPA_EPOCHS=2 \
#          tools/gppd_grace_run.sbatch
#   sbatch --job-name=tf_opt  --nodelist=grace2 --array=0-0 \
#          --export=ALL,APPROACH=tensorflow_opt,IMAGE=tf_distributed_arm:latest,HCPA_EPOCHS=2 \
#          tools/gppd_grace_run.sbatch
# Full: trocar --array=0-9%1 e HCPA_EPOCHS=200.
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/bmmorales/projects/hcpa/logs/%x_%A_%a.out
#SBATCH --error=/home/users/bmmorales/projects/hcpa/logs/%x_%A_%a.err
set -uo pipefail

PROJ=/home/users/bmmorales/projects/hcpa
APPROACH="${APPROACH:?defina APPROACH=pytorch_opt|tensorflow_opt}"
IMAGE="${IMAGE:?defina IMAGE (distributed_arm:latest | tf_distributed_arm:latest)}"
EPOCHS="${HCPA_EPOCHS:-200}"
RUN_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((42 + RUN_ID))
DATA="${DATA_DIR:-$PROJ/data/all-tfrec-v2}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-results_gppd_grace}"
SAMPLE_MS="${HCPA_GPU_SAMPLE_MS:-200}"
TMP="/tmp/hcpa_${SLURM_JOB_ID:-manual}_${APPROACH}_r${RUN_ID}"
mkdir -p "$TMP" "$PROJ/$APPROACH/$RESULTS_SUBDIR/run_${RUN_ID}" "$PROJ/logs"

# ---- receita _opt por abordagem (identica a run_g5k_hydra.sh + common_recipe.sh) ----
if [[ "$APPROACH" == "pytorch_opt" ]]; then
  KBACK=torch
  ARGS="--model inception_v3 --normalize preprocess --batch_size 96 --epochs $EPOCHS \
--lrate 5e-4 --fine_tune_lr 5e-4 --fine_tune_lr_factor 1.0 --freeze_epochs 0 \
--warmup_epochs 5 --min_lr 1e-6 --optimizer adamw --clip_grad_norm 1.0 \
--mixup_alpha 0.0 --cutmix_alpha 0.0 --label_smoothing 0.0 --focal_gamma 0.0 \
--pos_weight 1.0 --fundus_crop_ratio 1.0 --tta_views 1 --enable_amp --disable_dali --seed $SEED"
  EXTRA_ENV=""
elif [[ "$APPROACH" == "tensorflow_opt" ]]; then
  KBACK=tensorflow
  ARGS="--batch_size 96 --model InceptionV3 --normalize preprocess --epochs $EPOCHS \
--freeze_epochs 0 --fine_tune_at 0 --no-freeze_bn --lrate 5e-4 --fine_tune_lr 5e-4 \
--warmup_epochs 5 --min_lr 1e-6 --optimizer adamw --grad_clip_norm 1.0 \
--mixup_alpha 0.0 --cutmix_alpha 0.0 --label_smoothing 0.0 --focal_gamma 0.0 \
--pos_weight 1.0 --fundus_crop_ratio 1.0 --tta_views 1 --ema_decay 0.0 \
--jit_compile --auc_target 0.95 --cache_dir memory --exact_eval_interval 1 --seed $SEED"
  EXTRA_ENV="-e TF_GPU_ALLOCATOR=cuda_malloc_async -e TF_FORCE_GPU_ALLOW_GROWTH=true \
-e TF_ENABLE_GPU_GC=1 -e TF_NUM_INTEROP_THREADS=0 -e TF_CPP_MIN_LOG_LEVEL=2 -e DALI_LOG=ERROR"
else
  echo "APPROACH invalido: $APPROACH"; exit 2
fi

echo "[$(hostname)] APPROACH=$APPROACH RUN=$RUN_ID EPOCHS=$EPOCHS SEED=$SEED IMAGE=$IMAGE"
echo "[$(hostname)] DATA=$DATA RESULTS=$APPROACH/$RESULTS_SUBDIR/run_${RUN_ID}"

# shellcheck disable=SC2086
docker run --rm --name "hcpa_${APPROACH}_${SLURM_JOB_ID:-m}_r${RUN_ID}" \
  --gpus all --runtime=nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -v "$PROJ/$APPROACH:/workspace/$APPROACH" \
  -v "$PROJ/tools:/workspace/tools:ro" \
  -v "$DATA:/workspace/data:ro" \
  -v "$TMP:/tmp/hcpa" \
  -w "/workspace/$APPROACH" \
  -e HOME=/tmp/hcpa -e TMPDIR=/tmp/hcpa -e TMP=/tmp/hcpa -e TEMP=/tmp/hcpa \
  -e MPLCONFIGDIR=/tmp/hcpa/mpl -e XDG_CACHE_HOME=/tmp/hcpa/cache \
  -e HF_HOME=/tmp/hcpa/cache/hf -e TORCH_HOME=/tmp/hcpa/cache/torch \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/hcpa/cache/inductor -e TRITON_CACHE_DIR=/tmp/hcpa/cache/triton \
  -e KERAS_BACKEND="$KBACK" -e PYTHONUNBUFFERED=1 -e HCPA_GPU_SAMPLE_MS="$SAMPLE_MS" \
  $EXTRA_ENV \
  "$IMAGE" \
  bash -lc "mkdir -p /tmp/hcpa/mpl /tmp/hcpa/cache; \
    echo '[container] GPU:'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; \
    python3 dr_hcpa_v2_2024.py --tfrec_dir /workspace/data --dataset all-tfrec-v2 \
      --results ./$RESULTS_SUBDIR/run_${RUN_ID} --exec ${RUN_ID} $ARGS"
rc=$?
echo "[$(hostname)] run $RUN_ID FINISHED rc=$rc"
exit $rc
