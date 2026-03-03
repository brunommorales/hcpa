#!/usr/bin/env bash
#SBATCH --job-name=monai_base
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --array=1-10%1
#SBATCH --partition=grace
#SBATCH --nodelist=grace1
#SBATCH --output=logs/%x_%j_%N.out
#SBATCH --error=logs/%x_%j_%N.err

set -euo pipefail

sanitize_tag() {
  local s="${1:-}"
  s="$(printf '%s' "${s}" | tr '[:upper:]' '[:lower:]')"
  s="$(printf '%s' "${s}" | sed -E 's/[^a-z0-9._-]+/-/g')"
  s="$(printf '%s' "${s}" | sed -E 's/-+/-/g' | sed -E 's/^[-.]+|[-.]+$//')"
  if [[ -z "${s}" ]]; then
    printf '%s' "unknown"
  else
    printf '%s' "${s}"
  fi
}

# --- Ajustáveis ---
: "${IMAGE:=monai_base:latest}"
: "${BATCH_SIZE:=96}"
: "${EPOCHS:=200}"
: "${LR:=3e-4}"
: "${WEIGHT_DECAY:=1e-4}"
: "${IMAGE_SIZE:=299}"
: "${MODEL:=inception_v3}"
: "${NORMALIZE:=inception}"
: "${FUNDUS_CROP:=0.9}"
: "${AUGMENT:=1}"
: "${MIXUP:=0.2}"
: "${CUTMIX:=0.0}"
: "${LABEL_SMOOTH:=0.02}"
: "${EMA_DECAY:=0.0}"
: "${GRAD_CLIP:=1.0}"
: "${GRAD_ACC:=1}"
: "${SEED:=2026}"
: "${USE_COMPILE:=0}"
: "${RUN_EVAL:=1}"
: "${RUN_BENCHMARK:=0}"
: "${BENCH_WARMUP:=20}"
: "${BENCH_STEPS:=200}"

: "${RESULTS_ROOT:=results}"
: "${GPU_NAME_RAW:=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || echo gpu)}"
GPU_NAME="${GPU_NAME_RAW}"

# GPUs por nó via Slurm; fallback CUDA_VISIBLE_DEVICES; último recurso nvidia-smi
GPU_COUNT="${SLURM_GPUS_ON_NODE:-${GPU_COUNT:-}}"
if [[ -z "${GPU_COUNT:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _cvd <<<"${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT=${#_cvd[@]}
  else
    GPU_COUNT=$(nvidia-smi -L | grep -c '^GPU' || true)
  fi
fi
if ! [[ "${GPU_COUNT:-}" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 )); then
  GPU_COUNT=1
fi
NNODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
WORLD_SIZE=$((NNODES * GPU_COUNT))
: "${JOB_TAG:=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}}"

GPU_TAG="$(sanitize_tag "${GPU_NAME_RAW}")"
if [[ -z "${GPU_TAG}" ]]; then GPU_TAG="gpu"; fi
GPU_DESC="${GPU_COUNT}x${GPU_TAG}"
NODEGROUP_RAW=""
if [[ -n "${SLURM_NODELIST:-}" ]]; then
  FIRST_SEG="${SLURM_NODELIST%%,*}"
  FIRST_SEG="${FIRST_SEG%%[*}"
  NODEGROUP_RAW="$(printf '%s' "${FIRST_SEG}" | sed -E 's/[0-9]+$//')"
fi
NODEGROUP_TAG="$(sanitize_tag "${NODEGROUP_RAW:-node}")"
BATCH_TAG="bs$(sanitize_tag "${BATCH_SIZE}")"
if [[ -z "${RESULTS_SUBDIR:-}" ]]; then
  RESULTS_SUBDIR="${RESULTS_ROOT}/result${JOB_TAG}_${NODEGROUP_TAG}_${GPU_DESC}_${BATCH_TAG}"
fi

: "${APP_DIR:=${SLURM_SUBMIT_DIR}}"
HOST_PROJ_DIR="${HOST_PROJ_DIR:-${APP_DIR}}"
CONT_WORKDIR="/workspace"
: "${TFREC_DIR:=${HOST_PROJ_DIR}/data/all-tfrec}"
HOST_TFREC_DIR="${TFREC_DIR}"
CONT_TFREC_DIR="${CONT_WORKDIR}/data/all-tfrec"
: "${RUN_ID:=$(( (${SLURM_ARRAY_TASK_ID:-1}-1) ))}"
if (( RUN_ID < 0 )); then RUN_ID=0; fi
RUN_DIR_BASE="${HOST_PROJ_DIR}/${RESULTS_SUBDIR}"
RUN_DIR_HOST="${RUN_DIR_BASE}/run_${RUN_ID}"
RUN_DIR_CONT="${CONT_WORKDIR}/${RESULTS_SUBDIR}/run_${RUN_ID}"
mkdir -p "${RUN_DIR_HOST}" "${RUN_DIR_BASE}" "${HOST_PROJ_DIR}/logs"
: "${HOST_UID:=$(id -u)}"
: "${HOST_GID:=$(id -g)}"

DOCKER_GPU_ARGS=(--gpus all)
DOCKER_RUNTIME_ARG=(docker)
DOCKER_USER_ARGS=(-u "${HOST_UID}:${HOST_GID}")
CACHE_DIR="${CONT_WORKDIR}/.cache"
MPL_DIR="${CACHE_DIR}/matplotlib"
HF_CACHE="${CACHE_DIR}/huggingface"
TORCH_CACHE="${CACHE_DIR}/torch"

MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}" 2>/dev/null | head -n1 || hostname)}
MASTER_PORT=${MASTER_PORT:-29500}

echo "[monai_base][${SLURM_JOB_ID:-manual}] RUN_ID=${RUN_ID} BATCH=${BATCH_SIZE} EPOCHS=${EPOCHS} MODEL=${MODEL} COMPILE=${USE_COMPILE} GPU=${GPU_NAME} JOB=${JOB_TAG} NNODES=${NNODES} GPUS_PER_NODE=${GPU_COUNT} WORLD_SIZE=${WORLD_SIZE}"
{
  echo "job_id=${SLURM_JOB_ID:-}"
  echo "array_job_id=${SLURM_ARRAY_JOB_ID:-}"
  echo "array_task_id=${SLURM_ARRAY_TASK_ID:-}"
  echo "job_name=${SLURM_JOB_NAME:-}"
  echo "partition=${SLURM_JOB_PARTITION:-${SLURM_PARTITION:-}}"
  echo "nodelist=${SLURM_NODELIST:-}"
  echo "nodegroup=${NODEGROUP_TAG}"
  echo "gpu_name=${GPU_NAME_RAW}"
  echo "gpu_count=${GPU_COUNT}"
  echo "world_size=${WORLD_SIZE}"
  echo "batch_size=${BATCH_SIZE}"
} > "${HOST_PROJ_DIR}/${RESULTS_SUBDIR}/env_manifest.txt"

srun --ntasks=${NNODES} --ntasks-per-node=1 \
  "${DOCKER_RUNTIME_ARG[@]}" run --rm \
  "${DOCKER_GPU_ARGS[@]}" \
  "${DOCKER_USER_ARGS[@]}" \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -e PYTHONUNBUFFERED=1 \
  -e MASTER_ADDR="${MASTER_ADDR}" \
  -e MASTER_PORT="${MASTER_PORT}" \
  -e WORLD_SIZE="${WORLD_SIZE}" \
  -e NNODES="${NNODES}" \
  -e SLURM_PROCID="${SLURM_PROCID:-0}" \
  -e SLURM_LOCALID="${SLURM_LOCALID:-0}" \
  -e SLURM_NODEID="${SLURM_NODEID:-0}" \
  -e HOME="${CONT_WORKDIR}" \
  -e XDG_CACHE_HOME="${CACHE_DIR}" \
  -e MPLCONFIGDIR="${MPL_DIR}" \
  -e HF_HOME="${HF_CACHE}" \
  -e HF_HUB_CACHE="${HF_CACHE}" \
  -e TRANSFORMERS_CACHE="${HF_CACHE}" \
  -e TORCH_HOME="${TORCH_CACHE}" \
  -v "${HOST_PROJ_DIR}:${CONT_WORKDIR}" \
  -v "${HOST_TFREC_DIR}:${CONT_TFREC_DIR}" \
  -w "${CONT_WORKDIR}" \
  "${IMAGE}" \
  bash -lc "set -euo pipefail; \
    NODE_RANK=\${NODE_RANK:-\${SLURM_PROCID:-0}}; \
    if [[ -n \"\${CUDA_VISIBLE_DEVICES:-}\" ]]; then \
      IFS=',' read -ra _cvd <<<\"\${CUDA_VISIBLE_DEVICES}\"; \
      NPROC_PER_NODE=\${NPROC_PER_NODE:-\${#_cvd[@]}}; \
    else \
      NPROC_PER_NODE=\${NPROC_PER_NODE:-\$(nvidia-smi -L | grep -c '^GPU' || echo 1)}; \
    fi; \
    mkdir -p '${RUN_DIR_CONT}' '${RUN_DIR_CONT}/tmp' '${CACHE_DIR}' '${MPL_DIR}' '${HF_CACHE}' '${TORCH_CACHE}'; \
    torchrun \
      --nnodes=${NNODES} \
      --nproc_per_node=\${NPROC_PER_NODE} \
      --node_rank=\${NODE_RANK} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      train.py \
        --results '${RUN_DIR_CONT}' \
        --tfrec_dir '${CONT_TFREC_DIR}' \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --image_size ${IMAGE_SIZE} \
        --model ${MODEL} \
        --normalize ${NORMALIZE} \
        --fundus_crop_ratio ${FUNDUS_CROP} \
        --mixup_alpha ${MIXUP} \
        --cutmix_alpha ${CUTMIX} \
        --label_smoothing ${LABEL_SMOOTH} \
        --ema_decay ${EMA_DECAY} \
        --grad_clip_norm ${GRAD_CLIP} \
        --gradient_accumulation ${GRAD_ACC} \
        --channels_last \
        --amp \
        $([ ${USE_COMPILE} -eq 1 ] || echo --no_compile) \
        $([ ${AUGMENT} -eq 1 ] && echo --augment || echo --no_augment) \
        --seed ${SEED} \
    | tee '${RUN_DIR_CONT}/train.log'; \
    if [ ${RUN_EVAL} -eq 1 ]; then \
      python3 -u eval.py --results '${RUN_DIR_CONT}' | tee -a '${RUN_DIR_CONT}/train.log'; \
    fi; \
    if [ ${RUN_BENCHMARK} -eq 1 ]; then \
      python3 -u benchmark.py \
        --results '${RUN_DIR_CONT}/benchmark' \
        --tfrec_dir '${CONT_TFREC_DIR}' \
        --batch_size ${BATCH_SIZE} \
        --image_size ${IMAGE_SIZE} \
        --model ${MODEL} \
        --warmup_steps ${BENCH_WARMUP} \
        --measure_steps ${BENCH_STEPS} \
      | tee -a '${RUN_DIR_CONT}/train.log'; \
    fi"

echo "[monai_base] Finalizado run_${RUN_ID}"
