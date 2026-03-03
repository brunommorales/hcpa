#!/usr/bin/env bash
set -euo pipefail

# pega o nodefile em qualquer nome que aparecer
NODEFILE="${OAR_NODE_FILE:-${OAR_NODEFILE:-}}"
if [[ -z "${NODEFILE}" || ! -f "${NODEFILE}" ]]; then
  echo "ERROR: não achei OAR_NODE_FILE/OAR_NODEFILE"
  exit 1
fi

mapfile -t NODES < "${NODEFILE}"
NNODES="${#NODES[@]}"
if (( NNODES < 2 )); then
  echo "ERROR: precisa de >=2 nós. NNODES=${NNODES}"
  exit 2
fi

# configs
SIF="${SIF:-hcpa.sif}"
TF_PORT_BASE="${TF_PORT_BASE:-12345}"
WORKDIR="${WORKDIR:-/workspace}"

HOST_PROJ_DIR="${HOST_PROJ_DIR:-$PWD}"
HOST_TFREC_DIR="${HOST_TFREC_DIR:-$PWD/data/all-tfrec}"

GPU_NAME_RAW="${GPU_NAME_RAW:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || true)}"
GPU_TAG="$(printf '%s' \"${GPU_NAME_RAW:-gpu}\" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g' | sed -E 's/-+/-/g')"
if [[ -z "${GPU_TAG}" || "${GPU_TAG}" == "unknown" ]]; then
  GPU_TAG="gpu"
fi
JOB_ID_TAG="${OAR_JOB_ID:-manual}"
HOST_RESULTS_DIR="${HOST_RESULTS_DIR:-$PWD/results/${GPU_TAG}+${JOB_ID_TAG}}"
RUNS_DIR_HOST="${HOST_RESULTS_DIR}/runs"
RUN_ID="${RUN_ID:-0}"

mkdir -p logs "${HOST_RESULTS_DIR}" "${RUNS_DIR_HOST}/${RUN_ID}"

# lista de endpoints worker: host:porta
WORKERS=()
for i in "${!NODES[@]}"; do
  WORKERS+=("${NODES[$i]}:$((TF_PORT_BASE+i))")
done
WORKER_LIST="$(IFS=,; echo "${WORKERS[*]}")"

echo "[OAR] NNODES=${NNODES}"
echo "[OAR] workers=${WORKER_LIST}"
echo "[OAR] results=${HOST_RESULTS_DIR} (run=${RUN_ID})"

# cria TF_CONFIG por rank em arquivo (evita dor com aspas/JSON)
for i in "${!NODES[@]}"; do
  python3 - <<PY > "${HOST_RESULTS_DIR}/tf_config_rank${i}.json"
import json
workers = "${WORKER_LIST}".split(",")
cfg = {"cluster": {"worker": workers}, "task": {"type": "worker", "index": int(${i})}}
print(json.dumps(cfg))
PY
done

# dispara 1 worker por nó
PIDS=()
for i in "${!NODES[@]}"; do
  node="${NODES[$i]}"
  log="logs/worker_rank${i}_job${OAR_JOB_ID:-manual}.log"

  echo "[LAUNCH] rank=${i} node=${node} -> ${log}"

  oarsh "${node}" bash -lc "cd '${HOST_PROJ_DIR}' && \
    export TF_CONFIG=\$(cat '${HOST_RESULTS_DIR}/tf_config_rank${i}.json') && \
    export TF_CPP_MIN_LOG_LEVEL=1 && \
    export TF_GPU_ALLOCATOR=cuda_malloc_async && \
    singularity exec --nv \
      -B '${HOST_PROJ_DIR}:${WORKDIR}' \
      -B '${HOST_TFREC_DIR}:${WORKDIR}/data' \
      -B '${HOST_RESULTS_DIR}:${WORKDIR}/results' \
      '${SIF}' \
      python3 ${WORKDIR}/dr_hcpa_v2_2024.py \
        --tfrec_dir ${WORKDIR}/data \
        --results ${WORKDIR}/results/runs/${RUN_ID} \
        --batch_size 96 \
        --epochs 200 \
  " >"${log}" 2>&1 &

  PIDS+=($!)
done

# espera todo mundo
fail=0
for p in "${PIDS[@]}"; do
  wait "${p}" || fail=1
done

exit "${fail}"
