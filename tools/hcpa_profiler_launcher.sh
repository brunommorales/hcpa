#!/usr/bin/env bash

hcpa_profile_init_env() {
  export PROFILE_TOOL="${PROFILE_TOOL:-none}"
  export ENABLE_TORCH_TRACE="${ENABLE_TORCH_TRACE:-0}"
  export ENABLE_NVTX="${ENABLE_NVTX:-0}"
  export PROFILE_RANK0_ONLY="${PROFILE_RANK0_ONLY:-1}"
  export PROFILE_STAGES="${PROFILE_STAGES:-train,val,test,eval}"
  export TORCH_PROFILE_WAIT="${TORCH_PROFILE_WAIT:-1}"
  export TORCH_PROFILE_WARMUP="${TORCH_PROFILE_WARMUP:-1}"
  export TORCH_PROFILE_ACTIVE="${TORCH_PROFILE_ACTIVE:-3}"
  export TORCH_PROFILE_REPEAT="${TORCH_PROFILE_REPEAT:-1}"
  export TORCH_PROFILE_MAX_STEPS="${TORCH_PROFILE_MAX_STEPS:-0}"
  export TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-1}"
  export TORCH_PROFILE_PROFILE_MEMORY="${TORCH_PROFILE_PROFILE_MEMORY:-1}"
  export TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK:-0}"
  export NSYS_EXPORT_SQLITE="${NSYS_EXPORT_SQLITE:-1}"
  export NCU_EXPORT_CSV="${NCU_EXPORT_CSV:-1}"
  export NCU_SET="${NCU_SET:-full}"
  export NCU_TARGET_PROCESSES="${NCU_TARGET_PROCESSES:-all}"
  export HCPA_REQUESTED_TORCH_PROFILER="${ENABLE_TORCH_TRACE}"

  if [[ "${PROFILE_TOOL}" != "none" ]]; then
    ENABLE_NVTX=1
  fi

  if [[ "${PROFILE_TOOL}" != "none" && "${ENABLE_TORCH_TRACE}" == "1" ]]; then
    echo "Aviso: desativando torch.profiler porque PROFILE_TOOL=${PROFILE_TOOL} ja usa CUPTI; mantendo NVTX." >&2
    ENABLE_TORCH_TRACE=0
  fi

  export HCPA_ENABLE_TORCH_PROFILER="${ENABLE_TORCH_TRACE}"
  export HCPA_ENABLE_RUNTIME_TRACE="${ENABLE_TORCH_TRACE}"
  export HCPA_ENABLE_TF_PROFILER="${ENABLE_TORCH_TRACE}"
  export HCPA_EMIT_NVTX="${ENABLE_NVTX}"
  export HCPA_PROFILE_RANK0_ONLY="${PROFILE_RANK0_ONLY}"
  export HCPA_PROFILE_STAGES="${PROFILE_STAGES}"
  export HCPA_TORCH_PROFILER_WAIT="${TORCH_PROFILE_WAIT}"
  export HCPA_TORCH_PROFILER_WARMUP="${TORCH_PROFILE_WARMUP}"
  export HCPA_TORCH_PROFILER_ACTIVE="${TORCH_PROFILE_ACTIVE}"
  export HCPA_TORCH_PROFILER_REPEAT="${TORCH_PROFILE_REPEAT}"
  export HCPA_TORCH_PROFILER_MAX_STEPS="${TORCH_PROFILE_MAX_STEPS}"
  export HCPA_TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES}"
  export HCPA_TORCH_PROFILER_PROFILE_MEMORY="${TORCH_PROFILE_PROFILE_MEMORY}"
  export HCPA_TORCH_PROFILER_WITH_STACK="${TORCH_PROFILE_WITH_STACK}"
  export HCPA_RUNTIME_TRACE_DIR="${HCPA_TORCH_PROFILER_DIR:-}"
}

hcpa_profile_prepare_paths() {
  local profile_root="$1"
  local artifact_stem="$2"
  mkdir -p "${profile_root}"
  export HCPA_TORCH_PROFILER_DIR="${profile_root}/torch_traces"
  mkdir -p "${HCPA_TORCH_PROFILER_DIR}"
  export HCPA_PROFILE_ARTIFACT_ROOT="${profile_root}"
  export HCPA_PROFILE_ARTIFACT_STEM="${artifact_stem}"
  export HCPA_PROFILE_ARTIFACT_PREFIX="${profile_root}/${artifact_stem}"
}

hcpa_profile_wrap_command() {
  PROFILED_CMD=("$@")
  case "${PROFILE_TOOL}" in
    nsys)
      PROFILED_CMD=(
        nsys
        profile
        --force-overwrite=true
        --trace=cuda,nvtx,osrt
        --sample=none
        --cpuctxsw=none
        -o "${HCPA_PROFILE_ARTIFACT_PREFIX}"
        "${PROFILED_CMD[@]}"
      )
      ;;
    ncu)
      PROFILED_CMD=(
        ncu
        --target-processes "${NCU_TARGET_PROCESSES}"
        --set "${NCU_SET}"
        --force-overwrite
        --export "${HCPA_PROFILE_ARTIFACT_PREFIX}"
        "${PROFILED_CMD[@]}"
      )
      ;;
    none)
      ;;
    *)
      echo "Erro: PROFILE_TOOL inválido: ${PROFILE_TOOL}" >&2
      return 1
      ;;
  esac
}

hcpa_profile_postprocess() {
  case "${PROFILE_TOOL}" in
    nsys)
      if [[ "${NSYS_EXPORT_SQLITE}" == "1" ]] && command -v nsys >/dev/null 2>&1; then
        if [[ -f "${HCPA_PROFILE_ARTIFACT_PREFIX}.nsys-rep" ]]; then
          nsys export \
            --type sqlite \
            --force-overwrite=true \
            --output "${HCPA_PROFILE_ARTIFACT_PREFIX}" \
            "${HCPA_PROFILE_ARTIFACT_PREFIX}.nsys-rep" || true
        fi
      fi
      ;;
    ncu)
      if [[ "${NCU_EXPORT_CSV}" == "1" ]] && command -v ncu >/dev/null 2>&1; then
        if [[ -f "${HCPA_PROFILE_ARTIFACT_PREFIX}.ncu-rep" ]]; then
          ncu \
            --import "${HCPA_PROFILE_ARTIFACT_PREFIX}.ncu-rep" \
            --csv \
            --page raw \
            --log-file "${HCPA_PROFILE_ARTIFACT_PREFIX}.csv" >/dev/null 2>&1 || true
        fi
      fi
      ;;
  esac
}

hcpa_profile_write_manifest() {
  local manifest_path="${HCPA_PROFILE_ARTIFACT_ROOT}/profiling_manifest.txt"
  {
    echo "profile_tool=${PROFILE_TOOL}"
    echo "requested_torch_trace=${HCPA_REQUESTED_TORCH_PROFILER}"
    echo "enable_torch_trace=${ENABLE_TORCH_TRACE}"
    echo "enable_nvtx=${ENABLE_NVTX}"
    echo "rank0_only=${PROFILE_RANK0_ONLY}"
    echo "profile_stages=${PROFILE_STAGES}"
    echo "torch_profiler_dir=${HCPA_TORCH_PROFILER_DIR}"
    echo "artifact_prefix=${HCPA_PROFILE_ARTIFACT_PREFIX}"
    echo "torch_wait=${TORCH_PROFILE_WAIT}"
    echo "torch_warmup=${TORCH_PROFILE_WARMUP}"
    echo "torch_active=${TORCH_PROFILE_ACTIVE}"
    echo "torch_repeat=${TORCH_PROFILE_REPEAT}"
    echo "torch_max_steps=${TORCH_PROFILE_MAX_STEPS}"
  } > "${manifest_path}"
}
