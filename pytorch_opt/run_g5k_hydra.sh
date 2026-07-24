#!/usr/bin/env bash
###############################################################################
# pytorch_opt/run_g5k_hydra.sh
#
# Roda dr_hcpa_v2_2024.py (PyTorch) no node Hydra (Lyon), preparado pelo
# tools/g5k_setup_node.sh. Itera RUN_START..RUN_END no padrao 0..9.
#
# A receita de treino vem de tools/common_recipe.sh — a MESMA usada pelo
# tensorflow_opt. Só o que é específico do runtime PyTorch fica aqui embaixo.
#
# Pre-requisitos:
#   - tools/g5k_setup_node.sh ja submeteu o job e fez kadeploy
#   - hcpa.sif buildado no node (dentro de pytorch_opt/)
#   - data/all-tfrec presente em ${PROJECT_DIR}/data/all-tfrec
#
# Uso:
#   bash run_g5k_hydra.sh
#   RUN_START=0 RUN_END=2 bash run_g5k_hydra.sh    # so 3 runs
#   HCPA_EPOCHS=50 bash run_g5k_hydra.sh           # epocas reduzidas
###############################################################################

set -euo pipefail

APPROACH=pytorch_opt
ENTRY=dr_hcpa_v2_2024.py
RESULTS_FLAG=--results
EXEC_FLAG=--exec
SEED_FLAG=--seed
TRAIN_ACCEPTS_DATASET=1

TOOLS_DIR="$(cd "$(dirname "$0")/../tools" && pwd)"
# shellcheck source=../tools/common_recipe.sh
source "${TOOLS_DIR}/common_recipe.sh"

# Compatibilidade: TARGET_EPOCHS continua funcionando como atalho.
HCPA_EPOCHS="${TARGET_EPOCHS:-${HCPA_EPOCHS}}"

# Com freeze_epochs=0 o LR operante é fine_tune_lr (não --lrate); escrevemos os
# dois com o mesmo valor. O JIT aqui é torch.compile (ligado por padrão; use
# PT_DISABLE_COMPILE=1 para o baseline eager). AMP/FP16 é o default do script.
TRAIN_STATIC_ARGS=(
  --model             inception_v3
  --normalize         preprocess
  --batch_size        "${HCPA_BATCH_SIZE}"
  --epochs            "${HCPA_EPOCHS}"
  --lrate             "${HCPA_LRATE}"
  --fine_tune_lr      "${HCPA_LRATE}"
  --fine_tune_lr_factor 1.0
  --freeze_epochs     "${HCPA_FREEZE_EPOCHS}"
  --warmup_epochs     "${HCPA_WARMUP_EPOCHS}"
  --min_lr            "${HCPA_MIN_LR}"
  --optimizer         "${HCPA_OPTIMIZER}"
  --clip_grad_norm    "${HCPA_GRAD_CLIP}"
  --mixup_alpha       "${HCPA_MIXUP}"
  --cutmix_alpha      "${HCPA_CUTMIX}"
  --label_smoothing   "${HCPA_LABEL_SMOOTHING}"
  --focal_gamma       "${HCPA_FOCAL_GAMMA}"
  --pos_weight        "${HCPA_POS_WEIGHT}"
  --fundus_crop_ratio "${HCPA_CROP_RATIO}"
  --tta_views         "${HCPA_TTA_VIEWS}"
  --enable_amp
)

# DALI: desligado nos dois _opt (o tensorflow_opt tem `use_dali = False`
# hardcoded no ARM/GH200). Ligar aqui sozinho quebraria a comparabilidade.
if [[ "${HCPA_USE_DALI}" == "1" ]]; then
  TRAIN_STATIC_ARGS+=(--enable_dali)
else
  TRAIN_STATIC_ARGS+=(--disable_dali)
fi

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

hcpa_recipe_banner

# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
