#!/usr/bin/env bash
###############################################################################
# tensorflow_opt/run_g5k_hydra.sh
#
# Roda dr_hcpa_v2_2024.py (TensorFlow/Keras) no node Hydra (Lyon).
# Itera RUN_START..RUN_END (default 0..9) usando o hcpa.sif preparado pelo
# tools/g5k_setup_node.sh.
#
# A receita de treino vem de tools/common_recipe.sh — a MESMA usada pelo
# pytorch_opt. Só o que é específico do runtime TF fica aqui embaixo.
###############################################################################

set -euo pipefail

APPROACH=tensorflow_opt
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

# --jit_compile (XLA) é a "otimização" que define o _opt neste framework: sem ele
# a GPU ocia entre kernels. O par no pytorch_opt é torch.compile.
#
# --fine_tune_at 0  -> backbone INTEIRO treinável. O default (-200) treinava só as
#                      ~200 camadas finais, enquanto o pytorch_opt treina tudo.
# --no-freeze_bn    -> BatchNorm atualiza estatísticas, como no pytorch_opt
#                      (cujo default congelava o BN).
# --recompute_backbone foi REMOVIDO: era no-op (o script ignora e só imprime
#                      "[INFO] recompute_grad desativado nesta versão otimizada").
TRAIN_STATIC_ARGS=(
  --batch_size        "${HCPA_BATCH_SIZE}"
  --model             InceptionV3
  --normalize         preprocess
  --epochs            "${HCPA_EPOCHS}"
  --freeze_epochs     "${HCPA_FREEZE_EPOCHS}"
  --fine_tune_at      0
  --no-freeze_bn
  --lrate             "${HCPA_LRATE}"
  --fine_tune_lr      "${HCPA_LRATE}"
  --warmup_epochs     "${HCPA_WARMUP_EPOCHS}"
  --min_lr            "${HCPA_MIN_LR}"
  --optimizer         "${HCPA_OPTIMIZER}"
  --grad_clip_norm    "${HCPA_GRAD_CLIP}"
  --mixup_alpha       "${HCPA_MIXUP}"
  --cutmix_alpha      "${HCPA_CUTMIX}"
  --label_smoothing   "${HCPA_LABEL_SMOOTHING}"
  --focal_gamma       "${HCPA_FOCAL_GAMMA}"
  --pos_weight        "${HCPA_POS_WEIGHT}"
  --fundus_crop_ratio "${HCPA_CROP_RATIO}"
  --tta_views         "${HCPA_TTA_VIEWS}"
  --ema_decay         "${HCPA_EMA_DECAY}"
  --jit_compile
  --auc_target          0.95
  # Cache in-memory LIGADO (paridade com o cache do PyTorch e do tensorflow_base).
  # Antes era 'none' -> o .cache() do codigo NUNCA era chamado e o TF redecodificava
  # o JPEG toda epoca, igual ao PyTorch sem cache. So que a comparacao ficava presa
  # a eficiencia do pipeline (tf.data C++ vs workers Python), nao ao modelo.
  --cache_dir           memory
  # P3: AUC exata (roc_auc_score) TODA epoca. Antes era a cada 5; nas outras 4
  # o val_auc do CSV vinha da metrica Keras binada (num_thresholds=200), e o
  # best-checkpoint comparava epocas medidas por estimadores diferentes. Custo:
  # 1 passada de predict na validacao (val* ~1650 imgs) por epoca — barato.
  --exact_eval_interval 1
)

export APPROACH ENTRY RESULTS_FLAG EXEC_FLAG SEED_FLAG TRAIN_ACCEPTS_DATASET
export KERAS_BACKEND="${KERAS_BACKEND:-tensorflow}"

# Otimizacoes TF para GH200 (espelha distributed_run_arm.slurm)
export TF_GPU_ALLOCATOR="cuda_malloc_async"
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export TF_ENABLE_GPU_GC="1"
export TF_NUM_INTEROP_THREADS="0"
export TF_CPP_MIN_LOG_LEVEL="2"
export DALI_LOG="ERROR"

hcpa_recipe_banner

# shellcheck source=../tools/g5k_run_common.sh
source "${TOOLS_DIR}/g5k_run_common.sh"

g5k_run_all
