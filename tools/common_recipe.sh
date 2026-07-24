#!/usr/bin/env bash
###############################################################################
# tools/common_recipe.sh
#
# RECEITA COMUM aos dois "_opt" (tensorflow_opt e pytorch_opt).
#
# Por que existe: o estudo compara FRAMEWORK/RUNTIME (XLA vs torch.compile), não
# receitas de treino. Se cada script tiver sua própria lista de hiperparâmetros,
# eles derivam com o tempo e a comparação deixa de isolar o runtime. Aqui há UMA
# fonte de verdade; os run scripts só a traduzem para as flags de cada CLI.
#
# Tudo é sobreponível por variável de ambiente:
#   HCPA_LRATE=1e-3 bash tensorflow_opt/run_g5k_hydra.sh
#
# ---------------------------------------------------------------------------
# O que ficou IGUAL nos dois (verificado no código, não nas flags):
#   modelo InceptionV3 / 299px / batch 96 / 200 épocas
#   precisão FP16 (TF mixed_precision ; PyTorch AMP)
#   otimizador AdamW, beta 0.9/0.999, eps 1e-7
#   weight decay: head 1e-4, fine-tune 1e-5
#   grad clip 1.0 ; warmup 5 ; cosine ; min_lr 1e-6
#   JIT ligado (TF: --jit_compile/XLA ; PyTorch: torch.compile)
#   EMA DESLIGADO
#   DALI DESLIGADO (decode na CPU) — ver nota abaixo
#
# Com freeze_epochs=0 os DOIS scripts passam a usar `fine_tune_lr` como LR
# operante (TF: linha ~2887; PyTorch: linha ~1362). Por isso HCPA_LRATE é
# escrito nas duas flags (--lrate e --fine_tune_lr).
#
# ---------------------------------------------------------------------------
# DIFERENÇAS QUE PERMANECEM (declarar como limitação, não são elimináveis aqui):
#   1. JIT: XLA (fusão de grafo) != TorchInductor/Triton. "Ligado nos dois" não
#      significa "equivalente".
#   2. Pipeline de dados: tf.data (decode_jpeg na CPU, AUTOTUNE) vs DataLoader
#      PyTorch (workers na CPU). Ambos decodificam na CPU e nenhum usa DALI,
#      mas o escalonamento dos workers difere.
#   3. Kernels de convolução: cuDNN via TF vs via PyTorch podem escolher
#      algoritmos diferentes para a mesma camada.
#
# NOTA SOBRE DALI: nenhum dos dois usa DALI hoje.
#   - tensorflow_opt: `use_dali = False` HARDCODED (device mismatch no ARM/GH200).
#   - pytorch_opt: `--enable_dali` existe mas o default é desligado.
#   Ou seja, nessa dimensão os dois JÁ estão alinhados (decode na CPU). Ligar
#   DALI nos dois exigiria destravar o caminho ARM do TF; enquanto isso, manter
#   desligado nos dois é a alinhagem correta e de risco zero.
#   Para a ablação futura: HCPA_USE_DALI=1 (só pytorch_opt hoje).
###############################################################################

# --- duração e forma do problema ---
: "${HCPA_EPOCHS:=200}"
: "${HCPA_BATCH_SIZE:=96}"
: "${HCPA_IMG_SIZE:=299}"

# --- otimização ---
# LRATE: definido pela mini-varredura (tools/lr_sweep.sh). Enquanto ela não roda,
# 5e-4 é o valor que comprovadamente converge no pytorch_opt.
: "${HCPA_LRATE:=5e-4}"
: "${HCPA_WARMUP_EPOCHS:=5}"
: "${HCPA_MIN_LR:=1e-6}"
: "${HCPA_OPTIMIZER:=adamw}"
: "${HCPA_GRAD_CLIP:=1.0}"

# --- regularização / augmentation: ZERADAS para isolar o runtime ---
: "${HCPA_FREEZE_EPOCHS:=0}"
: "${HCPA_MIXUP:=0.0}"
: "${HCPA_CUTMIX:=0.0}"
: "${HCPA_LABEL_SMOOTHING:=0.0}"
: "${HCPA_FOCAL_GAMMA:=0.0}"
: "${HCPA_POS_WEIGHT:=1.0}"
: "${HCPA_CROP_RATIO:=1.0}"
: "${HCPA_TTA_VIEWS:=1}"

# --- EMA: desligado (medido: nunca chega ao teste, custa ~6,6%/época) ---
: "${HCPA_EMA_DECAY:=0.0}"

# --- pipeline de dados ---
: "${HCPA_USE_DALI:=0}"

# --- telemetria (gpu_energy.py) ---
# vazio = usa DEFAULT_SAMPLE_INTERVAL_S (200 ms). 0 = desliga a amostragem.
: "${HCPA_GPU_SAMPLE_MS:=}"
[[ -n "${HCPA_GPU_SAMPLE_MS}" ]] && export HCPA_GPU_SAMPLE_MS

hcpa_recipe_banner() {
  echo "[receita comum] epochs=${HCPA_EPOCHS} batch=${HCPA_BATCH_SIZE} lr=${HCPA_LRATE}"
  echo "[receita comum] freeze=${HCPA_FREEZE_EPOCHS} mixup=${HCPA_MIXUP} cutmix=${HCPA_CUTMIX}" \
       "ls=${HCPA_LABEL_SMOOTHING} focal=${HCPA_FOCAL_GAMMA} pos_w=${HCPA_POS_WEIGHT}"
  echo "[receita comum] crop=${HCPA_CROP_RATIO} tta=${HCPA_TTA_VIEWS} ema=${HCPA_EMA_DECAY}" \
       "dali=${HCPA_USE_DALI} sample_ms=${HCPA_GPU_SAMPLE_MS:-default}"
}
