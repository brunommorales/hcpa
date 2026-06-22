# HCPA — Diabetic Retinopathy Detection on Fundus Images

This repository contains training approaches for binary classification of Diabetic Retinopathy (DR)
using fundus photographs from the HCPA (Hospital de Clínicas de Porto Alegre) dataset.

Nine approaches are provided, organized in five groups:

| Group | Approaches |
|---|---|
| TensorFlow baselines | `tensorflow_base`, `tensorflow_opt` |
| PyTorch baselines | `pytorch_base`, `pytorch_opt` |
| CNN + Transformer hybrids | `hybrid_simple`, `hybrid_token_reduction`, `hybrid_token_reduction_opt` |
| Foundation models | `retfound_green`, `vit_pure` |

---

## Dataset

All approaches expect the dataset as TFRecord files.  
Default path: `/path/to/data/all-tfrec/`  
Pass it via `--data_dir` (or `--tfrec_dir`) depending on the approach.

Input format: RGB fundus images.  
Task: Binary classification — **0** = No DR / **1** = DR present.

---

## CNN Backbone — InceptionV3

Six of the nine approaches (all TensorFlow, PyTorch, and Hybrid variants) use **InceptionV3** as the
convolutional backbone.

| Property | Value |
|---|---|
| Architecture | Deep CNN with parallel Inception modules (1×1, 3×3, 5×5 conv + max-pool) |
| Native input size | 299 × 299 px (RGB) |
| Feature output | 2048-dimensional vector after Global Average Pooling |
| Pre-training | ImageNet weights (via Keras or timm) |

In the **hybrid** approaches, InceptionV3 acts as a feature extractor and its output is fed as a
sequence of tokens into a Transformer encoder.

---

## Parameters Overview

### Common parameters across all approaches

| Parameter | Value |
|---|---|
| Epochs | 200 |
| Optimizer | AdamW (default) |
| Training distribution | DDP — DistributedDataParallel (multi-GPU) |
| Loss function | BCEWithLogitsLoss / BinaryCrossentropy + automatic positive weight |
| Task | Binary classification (no DR / DR) |

---

### Input parameters per approach

| Approach | Framework | Backbone | Batch Size | Image Size | Learning Rate |
|---|---|---|---|---|---|
| `tensorflow_base` | TensorFlow | InceptionV3 | 96 | 299 × 299 | 5e-4 |
| `tensorflow_opt` | TensorFlow | InceptionV3 | 96 | 299 × 299 | 3e-3 |
| `pytorch_base` | PyTorch | InceptionV3 (timm) | 96 | 299 × 299 | 5e-4 |
| `pytorch_opt` | PyTorch | InceptionV3 (timm) | 96 | 299 × 299 | 5e-4 |
| `hybrid_simple` | PyTorch | InceptionV3 + Transformer | 96 | 299 × 299 | 1e-4 |
| `hybrid_token_reduction` | PyTorch | InceptionV3 + Transformer | 96 | 299 × 299 | 1e-4 |
| `hybrid_token_reduction_opt` | PyTorch | InceptionV3 + Transformer | 96 | 299 × 299 | 1e-4 |
| `retfound_green` | PyTorch | ViT-S/14 (DINOv2 / RETFound) | 96 | 392 × 392 | 5e-5 (discriminative) |
| `vit_pure` | PyTorch | ViT-B/16 | 32 | 320 × 320 | 1e-4 |

> **Note — `retfound_green`:** uses `vit_small_patch14_reg4_dinov2` backbone with specialized
> retinal pre-training (RETFound-Green). Higher resolution (392 px) preserves fine vascular detail.
>
> **Note — `vit_pure`:** smaller batch size (32) reflects the memory cost of quadratic attention
> over 400 tokens (20 × 20 patches of 16 px each).

---

## Optimizations per Approach

| Approach | AMP | DALI | JIT/Compile | Flash Attn. | EMA | Mixup/CutMix | Token Reduction | LR Scheduler |
|---|---|---|---|---|---|---|---|---|
| `tensorflow_base` | — | — | — | — | — | — | — | — |
| `tensorflow_opt` | ✓ | flag | flag | — | — | ✓ | — | ✓ Cosine+WU |
| `pytorch_base` | — | — | — | — | — | — | — | — |
| `pytorch_opt` | ✓ | ✓ | — | — | ✓ | ✓ | — | ✓ Cosine+WU |
| `hybrid_simple` | flag | — | — | — | flag | — | — | flag |
| `hybrid_token_reduction` | ✓ | — | — | — | — | — | ✓ 50% | ✓ Cosine+WU |
| `hybrid_token_reduction_opt` | ✓ bf16 | — | ✓ | ✓ | ✓ | ✓ | ✓ 50%+WU | ✓ Cosine+WU |
| `retfound_green` | ✓ | — | — | — | ✓ | — | — | ✓ Cosine+WU |
| `vit_pure` | ✓ | — | — | flag | ✓ | — | — | ✓ Cosine+WU |

**Legend:** `✓` = enabled by default · `flag` = available via CLI argument · `—` = not implemented  
**WU** = linear warmup · **bf16** = BFloat16 preferred over FP16

---

## Approach Details

### tensorflow_base

Minimal TensorFlow baseline with no optimizations. Used as reference for framework comparison.

```
Entry:          tensorflow_base/dr_hcpa_v2_2024.py
Backbone:       InceptionV3 (Keras, ImageNet)
Classifier:     Dense(1) after GlobalAveragePooling(2048)
Data pipeline:  tf.data with basic augmentation (flip, rotation)
Scheduler:      None (fixed LR)
```

---

### tensorflow_opt

Optimized TensorFlow version. Adds GPU-accelerated data loading, mixed precision, and advanced
augmentation on top of the baseline.

```
Entry:          tensorflow_opt/dr_hcpa_v2_2024.py
Backbone:       InceptionV3 (Keras, ImageNet)
LR vs base:     3e-3 (higher — cosine scheduler requires larger initial LR)
```

**Optimizations applied:**

| # | Optimization | Details |
|---|---|---|
| 1 | **AMP — FP16** | Reduces memory ~50 %, accelerates Tensor Core ops |
| 2 | **DALI pipeline** (`--use_dali`) | GPU-side JPEG decode + resize, eliminates CPU bottleneck |
| 3 | **XLA JIT compile** (`--jit_compile`) | Fuses graph ops into optimized kernels; measured ~3 976 img/s on H200 |
| 4 | **Mixup** (α=0.3) **+ CutMix** | Regularization via image interpolation/patch mixing |
| 5 | **Focal Loss** (`--use_focal_loss`) | Up-weights hard examples — useful for class imbalance |
| 6 | **Cosine Annealing + Warmup** | LR warms up linearly then decays via cosine schedule |
| 7 | **BatchNorm freezing** (`--freeze_bn`) | Freezes backbone BN layers to stabilize fine-tuning |
| 8 | **Label smoothing** (0.01) | Prevents overconfidence; improves calibration |

---

### pytorch_base

Minimal PyTorch baseline. Equivalent to `tensorflow_base` but using PyTorch + timm.

```
Entry:          pytorch_base/dr_hcpa_v2_2024.py
Backbone:       inception_v3 (timm, ImageNet)
Classifier:     Linear(2048 → 1) after GlobalAveragePooling
Data pipeline:  DataLoader with basic augmentation
Distribution:   DDP
```

---

### pytorch_opt

Optimized PyTorch version. Adds DALI, AMP, EMA, and advanced augmentation.

```
Entry:          pytorch_opt/dr_hcpa_v2_2024.py
Backbone:       inception_v3 (timm, ImageNet)
Optimizer:      AdamW by default; switchable to Adam/SGD/RMSprop/Adadelta via --optimizer
```

**Optimizations applied:**

| # | Optimization | Details |
|---|---|---|
| 1 | **AMP — FP16** | `torch.cuda.amp` GradScaler + autocast |
| 2 | **DALI pipeline** | TFRecord decode directly on GPU |
| 3 | **Gradient clipping** (norm=1.0) | Prevents gradient explosion during fine-tuning |
| 4 | **Cosine Annealing + Warmup** | Smooth LR decay after linear warmup phase |
| 5 | **Mixup** (configurable α) **+ CutMix** | Augmentation regularization |
| 6 | **EMA** | Exponential Moving Average of weights; improves generalization |
| 7 | **Configurable optimizer** | AdamW / Adam / SGD / RMSprop / Adadelta via CLI flag |

---

### hybrid_simple

Minimal CNN + Transformer hybrid. Validates the two-stage architecture without stacking
optimizations.

```
Entry:          hybrid_simple/train.py
Backbone:       InceptionV3 → 2048 features
Architecture:   Linear projection → 4× TransformerEncoderLayer (4 heads) → [CLS] → Linear(1)
Backbone freeze: Yes, for initial N epochs (--backbone_freeze_epochs)
Gradient clip:  norm = 1.0
```

Optional via flag: AMP (`--amp`), EMA (`--ema`), Cosine LR (`--cosine_lr`).

---

### hybrid_token_reduction

Adds token reduction to the hybrid architecture, reducing Transformer attention complexity.

```
Entry:          hybrid_token_reduction/train.py
Backbone:       InceptionV3 → feature sequence
Token selector: Keeps top-k tokens by activation magnitude (keep_ratio = 0.5)
Complexity:     O(n²) → O(k²)  where k = 50 % of n
Memory saving:  ~62.5 % fewer attention operations
```

AMP and Cosine Annealing enabled by default.

---

### hybrid_token_reduction_opt

Fully optimized hybrid — all available optimizations stacked on top of token reduction.

```
Entry:          hybrid_token_reduction_opt/train.py
Backbone:       InceptionV3 → feature sequence
Token selector: keep_ratio = 0.5 with progressive warmup (5 epochs)
```

**Optimizations applied:**

| # | Optimization | Details |
|---|---|---|
| 1 | **Token reduction + warmup** | keep_ratio relaxed gradually over first 5 epochs |
| 2 | **Flash Attention** | IO-aware attention (no full N×N matrix); requires Ampere+ GPU |
| 3 | **AMP — bf16 preferred** | `torch.autocast`; bf16 avoids overflow without GradScaler |
| 4 | **torch.compile** (reduce-overhead) | Triton kernels to minimize Python loop overhead |
| 5 | **Channels Last** (NHWC) | Better Tensor Core throughput for convolutions |
| 6 | **EMA** | Enabled by default |
| 7 | **Mixup** (α=0.2) **+ CutMix** (α=0.5) | Augmentation regularization |
| 8 | **NaN/Inf checks** | Gradient stability check every step; aborts on detection |

---

### retfound_green

Foundation model approach using **RETFound-Green**, a ViT-Small pre-trained on retinal images
with DINOv2 self-supervised learning. No CNN involved.

```
Entry:          retfound_green/train.py
Backbone:       vit_small_patch14_reg4_dinov2  (patch 14 px, 4 register tokens)
Pre-training:   RETFound-Green weights (retina-specialized)
Architecture:   ViT-Small: 12 Transformer layers, 6 heads, embed_dim = 384
Tokens:         784 image patches + 4 register tokens = 788 tokens
Classifier:     Linear(384 → 1) over [CLS] token
```

**Optimizations applied:**

| # | Optimization | Details |
|---|---|---|
| 1 | **Discriminative LR** | Backbone LR = base_lr / 10 (preserves retinal pre-training) |
| 2 | **AMP** | Enabled by default |
| 3 | **Cosine Annealing + Warmup** | Enabled by default |
| 4 | **EMA** | Enabled by default |
| 5 | **LP-FT** | Linear Probe phase (backbone frozen) before full fine-tuning |
| 6 | **Automatic positive weight** | pos_weight derived from class ratio in the dataset |

> Higher input resolution (392 × 392) is required to match the ViT-S/14 patch grid
> (28 patches × 14 px = 392 px).

---

### vit_pure

Pure Vision Transformer (ViT-B/16) — no CNN component. All image processing is done via
patch-based self-attention end-to-end.

```
Entry:          vit_pure/train.py
Backbone:       ViT-B/16 (~86 M parameters)
Architecture:   12 Transformer layers, 12 heads, embed_dim = 768, MLP ratio = 4.0
Tokens:         400 image patches (20×20) + 1 [CLS] = 401 tokens
Classifier:     Linear(768 → 1) over [CLS] token
Batch size:     32  (smaller — quadratic attention over 400 tokens)
```

**Optimizations applied:**

| # | Optimization | Details |
|---|---|---|
| 1 | **AMP** | Enabled by default |
| 2 | **Cosine Annealing + Warmup** | Enabled by default |
| 3 | **EMA** | Enabled by default |
| 4 | **Gradient accumulation** | `--grad_accum N` to simulate larger effective batch size |
| 5 | **Flash / SDPA Attention** (`--flash_attn`) | PyTorch 2.0 scaled_dot_product_attention |
| 6 | **Automatic positive weight** | pos_weight derived from class ratio |

---

## Architecture Comparison

| Approach | Backbone type | Parameters | Tokens | Classifier |
|---|---|---|---|---|
| `tensorflow_base` | CNN (InceptionV3) | ~24 M | — | Dense(1) after GAP(2048) |
| `tensorflow_opt` | CNN (InceptionV3) | ~24 M | — | Dense(1) after GAP(2048) |
| `pytorch_base` | CNN (InceptionV3) | ~24 M | — | Linear(2048 → 1) after GAP |
| `pytorch_opt` | CNN (InceptionV3) | ~24 M | — | Linear(2048 → 1) after GAP |
| `hybrid_simple` | CNN + Transformer | ~28 M | projected seq. | Linear([CLS] → 1) |
| `hybrid_token_reduction` | CNN + Transformer | ~28 M | k = 50 % of seq. | Linear([CLS] → 1) |
| `hybrid_token_reduction_opt` | CNN + Transformer | ~28 M | k = 50 % + warmup | Linear([CLS] → 1) + Flash Attn |
| `retfound_green` | ViT-S/14 (DINOv2) | ~22 M | 784 + 4 = 788 | Linear(384 → 1) on [CLS] |
| `vit_pure` | ViT-B/16 | ~86 M | 400 + 1 = 401 | Linear(768 → 1) on [CLS] |

**GAP** = Global Average Pooling

---

## Base vs Optimized — What Changed

### TensorFlow Base → TensorFlow Opt

| Parameter / Feature | `tensorflow_base` | `tensorflow_opt` |
|---|---|---|
| Learning rate | 5e-4 | 3e-3 |
| Precision | FP32 | AMP FP16 |
| Data pipeline | tf.data (CPU) | DALI (GPU) |
| JIT compilation | — | XLA (flag) |
| Augmentation | Basic | Mixup + CutMix + Label Smoothing |
| LR scheduler | None | Cosine + Warmup |
| Loss | BCE | BCE or Focal (flag) |

### PyTorch Base → PyTorch Opt

| Parameter / Feature | `pytorch_base` | `pytorch_opt` |
|---|---|---|
| Precision | FP32 | AMP FP16 + GradScaler |
| Data pipeline | DataLoader (CPU) | DALI (GPU) |
| LR scheduler | None | Cosine + Warmup |
| Augmentation | Basic | Mixup + CutMix |
| EMA | — | ✓ |
| Gradient clipping | — | norm = 1.0 |
| Optimizer | AdamW (fixed) | Configurable via flag |

### Hybrid Simple → Token Reduction → Token Reduction Opt

| Feature | `hybrid_simple` | `hybrid_token_reduction` | `hybrid_token_reduction_opt` |
|---|---|---|---|
| Token Reduction | — | ✓ 50 % | ✓ 50 % + 5-epoch warmup |
| AMP | flag | ✓ FP16 | ✓ bf16 preferred |
| Flash Attention | — | — | ✓ |
| torch.compile | — | — | ✓ reduce-overhead |
| Channels Last | — | — | ✓ |
| EMA | flag | — | ✓ |
| Mixup + CutMix | — | — | ✓ |
| NaN/Inf checks | — | — | ✓ |
| Cosine LR | flag | ✓ | ✓ |

---

## Running an Approach

Each approach directory contains:

- **`train.py`** (or `dr_hcpa_v2_2024.py`) — main training script
- **`model.py`** — model architecture definition
- **`config.py`** — default hyperparameters
- **`evaluate.py`** — evaluation utilities
- **`utils.py`** — shared helpers
- **`Dockerfile`** / **`Apptainer.def`** — container definitions
- **`*.slurm`** / **`*.oar`** / **`*.sh`** — HPC job submission scripts
- **`requirements`** / **`requirements-ngc.txt`** — Python dependencies

### Example — single GPU

```bash
# TensorFlow approaches
python tensorflow_opt/dr_hcpa_v2_2024.py \
    --data_dir /path/to/data/all-tfrec \
    --epochs 200 \
    --batch_size 96 \
    --jit_compile \
    --use_dali

# PyTorch / Hybrid / ViT approaches
torchrun --nproc_per_node=1 pytorch_opt/dr_hcpa_v2_2024.py \
    --data_dir /path/to/data/all-tfrec \
    --epochs 200 \
    --batch_size 96
```

### Example — multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 hybrid_token_reduction_opt/train.py \
    --data_dir /path/to/data/all-tfrec \
    --epochs 200 \
    --batch_size 96
```

---

## Glossary

| Term | Description |
|---|---|
| **AMP** | Automatic Mixed Precision — FP16 compute with FP32 accumulators |
| **bf16** | BFloat16 — wider dynamic range than FP16; preferred on A100/H100/GH200 |
| **CutMix** | Augmentation: paste a patch from one image into another, mix labels proportionally |
| **DALI** | NVIDIA Data Loading Library — GPU-accelerated image decode and augmentation |
| **DINOv2** | Self-supervised pre-training via teacher-student distillation (Meta AI) |
| **DDP** | DistributedDataParallel — PyTorch multi-GPU training with all-reduce gradient sync |
| **EMA** | Exponential Moving Average of model weights; improves generalization |
| **Flash Attention** | IO-aware attention (tiling + recomputation); O(n) memory instead of O(n²) |
| **Focal Loss** | Cross-entropy variant that down-weights easy examples |
| **GAP** | Global Average Pooling — reduces spatial feature map (H×W×C) to a vector (C) |
| **InceptionV3** | CNN with parallel multi-scale convolutions (1×1, 3×3, 5×5) |
| **JIT / XLA** | Just-In-Time graph compilation to fused GPU kernels |
| **Label Smoothing** | Replaces hard 0/1 labels with soft values; prevents overconfidence |
| **LP-FT** | Linear Probe → Fine-Tuning: freeze backbone first, then unfreeze for full training |
| **Mixup** | Augmentation: linearly interpolate two images and their labels |
| **Token Reduction** | Select top-k informative tokens before Transformer, reducing attention from O(n²) to O(k²) |
| **ViT** | Vision Transformer — splits image into fixed patches processed by self-attention |
| **Warmup** | Linear LR ramp-up at training start to avoid instability with random weights |
