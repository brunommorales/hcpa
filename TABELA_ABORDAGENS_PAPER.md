# Tabela das 8 abordagens — o que é IGUAL vs o que é POR-ARQUITETURA

> Gerada em 2026-07-11 a partir de verificação no código (não das flags): schemas
> por `diff`, receita comum em `tools/common_recipe.sh`, hiperparâmetros nos
> `*/run_g5k_hydra.sh`. Serve de base para a tabela de "setup experimental" do paper.
> **Fora deste estudo: vit_pure.**

---

## 1. Identidade das 8 abordagens

| # | abordagem | família / entrypoint | backbone | img | papel no estudo |
|---|---|---|---|---|---|
| 1 | `pytorch_base` | dr_hcpa_v2_2024.py | InceptionV3 | 299 | CNN em PyTorch eager, sem otimização |
| 2 | `pytorch_opt` | dr_hcpa_v2_2024.py | InceptionV3 | 299 | CNN em PyTorch + `torch.compile` + AMP |
| 3 | `tensorflow_base` | dr_hcpa_v2_2024.py | InceptionV3 | 299 | CNN em TF eager, sem otimização |
| 4 | `tensorflow_opt` | dr_hcpa_v2_2024.py | InceptionV3 | 299 | CNN em TF + XLA (`--jit_compile`) + mixed precision |
| 5 | `retfound_green` | train.py + hybrid_shared | RETFound-Green (DINOv2 ViT-S, `vit_small_patch14_reg4_dinov2`) | 392 | modelo de fundação de retina, fine-tuning |
| 6 | `hybrid_simple` | train.py + hybrid_shared | CNN backbone + encoder ViT | 299 | híbrido CNN+Transformer, sem redução de tokens |
| 7 | `hybrid_token_reduction` | train.py + hybrid_shared | CNN + ViT + `token_selector` | 299 | híbrido com redução de tokens (keep_ratio) |
| 8 | `hybrid_token_reduction_opt` | train.py + hybrid_shared | idem #7 | 299 | versão "opt": keep_ratio 0.5, channels_last, mixup/cutmix, amp_dtype |

---

## 2. O que é mantido IGUAL nas 8 (base da comparabilidade)

Verificado no código, não nas flags:

| dimensão | valor comum | onde se garante |
|---|---|---|
| **batch size** | 96 | run scripts das 8 |
| **épocas** | 200 | `HCPA_EPOCHS` / `TARGET_EPOCHS` |
| **dataset + split** | `all-tfrec-v2`: train 9350 / val 1650 / test 4816, **por paciente, 0 vazamento** | `tools/g5k_run_common.sh` (default v2 + guarda anti-`val==test`) |
| **pipeline de dados** | cache in-memory da imagem decodificada + shuffle full (buffer ≥ 9350) | F1–F4, niveladas nas 8 |
| **DALI** | desligado nas 8 (decode na CPU) | `enable_dali=0` / hardcoded no tf_opt |
| **schema de métricas** | **33 colunas idênticas, mesma ordem** | `COMMON_CSV_FIELDS` == `METRICS_CSV_FIELDS` (diff = ∅) |
| **seleção de modelo** | best-checkpoint pela **val_auc**; teste avaliado 1× no fim | val→test nas 8 |
| **janela de energia** | contador NVML, fase de treino kernel-only (fecha antes da validação) | `EnergyScope` (PyTorch) / `on_test_begin` (TF) |
| **amostragem de telemetria** | thread de fundo, 200 ms, fora do laço de treino | `gpu_energy.py` (uniforme nas 8) |
| **1 GPU** | `HCPA_VISIBLE_GPUS=0` (evita MirroredStrategy multi-GPU no TF) | runner |

---

## 3. O eixo BASE vs OPT (a variável independente do estudo)

Não nivelar — é o que o paper mede.

| recurso | base (`*_base`) | opt (`*_opt`, híbridos) |
|---|---|---|
| precisão mista (AMP / mixed_precision) | OFF (`--disable_amp`) | ON |
| cosine schedule | OFF (`--disable_cosine`) | ON |
| compilação de grafo | — | PyTorch `torch.compile` · TF XLA (`--jit_compile`) |
| `channels_last` | — | só `hybrid_token_reduction_opt` |

**`pytorch_opt` e `tensorflow_opt` compartilham a receita comum** (`tools/common_recipe.sh`):
InceptionV3 · 299px · batch 96 · 200 ép · AdamW (β 0.9/0.999, eps 1e-7) · weight decay
head 1e-4 / fine-tune 1e-5 · grad clip 1.0 · warmup 5 · cosine · min_lr 1e-6 · **EMA off** ·
mixup/cutmix/label_smoothing/focal = 0 (zerados para isolar o runtime). A única diferença
entre os dois é o runtime → **XLA vs torch.compile**.

---

## 4. Hiperparâmetros POR-ARQUITETURA (declarar como escolha, não bug)

| parâmetro | CNN (base/opt) | retfound_green | híbridos |
|---|---|---|---|
| learning rate | 5e-4 | **5e-5** | **1e-4** |
| img_size | 299 | **392** | 299 |
| freeze | 0 | 0 | **3 épocas** (`freeze_backbone_epochs`) — aquece a cabeça |
| justificativa | — | modelo de fundação exige LR pequeno e resolução nativa do ViT | híbrido aquece a cabeça antes de liberar o backbone |

> Regra para o texto: "cada arquitetura teve LR/resolução/freeze ajustados ao seu
> regime de convergência; todo o resto (dado, batch, épocas, seleção, métricas,
> janelas de medição) é idêntico".

---

## 5. Justificativa das métricas — dois níveis

**🟢 Métrica-exata (defensável número a número):** `*_energy_j` (contador NVML de
hardware), `*_elapsed_s` / `total_train_time_s` / `*_throughput_img_s`
(`perf_counter`+`synchronize`), `*_gpu_mem_peak_mb` (NVML.used, pico, nó exclusivo),
e as clínicas `val/test_auc/sens/spec/spec_at_sens95/precision/f1` (exatas, sobre
val/test sem augmentation).

**🟡 Diagnóstico (não ancorar claims):** `train_avg_power_w` (média amostrada,
subestima picos → usar energia÷tempo), `train_gpu_util_pct` / `train_mem_util_pct`
(ocupação temporal do driver, ~1 s), `train_busy_time_s` (derivada do util% →
**figuras bloqueadas até validar contra CUPTI**).

---

## 6. Ressalvas que ainda mudam números (ver VALIDACAO_FINAL.md)

1. **Energia GH200 = módulo Grace+Hopper**, não GPU-only (probe P10: +195 W só com a
   CPU). Não comparar joules GH200 vs A100 (GPU-only) sem rótulo/isolamento.
2. **Re-rodada 8×10×200 pendente** — os CSVs em `new_results/` são do schema antigo
   (55 col, pré-fix) e estão superados.
3. **util%/busy_time** não validados contra CUPTI.
