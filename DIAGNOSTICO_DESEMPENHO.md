# Diagnóstico técnico — desempenho e coerência das 8 abordagens HCPA

> Escopo: as 8 abordagens do estudo (vit_pure fora). Foco em **o que atrapalha o
> desempenho** (throughput, latência, ociosidade da GPU) e na **coerência das
> métricas**. Nenhuma implementação — só diagnóstico.
>
> Base de evidência: leitura do código + traces CUPTI + resultados do run
> 1×200 épocas na A100 (dataset v2 corrigido, 2026-07-11).

---

## PARTE 0 — Resultados honestos que embasam o diagnóstico

Primeira medição com o split corrigido (val* seleciona, test* reporta):

| abordagem | best_ep | val_auc | test_auc | spec@95 | tempo (s) |
|---|---|---|---|---|---|
| tensorflow_opt | 125 | 0.9673 | 0.9816 | 0.9137 | 2448 |
| pytorch_opt | 98 | 0.9709 | 0.9845 | 0.9365 | **1883** |
| tensorflow_base | 115 | 0.9663 | 0.9839 | 0.9267 | 3780 |
| pytorch_base | 103 | 0.9744 | 0.9821 | 0.9177 | 3181 |
| hybrid_simple | 156 | 0.9563 | 0.9803 | **0.9812** | 4619 |
| hybrid_token_reduction | 86 | 0.9664 | 0.9792 | **0.9747** | 4156 |
| hybrid_token_reduction_opt | 75 | 0.9663 | 0.9835 | **0.9851** | 3033 |
| retfound_green | **9** | 0.9789 | 0.9843 | 0.9640 | 4529 |

**Leituras imediatas (n=1, indicativas):**
1. Os best-epochs agora são **tardios (75–156)**, exceto o retfound (época 9). Isso
   confirma que o antigo "83% de energia desperdiçada" era artefato do val=test.
2. **Os híbridos ganham em spec@95 (0.975–0.985)** contra os CNN puros (0.91–0.93).
   A especificidade em 95% de sensibilidade é a métrica clínica que mais importa —
   e é onde a arquitetura CNN-ViT se separa. Resultado publicável.
3. **pytorch_opt** é o mais rápido (1883 s) com test_auc alto (0.9845): melhor
   custo computacional. **retfound** converge na época 9 (é modelo de fundação).

⚠️ n=1 — tendências, não conclusões. A re-rodada n=10 confirma.

---

## PARTE 1 — Achados TRANSVERSAIS de desempenho

> ⚠️ **CORREÇÃO (2026-07-11)**: D1/D2 abaixo estavam parcialmente ERRADOS. Ao
> nivelar o pipeline descobri que o `tensorflow_opt` tinha o `.cache()` DESLIGADO
> (`--cache_dir none`) e o `tensorflow_base` não tinha cache nenhum — ou seja,
> **NINGUÉM cacheava**, nem o TF. O gargalo do PyTorch era eficiência de pipeline
> (workers Python vs `tf.data` C++), não ausência de cache relativa ao TF. Tudo
> isso foi corrigido: cache ligado nas 8. Ver `AUDITORIA_PIPELINE_NIVELADO.md`.

### D1 🔴 O PyTorch decodifica JPEG + aumenta na CPU a CADA acesso, sem cache
```python
# pytorch_opt/dr_hcpa_v2_2024.py:475 (RetinaTFRecordDataset.__getitem__)
img = Image.open(io.BytesIO(img_bytes)).convert("RGB")   # decode JPEG na CPU (PIL)
img = self._apply_augmentations(img)                      # augment na CPU (PIL)
```
**Toda época, toda imagem, é redecodificada e reaumentada na CPU** pelos workers do
DataLoader. Não há cache do resultado. Isso é o gargalo de dados do PyTorch e é
exatamente o que o CUPTI mediu: **21% do tempo de época a GPU fica esperando o
host** (`cpu_wait`). Afeta: `pytorch_base`, `pytorch_opt`, e os 4 híbridos/retfound
(todos usam `build_torch_loader`).

Agravantes:
- `num_workers = min(8, cpu_count)` — se a A100 tiver poucos cores por GPU, 8
  workers podem não dar conta do decode de InceptionV3 a 299px.
- Decode com **PIL** (Python puro) é lento; `Pillow-SIMD`, `cv2`, ou `torchvision.io`
  (nativo) seriam 2–4× mais rápidos.
- Augment em PIL (`ImageEnhance`, `ImageOps`) também é CPU-bound.

### D2 🟡 O TensorFlow cacheia decodificado — mas com shuffle fraco
```python
# tensorflow_opt: cache in-memory APÓS decode; augment e shuffle DEPOIS do cache
dataset = dataset.cache()                      # ~11 GB float32 na RAM
dataset = dataset.shuffle(2048, ...)           # <-- buffer de 2048 de 11000 imgs
```
O cache é **a favor** do TF: a partir da época 2 ele lê tensores pré-decodificados
da RAM (sem I/O nem decode). É por isso que o TF em regime é mais rápido por época
que o PyTorch — **mas essa vantagem é do PIPELINE, não do modelo**.

Porém o **shuffle buffer é 2048 de 11000** (18%). O embaralhamento é parcial: em
cada época, uma imagem só se mistura com as ~2048 vizinhas na ordem do TFRecord. O
PyTorch usa shuffle **completo** (o sampler embaralha os 11000 índices). Isso é uma
**assimetria de regularização**: o TF vê lotes menos diversos, o que pode afetar
convergência e a comparação clínica. Buffer ideal ≈ tamanho do dataset.

### D3 🟠 A assimetria de pipeline contamina energia E tempo
Consequência combinada de D1+D2: o TF (cache) e o PyTorch (decode+augment por época)
têm **pipelines de dados fundamentalmente diferentes**. Isso afeta diretamente:
- **tempo por época** (TF mais rápido em regime),
- **ociosidade da GPU** (`cpu_wait` do PyTorch alto por causa do decode),
- **energia** (GPU esperando gasta menos, mas o tempo total sobe).

Ou seja, parte da diferença "tensorflow_opt vs pytorch_opt" **não é XLA vs
TorchInductor** — é **cache vs decode-por-época**. Já está declarado como limitação,
mas o CUPTI permite QUANTIFICAR: os 21% de `cpu_wait` do PyTorch são majoritariamente
o pipeline, não o compute.

### D4 🟡 torch.compile só em 2 das 6 abordagens PyTorch
```
pytorch_opt                : torch.compile (reduce-overhead)  ✓
hybrid_token_reduction_opt : torch.compile                    ✓
pytorch_base               : eager
hybrid_simple              : eager
hybrid_token_reduction     : eager
retfound_green             : eager
```
Esperado (os "opt" compilam, os "base" não — é o ponto do estudo). Mas para
**desempenho puro**, as 4 em eager deixam ~1,5–1,8× de velocidade na mesa (medimos
1,77× no par pytorch). Não é um bug; é uma escolha de design que o paper deve
enquadrar como "base = referência não-otimizada".

### D5 🟡 Sem `channels_last` em nenhuma abordagem PyTorch
Nenhuma usa `memory_format=torch.channels_last`. Em GPUs com Tensor Cores (A100/
Hopper), `channels_last` + AMP tipicamente dá 1,1–1,3× em CNNs (InceptionV3 é
convolucional). É ganho grátis que ninguém está pegando — igual para todas, então
não distorce a comparação, mas todas rodam abaixo do teto.

### D6 🟡 TF32 ligado no PyTorch, ausente/implícito no resto
`pytorch_opt`/`base` ativam `allow_tf32=True` (matmul e cudnn). `hybrid_token_
reduction_opt` usa `set_float32_matmul_precision("high")` (equivalente). Os outros
hybrids e o retfound **não setam TF32** explicitamente — dependem do default do
PyTorch (que mudou entre versões). Sob AMP a maior parte já é FP16, então o impacto
é pequeno, mas é uma inconsistência de configuração entre abordagens.

### D7 🟢 O que está BEM (não mexer)
- **PyTorch**: `running_loss` acumula em GPU (`torch.zeros(device).add_`), só faz
  `.item()` no fim da época — **não** sincroniza por batch. `pin_memory=True`,
  `non_blocking=True`, `persistent_workers=True`, `prefetch_factor=4`. Bem feito.
- **`loss_finite_check_interval=0`** por padrão → sem `.item()` de verificação por
  batch. Bom.
- **runtime_profiler** é no-op quando desligado (yield direto). Sem overhead em
  produção.
- **TF**: `num_parallel_calls=AUTOTUNE`, `prefetch(AUTOTUNE)`, staging no NVMe local.

---

## PARTE 2 — Por abordagem

### tensorflow_opt
- **Pipeline**: cache in-memory (rápido) + XLA. Regime ~2,9 s/época (o mais rápido).
- **Problema**: shuffle 2048<<11000 (D2). `freeze_epochs=1` + `fine_tune_at 0` —
  hoje alinhado, mas a 1ª época congelada gera um degrau de tempo/energia que deve
  ser excluído do regime.
- **spec@95 = 0.9137** — o mais baixo das 8. Investigar: pode ser o shuffle fraco
  ou a calibração da cabeça.

### pytorch_opt
- **O melhor custo computacional**: 1883 s, test_auc 0.9845, spec@95 0.9365.
- **Gargalo**: decode+augment CPU por época (D1) → 21% cpu_wait. Se resolvido
  (cache ou decode nativo), cairia abaixo de 1500 s.
- torch.compile **confirmado ativo** (reduce-overhead / CUDA graphs).

### tensorflow_base
- **Sem XLA** (é base). Regime lento (3780 s total). Mesmo pipeline cache do opt.
- Coerente como baseline. spec@95 0.9267 (melhor que o tensorflow_opt — curioso,
  vale investigar se o XLA muda a numérica da cabeça).

### pytorch_base
- **Eager + decode CPU**: o mais lento dos CNN (3181 s) pela soma dos dois.
- **val_auc 0.9744** — o mais alto entre os CNN. Boa referência não-otimizada.

### hybrid_simple
- **spec@95 0.9812** — excelente. Mas **best_ep 156** (converge muito devagar) e o
  mais lento (4619 s): CNN backbone + transformer, tudo em eager, decode CPU.
- **freeze_backbone_epochs=3**: os 3 primeiros epochs com backbone congelado.
- Candidato nº1 a ganho de desempenho (eager→compile + cache dados).

### hybrid_token_reduction
- Igual ao simple + redução de tokens (keep_ratio 0.5). Mais rápido que o simple
  (4156 s) — a redução de tokens funciona. spec@95 0.9747.
- Eager, decode CPU: mesmos gargalos.

### hybrid_token_reduction_opt
- **A melhor spec@95 (0.9851)** e best_ep precoce (75). Único hybrid com
  torch.compile + `set_float32_matmul_precision`. 3033 s (bem mais rápido que os
  outros hybrids).
- **Cosine agora ligado** (P4). Mostra o efeito da otimização no hybrid.

### retfound_green
- **Converge na época 9** (modelo de fundação pré-treinado DINOv2). val_auc 0.9789
  (o melhor). Mas o **mais lento por época** (4529 s / 200 ep): ViT-S @ **392px**
  (784 tokens) é caro.
- **freeze_epochs=0**: treina o backbone inteiro desde o início. Para um modelo de
  fundação, um freeze inicial + warmup poderia estabilizar e economizar — mas
  **não mexer** (é baseline importado; a config é a original).
- Eager (sem compile) + decode CPU + 392px = o teto de custo do estudo.

---

## PARTE 3 — As coletas de métricas estão corretas e coerentes?

**Sim, após as correções desta semana.** Auditadas a fundo (12 bugs corrigidos,
schema unificado de 33 colunas idêntico nas 8, validado pelo `smoke_check` em 8/8).
Estado por métrica:

| métrica | fonte | correta? |
|---|---|---|
| `train_energy_j` / `val_energy_j` | contador HW NVML (`nvmlDeviceGetTotalEnergyConsumption`), janela kernel-only por fase | ✅ exata; val_energy agora medido no TF (era None) |
| `train_elapsed_s` / `total_train_time_s` | `perf_counter` + `cuda.synchronize` nas bordas | ✅ os dois frameworks sincronizam agora |
| `train_avg_power_w` | amostragem NVML 200 ms, thread de fundo | ✅ diagnóstico; usar E/t para exatidão |
| `train_gpu_util_pct` / `busy_time_s` | `nvmlUtilizationRates`, thread de fundo | ⚠️ ocupação temporal (não capacidade); OK relativo, mas validar vs CUPTI |
| `train_gpu_mem_peak_mb` | `nvmlMemoryInfo().used` (pico) | ✅ memória real, não caching allocator |
| `val_auc` (por época) | roc_auc_score / rank-sum EXATO toda época | ✅ P3 corrigido (TF era binado 4/5 épocas) |
| `val_spec_at_sens95` (por época) | derivado da ROC no ponto de 95% sens | ✅ agora existe por época nas 8 |
| `test_auc` / `test_spec_at_sens95` | teste final em **test\*** com best-ckpt de **val\*** | ✅ P1 corrigido — número honesto |
| CUPTI `busy_time` / `cpu_wait` | união dos intervalos de kernel | ✅ exato; h2d/d2h separados de d2d/memset (bug 12) |

**Coerência entre as 8**: schema idêntico, mesma janela de medição (batches →
métricas → fecha), amostragem em thread de fundo com o mesmo intervalo, uma GPU só,
114 batches uniformes. As métricas **são comparáveis** entre as 8.

### Ressalvas ainda abertas nas métricas (não são erros, são limites)
1. **`gpu_util_pct` / `busy_time_s`** são estimativas do NVML (ocupação temporal).
   As figuras que dependem delas seguem **bloqueadas** até validar contra o
   `busy_time` exato do CUPTI. O CUPTI já dá o número certo; falta a comparação
   sistemática (Fase B).
2. **Energia é GPU-only** — não inclui CPU, DRAM do host, refrigeração. Declarar
   "energia da GPU", não "energia do treino".
3. **Domínio de energia no GH200** — na A100 confirmamos GPU-only (idle 58 W). No
   GH200 (Grace+Hopper) ainda não foi verificado se o contador inclui a Grace.
   **Isto muda a magnitude absoluta de todos os joules do paper** se incluir a CPU.
4. **`avg_power_w` amostrado** subestima picos (janela 200 ms). Já usamos
   `E/t` como a potência exata nos gráficos; o amostrado é só diagnóstico.

---

## PARTE 4 — Prioridade das melhorias (se o objetivo fosse desempenho)

| # | melhoria | ganho estimado | afeta |
|---|---|---|---|
| 1 | Cache do dataset decodificado no PyTorch (ou decode nativo `torchvision.io`) | −20% tempo (mata o cpu_wait) | 6 abordagens PyTorch |
| 2 | Shuffle buffer do TF = tamanho do dataset | convergência/regularização | 2 TF |
| 3 | `channels_last` + AMP nos CNN PyTorch | 1,1–1,3× | 4 PyTorch CNN |
| 4 | Pillow-SIMD ou cv2 no decode | 2–4× no decode CPU | 6 PyTorch |
| 5 | TF32 explícito e uniforme nas 8 | consistência | hybrids/retfound |

⚠️ **Cuidado metodológico**: aplicar essas melhorias **muda a energia e o tempo
medidos** — que são o objeto do estudo. Se o objetivo é comparar as abordagens
COMO ESTÃO (cada uma no seu design), não se deve otimizar. Se o objetivo é o teto
de eficiência, aí sim. **São perguntas diferentes** — decidir antes de tocar.

E o `retfound_green` é **baseline importado**: só instrumentação e parâmetros de
entrada. Nada de cache/compile/channels_last nele.
