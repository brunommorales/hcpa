# Plano — estudo de eficiência energética (8 abordagens, GH200)

> **STATUS: implementado em 2026-07-09.** Ver `## Execução` no fim do documento.
> Escopo: 8 abordagens (vit_pure fora).
>
> ⚠️ **Três premissas deste plano estavam ERRADAS** e foram corrigidas ao
> inspecionar o código. Elas estão riscadas abaixo, com a correção ao lado.

## Perguntas de pesquisa que guiam TUDO

- **Q1.** O modelo mais rápido é o que gasta menos energia?
- **Q2.** Qual das 8 abordagens tem o melhor **custo clínico por energia**, e **quanta
  energia ela desperdiça** ao longo das 200 épocas?

Toda métrica usada nessas respostas precisa ser coletada **por época**.

---

## 0. Achados que mudam o plano (medidos, não suposições)

### 0.1 O EMA do tensorflow_opt é desperdício 100% puro
- O EMA liga na época `int(0.6×200) = 120`.
- O **best-checkpoint** fica na época **~19–29** (log: `[BEST] Restaurado best.ckpt: epoch=19`).
- No teste, `applied_ema = False if loaded_best_ckpt else ...` → como o best.ckpt **sempre** carrega,
  **os pesos do EMA nunca são usados**.
- Custo: **+6,6 % de energia por época** nas épocas 120–199 ≈ **7,6 kJ** (2,5 % do total).
- **É exatamente isso que faz a energia do tensorflow_opt não ser constante**: não há deriva,
  há um **degrau de +6,6 % na época 120**.

| abordagem | ep 5–30 | 90–119 | 120–150 | deriva pré-120 | salto no 120 |
|---|---|---|---|---|---|
| tensorflow_opt | 1445 J | 1450 J | 1546 J | +0,3 % | **+6,6 %** |
| pytorch_opt | 2255 J | 2226 J | 2205 J | −1,3 % | −0,9 % |
| pytorch_base | 4616 J | 4633 J | 4632 J | +0,4 % | 0,0 % |

### 0.2 Desperdício energético pós-best-epoch (resposta direta à Q2)
| abordagem | best_ep | E até best | E total | **desperdiçado** |
|---|---|---|---|---|
| tensorflow_opt | 29 | 50,7 kJ | 305,3 kJ | **83,4 %** |
| pytorch_opt | 27 | 88,2 kJ | 469,0 kJ | **81,2 %** |
| pytorch_base | 115 | 539,7 kJ | 928,2 kJ | **41,9 %** |

### 0.3 tensorflow_opt e pytorch_opt NÃO são comparáveis hoje
| dimensão | tensorflow_opt | pytorch_opt |
|---|---|---|
| precisão | FP16 (mixed_precision) | FP16 (AMP) ✔ iguais |
| JIT | XLA (`--jit_compile`) | `torch.compile` |
| **pipeline de dados** | **DALI (decode na GPU)** | **DataLoader CPU (8 workers)** |
| **grad checkpointing** | **`--recompute_backbone`** | não |
| **EMA** | **sim (ep. 120)** | não |
| learning rate | **0,003** | **5e-4** |
| freeze_epochs | 1 | 0 |
| mixup / cutmix | **0,3 / 0,6** | 0 / 0 |
| label_smoothing | 0,1 | 0 |
| focal_gamma / pos_weight | 2,0 / 2,0 | 0 / 1,0 |
| fundus_crop_ratio | 0,9 | 1,0 |
| TTA views | 2 | 1 |

### 0.4 ⚠️ FALHA METODOLÓGICA: a instrumentação não é uniforme entre frameworks
- **TF** (`GpuMemoryEnergyCallback._sample`): **4 chamadas NVML a CADA batch**
  → 115 batches × 4 = **460 chamadas/época**, síncronas no laço de treino.
- **PyTorch dr_hcpa**: thread de fundo a cada 200 ms → ~14 amostras × 4 = **56 chamadas/época**.
- **train.py (hybrids/retfound)**: também **por batch** (~4 chamadas/batch).

Consequência: o TF sofre **~8× mais overhead de host** que o PyTorch dr_hcpa —
**exatamente no lugar onde medimos ociosidade da GPU**. Isso pode explicar parte dos
48 % de util do tensorflow_opt vs 70 % do pytorch_opt. **O `busy_time` atual não é
comparável entre frameworks.**

**Conclusão:** `fig_time_busy_idle`, `fig_gpu_util`, `fig_busy_time` **não devem ser
usados como resultado** até serem validados e a instrumentação uniformizada.

---

## 1. Ferramentas de kernel — precisamos de outra plataforma?

**Não.** Tudo roda no **mesmo nó GH200**, e os gráficos saem no nosso pipeline Python/matplotlib.

| Ferramenta | O que dá | Onde | Custo |
|---|---|---|---|
| **CUPTI** (Activity API) | timestamps de início/fim de **cada kernel** | vem com o CUDA | médio |
| **`torch.profiler`** | wrapper de CUPTI; eventos + trace Chrome | PyTorch | médio |
| **`tf.profiler.experimental`** | idem para TF | TF | médio |
| **Nsight Systems (`nsys`)** | timeline completa, **sem alterar código** | binário; **conferir se está na imagem** | alto (trace grande) |
| **DCGM** (`SM_ACTIVE`, `TENSOR_ACTIVE`) | ocupação real de SM (muito melhor que `util%`) | daemon; **conferir disponibilidade** | baixo |
| CUDA events | tempo de uma região no stream — **não** dá timeline | qualquer | baixo |

⚠️ **Cuidado técnico:** o tempo GPU-ativa é a **união dos intervalos** dos kernels, **não a soma** —
kernels concorrentes em streams diferentes se sobrepõem.

⚠️ **Verificar antes:** `nsys` e `dcgmi` podem não estar na imagem `ubuntugh2404-arm64-big`.
CUPTI e os profilers dos frameworks são garantidos.

---

## 2. Tornar tensorflow_opt e pytorch_opt comparáveis

### 2.1 Decisão de projeto (precisa da tua escolha)
Há duas leituras possíveis, e elas respondem perguntas diferentes:

- **(A) Comparação de framework/runtime** — receita **idêntica**, só o framework muda.
  Isola a eficiência do runtime (XLA vs torch.compile, DALI vs DataLoader).
- **(B) Melhor esforço por framework** — cada um com sua melhor receita.
  Mede a eficiência prática que um usuário obteria.

Você pediu (A): *"deixar os dois exatamente iguais, mantendo o FP16 nos dois"*.

### 2.2 Receita comum proposta (opção A)
| parâmetro | valor comum | nota |
|---|---|---|
| modelo / resolução / batch | InceptionV3 / 299 / 96 | já iguais |
| precisão | **FP16 (AMP / mixed_precision)** | já iguais ✔ |
| JIT | **ligado nos dois** (XLA / torch.compile) | é a "otimização" que define o `_opt` |
| learning rate | **um único valor** | ⚠️ ver 2.3 |
| warmup / cosine / min_lr | 5 / cosine / 1e-6 | já iguais |
| freeze_epochs | **0** | tirar o freeze do TF |
| mixup / cutmix | **0 / 0** | tirar do TF |
| label_smoothing / focal / pos_weight | **0 / 0 / 1,0** | tirar do TF |
| fundus_crop_ratio | **1,0** | tirar o crop do TF |
| TTA views | **1** | tirar do TF |
| grad checkpointing | **desligado nos dois** | confunde energia (recomputa ativações) |
| **EMA** | **desligado** | ver §3 |
| pipeline de dados | ⚠️ **não alinhável** | ver 2.4 |

### 2.3 Learning rate — precisa de cuidado, não de um chute
LR 0,003 (TF) vs 5e-4 (PyTorch) é 6×. Copiar um para o outro pode **quebrar a convergência**
de um deles e invalidar a comparação clínica.

**Proposta:** mini-varredura de LR (3 valores: 5e-4, 1e-3, 3e-3), **1 run cada**, 60 épocas,
nos dois frameworks. Escolher o LR que dá a melhor `val_auc` **em ambos** (ou o melhor par
por framework, documentando). Custo estimado: ~6 runs curtos ≈ 1,5 h de GPU.

### 2.4 O confound que NÃO dá para eliminar (e precisa ser declarado)
- **DALI (decode na GPU)** vs **DataLoader CPU**: pipelines fundamentalmente diferentes.
  Isso afeta diretamente a ociosidade da GPU — que é o objeto do estudo.
- Alinhar exigiria portar um dos dois, o que é um projeto por si só.

**Recomendação:** manter os dois pipelines, **declarar como limitação**, e medir a
ociosidade com CUPTI (§5) — que vai mostrar exatamente quanto de cada pipeline a GPU espera.
Alternativa mais rigorosa (custosa): rodar ambos com o **mesmo** pipeline (DALI nos dois).

---

## 3. EMA: remover

**Evidência:** nunca é usado no teste (best.ckpt sempre carrega), custa 2,5 % da energia total,
e o best-checkpoint fica 90 épocas **antes** de ele ligar.

**Recomendação:** **remover o EMA** do tensorflow_opt.

**Alternativa científica (se quiser publicá-lo como ablação):** rodar 3 configurações
(EMA off / EMA on + best.ckpt / EMA on + pesos EMA no teste) e mostrar que o EMA não
melhora a AUC de teste — transformando o achado em resultado, não em bug.

---

## 4. Auditoria: quais métricas e gráficos são necessários?

### 4.1 Métricas — manter, adicionar, remover

**Essenciais por época (Q1 e Q2):**
| métrica | por quê |
|---|---|
| `val_auc` | trajetória clínica, detecta o best-epoch |
| 🆕 `val_spec_at_sens95` | **falta hoje**: só existe no teste. Q2 pede custo *clínico* por época |
| `train_energy_j` | energia por época |
| `train_elapsed_s` | tempo por época |
| `gpu_mem_peak_mb` | footprint |
| 🆕 `busy_time` **exato via CUPTI** | substitui a estimativa NVML |
| 🆕 `cpu_wait_time` (= elapsed − busy) | a espera pela CPU, medida e não estimada |

**Por run (final):** `test_auc`, `test_spec`, `spec@95` (derivado do ROC), `total_energy`, `total_time`, `best_auc_epoch`.

**Manter como diagnóstico barato:** `mem_util_pct` (vem no mesmo call NVML), `train_loss`.

**Reduzir/remover:**
| métrica | ação | motivo |
|---|---|---|
| `train_auc/precision/f1/sens/spec` | **não reportar** | refletem lotes com mixup/cutmix — enganosos |
| `avg_power_w` (amostrada) | **reduzir frequência** | redundante com `energia/tempo`; **é a causa do efeito observador** |
| `val_busy_time_s`, `val_gpu_util_pct` | remover | não usados em Q1/Q2 |
| `throughput_img_s` | manter (derivado, grátis) | |

### 4.2 Gráficos — 19 hoje. Núcleo do estudo: 7.

**Núcleo (respondem Q1/Q2):**
1. 🆕 **`fig_speed_vs_energy`** — dispersão tempo × energia (**a resposta da Q1**)
2. 🆕 **`fig_energy_wasted`** — energia até o best-epoch vs desperdiçada (**a resposta da Q2**)
3. `fig_pareto_energy_spec95` — custo clínico × energia (**o Pareto da Q2**)
4. `fig_cec_efficiency` — energia para atingir 99 % da AUC
5. `fig_auc_per_epoch` — trajetória clínica + pico (já feito)
6. `fig_energy_total` + `fig_traintime_breakdown` — os totais
7. 🆕 **piano roll** — onde a GPU calculou e onde esperou (§5)

**Diagnósticos úteis (manter, não são resultado):** `fig_energy_per_epoch`, `fig_memory`, `fig_spec95`, `fig_auc`.

**Bloqueados até validação (§0.4):** `fig_time_busy_idle`, `fig_gpu_util`, `fig_busy_time`.

**Candidatos a remover (redundantes):**
- `fig_energy_overview` (duplica `energy_total` + `energy_per_epoch`)
- `fig_throughput` (é ~1/tempo)
- `fig_power` (é energia/tempo)
- `fig_time_per_epoch` (contido no breakdown)
- `fig_best_auc_epoch` (vira painel C do `fig_auc_per_epoch`)
- `fig_mem_util` (diagnóstico, não resultado)

---

## 5. Plano de instrumentação de kernel (o "piano roll")

### Fase A — teste do efeito observador (barato, decisivo) ⏱️ ~1 h GPU
1. Adicionar `HCPA_SAMPLE_EVERY_N_BATCHES` (default 1 → compatível).
2. Rodar **1 época** de tensorflow_opt e pytorch_opt com N = 1, 20 e amostragem desligada.
3. Medir `train_elapsed_s` e `util%`.
- **Se o tempo cair com N maior** → nossa instrumentação inflava a ociosidade →
  refazer as figuras de busy/idle e **uniformizar** a amostragem (thread de fundo em todos).

### Fase B — GPU-busy exato via CUPTI ⏱️ ~2 h GPU
- PyTorch: `torch.profiler` (`ProfilerActivity.CUDA`).
- TF: `tf.profiler.experimental`.
- Extrair intervalos de kernel → **união** dos intervalos → `busy_time` exato.
- Comparar com o `busy_time` do NVML → **quantificar o erro da estimativa atual**.

### Fase C — piano roll ⏱️ ~2 h GPU
- Capturar timeline de kernel em **épocas representativas** (ex.: 5, 50, 120, 199) —
  não nas 200 (volume de trace).
- Figura: eixo x = tempo dentro da época; eixo y = época; **preenchido = kernel rodando**,
  **vazio = GPU parada**. Decompor a lacuna: cópia H2D, espera do dataloader, overhead de host.

⚠️ **Por que não dá para as 200 épocas com resolução de kernel:** o trace fica enorme.
E **amostrar NVML mais rápido não resolve** — o `util%` tem janela interna do driver (~1 s),
então amostrar a 2 ms só reamostra o mesmo número.

### Fase D — `busy_time` exato por época, para as 200 ⏱️ overhead a medir
- Agregação CUPTI leve: registrar só início/fim de kernel e computar a união por época,
  **sem despejar trace**.
- Se o overhead for aceitável (<2 %), vira a métrica oficial e aposenta a estimativa NVML.

---

## 6. Custo e cronograma

**Re-rodada final (8 abordagens × 10 runs, receita alinhada):**

| abordagem | tempo/run (s) | ×10 |
|---|---|---|
| tensorflow_base | 3392 | 9,4 h |
| tensorflow_opt | 1225 | 3,4 h |
| pytorch_base | 1901 | 5,3 h |
| pytorch_opt | 1331 | 3,7 h |
| hybrid_simple | 2970 | 8,3 h |
| hybrid_token_reduction | 2789 | 7,7 h |
| hybrid_token_reduction_opt | 2011 | 5,6 h |
| retfound_green | 2955 | 8,2 h |
| **total** | | **≈ 51,6 h de GPU** |

Em **3 nós hydra em paralelo** ≈ **17 h de relógio** → 2–3 janelas noturnas.
Somar: varredura de LR (~1,5 h) + Fases A–C (~5 h).

**Ordem sugerida:**
1. Fase A (efeito observador) — pode invalidar figuras existentes
2. Uniformizar a amostragem + adicionar `val_spec_at_sens95` por época
3. Varredura de LR + decisão da receita comum
4. Fase B/C (CUPTI, piano roll) em 1 abordagem de cada framework
5. Re-rodada final das 8
6. Fase D se o overhead permitir

---

## 7. Riscos e limitações a declarar no paper

1. **DALI vs DataLoader CPU** — confound não eliminável sem reescrever um pipeline.
2. **XLA vs torch.compile** — "JIT ligado nos dois" não significa JIT equivalente.
3. **`util%` do NVML é ocupação temporal**, não capacidade: um kernel usando 1 SM conta como "ocupado".
4. **Efeito observador** — a própria medição perturba o que mede; precisa ser quantificado (Fase A).
5. **Nós físicos diferentes** dão ~5 % de diferença de energia (já medido) → fixar o nó por abordagem.
6. **1 GPU, 1 dataset, 1 arquitetura** — as conclusões não generalizam automaticamente.

---

# CORREÇÕES AO PLANO (achadas ao inspecionar o código, 2026-07-09)

O plano acima foi escrito a partir das **flags dos run scripts**. O código as
ignora em três lugares. Corrigido:

### C1. ~~DALI (decode na GPU) vs DataLoader CPU~~ — esse confound NÃO EXISTE
- `tensorflow_opt` linha 2480: **`use_dali = False` HARDCODED**
  (comentário: "device mismatch no ARM/GH200").
- `pytorch_opt`: `--enable_dali` existe, mas o **default é desligado** e o run
  script nunca o passava.
- **Os dois já decodificam na CPU.** Nessa dimensão já estavam alinhados.
  Ligar DALI nos dois exigiria destravar o caminho ARM do TF — risco alto,
  ganho nulo para a comparabilidade. Mantido **desligado nos dois**, declarado.
  Manopla para a ablação futura: `HCPA_USE_DALI=1`.

### C2. ~~`--recompute_backbone` (gradient checkpointing) só no TF~~ — era NO-OP
- `tensorflow_opt` linha 2484: **`recompute_backbone = False`**, e o script só
  imprime `[INFO] recompute_grad desativado nesta versão otimizada`.
- A flag foi **removida** do run script (só confundia).

### C3. ~~LR 0,003 (TF) vs 5e-4 (PyTorch), 6×~~ — o LR efetivo do TF era 2e-4
- Com `freeze_epochs=1`, o TF usa `--lrate` por **1 época** (só a cabeça) e
  depois `--fine_tune_lr` (**2e-4**) nas outras 199.
- Com `freeze_epochs=0` (a receita comum), **os dois** passam a usar
  `fine_tune_lr` como LR operante. Por isso `HCPA_LRATE` é escrito nas duas
  flags. A razão real era **2,5×**, não 6×.

### C4. 🆕 Dois desalinhamentos que o plano NÃO tinha visto
- **`--fine_tune_at -200`**: o TF treinava só as ~200 camadas finais do backbone
  (`resolve_layer_index(total, -200)`), enquanto o `pytorch_opt` treina **tudo**.
  → receita comum usa `--fine_tune_at 0` (backbone inteiro).
- **`freeze_bn=True`** (default do TF): o BatchNorm ficava **congelado**; no
  PyTorch não há congelamento de BN.
  → receita comum usa `--no-freeze_bn`.

### C5. 🆕 BUG: a métrica `specificity` do TF era `SpecificityAtSensitivity(0.95)`
```python
tf.keras.metrics.SpecificityAtSensitivity(0.95, name='specificity')   # ANTES
```
O `val_spec` por época do TF era **spec@95sens**; o do PyTorch era **spec@0.5**.
Números diferentes, mesma coluna. É a explicação da "spec ruim do tensorflow_opt".
→ Criada a métrica `Specificity` (threshold 0.5) nos dois scripts TF, e a de 95%
renomeada para `specificity_at_sens95`. O fallback silencioso
`spec_at_sens95 = spec` foi **removido** (mascarava a regressão).

### C6. 🆕 O efeito observador era pior do que o §0.4 dizia
Além das ~4 chamadas NVML por batch, os 5 `train.py` (hybrids/retfound/vit)
faziam **`torch.cuda.synchronize(device)` a CADA batch** só para ler memória —
serializando o pipeline. Removido.

---

# EXECUÇÃO — o que foi implementado

| # | Mudança | Onde |
|---|---|---|
| 1 | Amostragem uniforme (thread de fundo) nos 9 scripts; zero chamadas NVML no laço de treino | `gpu_energy.py`, 4× `dr_hcpa`, 5× `train.py` |
| 2 | `HCPA_GPU_SAMPLE_MS=0` desliga a amostragem (braço de controle da Fase A) | `gpu_energy.py` |
| 3 | `EnergyScope.stop()` idempotente | `gpu_energy.py` |
| 4 | EMA **desligado por padrão** (`--ema_decay 0`), preservado como ablação | `tensorflow_opt` |
| 5 | Receita comum única para os dois `_opt` | `tools/common_recipe.sh` |
| 6 | `epsilon` do AdamW alinhado (1e-8 → 1e-7) | `tensorflow_opt` |
| 7 | Métrica `Specificity` (0.5) + `specificity_at_sens95` | `tensorflow_opt`, `tensorflow_base` |
| 8 | Schema CSV unificado: **55 → 35 colunas**, idêntico nas 8 abordagens | 4× `dr_hcpa`, `hybrid_shared/results.py` |
| 9 | `val_spec_at_sens95` **por época** (custo clínico por época, exigido pela Q2) | idem |
| 10 | Profiling de kernel CUPTI: `busy_time` exato (união) + `cpu_wait_time` | `gpu_kernel_profile.py` |
| 11 | Piano roll | `new_results/plot_piano_roll.py` |
| 12 | `fig_speed_vs_energy` (Q1) e `fig_energy_wasted` (Q2) | `new_results/make_plots.py` |
| 13 | Figuras: **19 → 12**; 3 bloqueadas até validar contra CUPTI | idem, `plot_gpu_idle.py` |
| 14 | Varredura de LR + relatório | `tools/lr_sweep.sh`, `tools/lr_sweep_report.py` |
| 15 | `tools/sync_shared_libs.sh` (as 9 cópias de `gpu_energy.py` estavam DEFASADAS) | `tools/` |
| 16 | `HCPA_*` agora atravessam o container | `tools/g5k_run_common.sh` |

## Métricas removidas (auditoria)
`train_auc/precision/f1/sens/spec` (lotes com augmentation → enganosos) ·
`val_gpu_util_pct` · `val_busy_time_s` · `*_avg_power_w` (val/test) ·
`*_avg_batch_time_ms` (val/test) · `*_inference_latency_*` · `*_gpu_mem_avg_mb`

## Figuras removidas / bloqueadas
- **Removidas (redundantes):** `fig_throughput` (≈1/tempo), `fig_power` (=E/t),
  `fig_time_per_epoch` (está no breakdown), `fig_energy_overview` (duplicata),
  `fig_best_auc_epoch` (virou painel C do `fig_auc_per_epoch`).
  Voltam com `HCPA_EXTRA_FIGS=1`.
- **Bloqueadas até validar:** `fig_gpu_util`, `fig_busy_time`,
  `fig_time_busy_idle`, `fig_gpu_util_timeline` — dependem do `util%` do NVML,
  medido com amostragem assimétrica. `HCPA_UNVALIDATED_FIGS=1` para inspecionar.

## Resultados já visíveis nos dados atuais

**Q1 — o mais rápido é o que gasta menos energia? NÃO.**
A potência média varia de **249 W** (tensorflow_opt) a **516 W** (retfound_green).
Contraexemplo direto: **tensorflow_base é 438 s MAIS LENTO que retfound_green e
ainda assim gasta 290 kJ a MENOS.**

**Q2 — desperdício após o best-epoch:**

| abordagem | best ep. | útil | desperdiçada | % |
|---|---|---|---|---|
| RETFound-Green | 11 | 99 kJ | 1426 kJ | **93 %** |
| tensorflow_opt | 29 | 50 kJ | 255 kJ | **83 %** |
| pytorch_opt | 27 | 88 kJ | 381 kJ | **81 %** |
| tensorflow_base | 73 | 463 kJ | 772 kJ | 63 % |
| pytorch_base | 115 | 540 kJ | 389 kJ | 42 % |
| hybrid_token_reduction | 128 | 765 kJ | 426 kJ | 36 % |

⚠️ Números da instrumentação ANTIGA. O `spec` do TF estava errado (C5) e o
`util%`/`busy_time` estavam enviesados (C6). **Energia e tempo NÃO são afetados**
pelo C5, e o viés do C6 sobre eles é de segunda ordem — mas a re-rodada é
necessária para publicar.

## O que falta rodar (nesta ordem)

1. **Smoke-test na GPU** — 1 época de cada framework: a telemetria em thread de
   fundo funciona, o `--ema_decay 0` não quebra o TF, o `--fine_tune_at 0` +
   `--no-freeze_bn` compilam, e `HCPA_PROFILE_EPOCHS=1` produz o JSON do CUPTI.
2. **Fase A (efeito observador)** — `HCPA_GPU_SAMPLE_MS` ∈ {0, 200} × 2
   frameworks, 3 épocas. Mede quanto a instrumentação custa agora.
3. **Fase B (validar o NVML)** — comparar `busy_time` do CUPTI com
   `elapsed × util%` na MESMA época. Quantifica o erro da estimativa.
4. **Varredura de LR** — `bash tools/lr_sweep.sh` (≈1,5 h de GPU). Fixa `HCPA_LRATE`.
5. **Re-rodada final** — 8 abordagens × 10 runs (~52 h de GPU; ~17 h em 3 nós).
6. **Piano roll** — `HCPA_PROFILE_EPOCHS=5,50,120,199` num run de cada framework.

⚠️ **Cota Grid5000**: passos 2–6 consomem GPU·h. De dia vale cota; rodar na
janela noturna / fim de semana. Sinalizar antes.

## Limitações a declarar
1. **XLA vs TorchInductor** — "JIT ligado nos dois" ≠ JIT equivalente.
2. **tf.data vs DataLoader** — ambos decodificam na CPU, mas escalonam workers
   de formas diferentes. Não eliminável sem reescrever um pipeline.
3. **cuDNN** pode escolher algoritmos diferentes para a mesma camada.
4. **`util%` do NVML é ocupação temporal**, não capacidade: 1 kernel em 1 SM
   conta como "ocupado". Por isso o CUPTI.
5. **O profiler contamina a época que mede** — épocas com CUPTI ficam fora das
   comparações de tempo/energia.
6. **Nós físicos** dão ~5 % de diferença de energia → fixar o nó por abordagem.
7. **1 GPU, 1 dataset, 1 arquitetura** — não generaliza automaticamente.

---

# RODADA 2 DE CORREÇÕES (2026-07-09, após auditoria das 8 abordagens)

### C7. 🔴 GRAVE — `pytorch_opt` e `pytorch_base` serializavam a GPU a cada batch
`if track_memory: torch.cuda.synchronize(device)` rodava em **todo batch de
treino** (`track_memory` é sempre `True` em GPU), só para ler memória que o
`EnergyScope` já lê em thread de fundo. Um `synchronize()` por batch bloqueia o
host até a fila da GPU esvaziar: destrói a sobreposição CPU↔GPU.

Efeito: o PyTorch corria algemado enquanto o TensorFlow (após a correção C6)
corria livre — **exatamente na comparação central do estudo**. Removido.

### C8. O TensorFlow nunca sincronizava a GPU na borda da medição
`_cuda_sync()` só agia se o `torch.cuda` estivesse inicializado. Em processos TF
era no-op, e confiava-se em que "o Keras sincroniza sozinho em `on_test_begin`" —
suposição nunca verificada. Agora usa `tf.test.experimental.sync_devices()`.

### C9. `time.time()` medindo duração
Os 4 scripts `dr_hcpa` usavam relógio de parede (sujeito a ajuste de NTP) para
medir intervalos. Trocado por `time.perf_counter()` (7 pontos).

### C10. A janela de medição não fechava no mesmo ponto
5 abordagens incluíam o cálculo das métricas da época na janela; 3 excluíam.
Uniformizado para **incluir** — única opção possível, já que no TF as métricas
são streaming dentro do `train_step`. Reordenados: `hybrid_simple`,
`hybrid_token_reduction`, `vit_pure` (as outras já estavam corretas).

---

# Q2 REFORMULADA: de "desperdício" (oráculo) para "early stopping" (realizável)

**A crítica:** "tudo que roda depois do best-epoch é desperdício" exige saber
qual é o pico **antes** de chegar nele. Não é executável. Como recomendação, o
número é indefensável.

**A reformulação:** simula-se um early stopping que decide a cada época com a
informação disponível até ali (paciência `p` = 20 épocas sem melhora de
`val_auc`; `HCPA_ES_PATIENCE`). Entrega-se o melhor checkpoint conhecido na
parada. O custo clínico de parar cedo é medido explicitamente:

    es_auc_penalty = (melhor AUC atingida no treino inteiro)
                   − (melhor AUC conhecida na época da parada)

`early_stop_epoch()` foi testada, inclusive a propriedade de **não-vazamento**:
truncar a curva depois da época de parada não muda a decisão.

## Resultado (GH200, p = 20) — este é o número publicável

| abordagem | energia necessária | economia | para na ep. | pico real | **custo ΔAUC** |
|---|---|---|---|---|---|
| RETFound-Green | 259 kJ | **−83 %** | 31 | 11 | 0,0000 |
| Hybrid CNN-ViT (TR) | 237 kJ | **−80 %** | 40 | 128 | 0,0031 |
| pytorch_base | 200 kJ | **−78 %** | 42 | 115 | 0,0012 |
| tensorflow_base | 266 kJ | **−78 %** | 41 | 73 | 0,0023 |
| pytorch_opt | 107 kJ | **−77 %** | 36 | 27 | 0,0001 |
| tensorflow_opt | 73 kJ | **−76 %** | 44 | 29 | 0,0001 |

Duas famílias, com falhas opostas:
- **`_opt`**: o pico chega cedo (ep. 27-29) e o early stopping treina **além**
  dele → paga em ENERGIA, ΔAUC ≈ 0,0001.
- **`base` / híbridas**: o pico chega tarde (ep. 73-128) e o early stopping para
  **antes** dele → paga em AUC (até 0,0031).

Figura: `fig_energy_early_stopping`. O número de oráculo continua no CSV
(`energy_wasted_pct`) como referência, e o tique preto o marca na figura — mas a
conclusão do paper deve citar o early stopping.

## Consequência: as 200 épocas não se justificam
Nenhuma abordagem melhora depois da época ~44 o suficiente para pagar 76-83 % da
energia. A recomendação prática do estudo passa a ser sobre o **protocolo de
treino**, não só sobre a escolha de framework.

---

# SMOKE-TEST NA A100 (chuc, Lille) — 2026-07-10

Primeira execução real do código corrigido. **Encontrou 11 bugs.** Sete deles não
apareceriam em nenhuma revisão de código parado, e **três teriam produzido
tabelas com números plausíveis e conclusões erradas**.

## Resultado

| abordagem | invariantes | CUPTI |
|---|---|---|
| `pytorch_opt` | **20/20** | 187.746 kernels · busy 6,14 s · cpu_wait 1,75 s (21 %) |
| `tensorflow_opt` | **20/20** | 211.929 kernels · busy 6,25 s · cpu_wait 2,54 s (28 %) |

Os dois fazem o mesmo trabalho de GPU (6,1 vs 6,2 s de kernel), mas o TensorFlow
**espera 45 % mais pelo host**. É exatamente o que o `util%` do NVML não sabia
dizer — e o que o piano roll mostra época adentro.

## Os 11 bugs

**Que produziriam números errados sem avisar:**

1. 🔴 **`pytorch_opt` treinava na CPU, em silêncio.** `torch.cuda.is_available()`
   era False, o torch emitia um `UserWarning`, marcava `AMP active=False` e
   seguia. Gravava um CSV completo cuja energia/tempo não tinham relação com
   GPU nenhuma. Descoberto olhando `nvidia-smi` no nó: 0 % de utilização.
   → guarda que **aborta alto** nas 9 abordagens (`HCPA_ALLOW_CPU=1` para debug).

2. 🔴 **Os nós chuc têm 4 GPUs.** O TF faz `if len(gpus) > 1: MirroredStrategy` —
   treinaria em 4 GPUs com batch global 384, enquanto o `pytorch_opt` usaria 1.
   E o NVML mede o handle da GPU 0. Na hydra (1 GPU/nó) o bug é invisível.
   → `CUDA_VISIBLE_DEVICES=0` no container (`HCPA_VISIBLE_GPUS`).

3. 🔴 **O parser de XPlane jogava os kernels no balde de cópias.** Sob XLA o TF
   funde tudo num stream, e a linha do trace se chama literalmente
   `Stream #13(MemcpyD2H,Memset,MemcpyD2D,Compute)`. Meu filtro por NOME DE LINHA
   via "Memcpy" primeiro: 250.343 kernels viravam cópias, `busy_time = 0,00 s`.
   O perverso: `copy_time = 6,71 s` ≈ o `busy = 6,14 s` que o PyTorch mediu.
   **O número estava certo, no campo errado.**
   → classificação **por evento** (cada evento traz seu nome), e só linhas de
   stream CUDA (`XLA Modules`/`Steps` são spans de op que inflariam a união).

**Que corrompiam métricas:**

4. `train_busy_time_s` do PyTorch vinha do `elapsed` interno do `EnergyScope`,
   enquanto o CSV reportava o `elapsed` do laço. Na época perfilada a razão
   `busy/elapsed` deu **1,229** — fisicamente impossível.
   → derivado do MESMO `elapsed` que vai ao CSV; `busy <= elapsed` por construção.

5. O `stop()` do profiler caía dentro da janela de energia da **validação**:
   `val_energy_j = 5.525 J` na época 3 contra ~330 J nas vizinhas.
   → callback do CUPTI movido para o **início** da lista.

6. A limpeza pós-run (`find ! -name '*.csv' ! -name '*.pdf' -delete`) apagava os
   JSON do CUPTI — exatamente o que o piano roll consome.

7. **`HCPA_EPOCHS` só valia para os dois `_opt`.** Os outros seis liam
   `TARGET_EPOCHS` ou `EPOCHS`. Pedir 5 épocas rodava **200, em silêncio**
   (flagrado com `tensorflow_base` em `Epoch 76/200`).
   → `HCPA_EPOCHS` é a manopla única das 8.

**Que impediam rodar:**

8. `_implements_train_batch_hooks` ausente em 5 callbacks Keras. O `CallbackList`
   do `tf_keras` chama esses métodos em TODOS os callbacks; a classe base do
   Keras 3 não os define. Depende da versão do container: passava na GH200,
   estourava no chuc.

9. `singularity --nv` precisa de `/usr/sbin/ldconfig.real`, um debianismo que a
   imagem `debiannvopen11` não traz. Sem ele: "Could not find any nv libraries",
   o container cai na `libcuda` de compatibilidade → **treino na CPU** (bug 1).
   E **cada `kadeploy` reimagina o nó**, então o symlink some toda vez.

10. `-t exotic` hardcoded no `oarsub` (chuc é x86: o job esperaria para sempre) e
    hostname montado como `chuc-6.lyon.grid5000.fr`.

11. Cache do singularity no disco raiz do nó (31 G) → `no space left on device`
    no meio do build. `/tmp` é NVMe com 1,6 T.

## Estado da infraestrutura

- **8 de 8 imagens `.sif` construídas** e persistidas no NFS home (`~/projects/hcpa/*/hcpa.sif`).
  Não precisam ser refeitas.
- `tools/smoke_check.py` valida 20 invariantes do CSV e do CUPTI. Testado contra
  os dados antigos: reprova as 4 regressões conhecidas.
- `chuc-2` está **Absent**; `chuc-1` e `chuc-5`, **Dead**. Só `chuc-6`/`chuc-7`
  (e `chuc-3`/`chuc-4`/`chuc-8`, ocupados de madrugada).

## Lição de operação

A janela noturna (19:00–09:00 CEST) expirou durante os builds. Os `.sif`
sobreviveram porque ficam no NFS, mas as reservas morreram e a Fase 3 falhou com
`No route to host`. **Buildar e treinar na mesma reserva é frágil**: os builds
levam ~7 min por imagem e não são paralelizáveis num nó só.

Próxima vez: reservar com walltime folgado, ou separar a reserva de build da de
treino (as imagens persistem).

## O que falta validar na GPU

- `tensorflow_base`, `hybrid_simple`, `hybrid_token_reduction`, `retfound_green`
  (os 4 que não couberam na janela). `pytorch_base` e `hybrid_token_reduction_opt`
  cobrem os dois caminhos de código restantes (`dr_hcpa` e `train.py`).
- Fase A: efeito observador (`HCPA_GPU_SAMPLE_MS` ∈ {0, 200}).
- Fase B: `busy_time` do CUPTI vs `elapsed × util%` do NVML, na mesma época.
- Varredura de LR, depois a re-rodada final das 8 × 10 runs.

---

# BUG 12 — "cópias" agrupava coisas fisicamente diferentes

Ao inspecionar por que o `tensorflow_opt` fazia 355 "cópias" por batch e o
`pytorch_opt` só 21, os streams entregaram a resposta:

    tensorflow_opt: 40.841 eventos não-kernel
        234 no stream 14 (MemcpyH2D)  -> 2 por batch, o pipeline de dados
        928 no stream 15 (MemcpyD2H)
     39.679 no stream 13 -> que é o stream de COMPUTE: são MemcpyD2D e Memset
            emitidos pelo XLA. NÃO são tráfego com o host.

    pytorch_opt: 2.420 eventos não-kernel, todos no stream dos kernels.

O classificador jogava tudo com "memcpy"/"memset" no nome num balde só, rotulado
"cópias H2D/D2H" no piano roll. **O rótulo era falso**: fazia parecer que o TF
transferia 355×/batch quando transfere ~2.

**Os dois pipelines transferem do host a cada batch** — como se espera, já que
ambos decodificam na CPU (nenhum usa DALI). A assimetria era artefato de medição.

Correção: `classify_event()` separa h2d/d2h/d2d/memset pelo NOME do evento, nos
dois backends. O resumo passa a ter:

    busy_time_s         união dos kernels
    device_work_time_s  união de kernel + d2d + memset  (ocupam os SMs)
    transfer_time_s     união de h2d + d2h              (ocupam a copy engine)
    n_h2d / n_d2h / n_d2d / n_memset

O JSON agora grava o tipo de cada evento — antes o nome era descartado, e por
isso não deu para auditar o trace já coletado sem re-rodar.
