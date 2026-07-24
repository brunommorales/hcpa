# Validação final — revisão honesta de toda a sessão (2026-07-11)

> O que foi pedido: analisar tudo que fizemos nesta sessão e validar se **agora está
> realmente tudo correto**. Este documento é a revisão honesta — separa o que está
> **PROVADO**, o que é **PARCIAL/inferido**, e o que continua **ABERTO**.

---

## 1. O arco da sessão (o que fizemos, em ordem)

1. **Diagnóstico** das 8 abordagens (`DIAGNOSTICO_DESEMPENHO.md`) — e nele cometi um
   **erro** (D1/D2: "TF cacheia, PyTorch não"), corrigido depois.
2. **Nivelamento do pipeline** (F1–F4) para comparação justa: cache + shuffle uniformes.
3. **Auditoria fresca** (F6, `AUDITORIA_PIPELINE_NIVELADO.md`) — sem bugs novos no código.
4. **Validação em hardware** no GH200 (G1–G5): smoke de 3 caminhos + A/B do cache.
5. **Correção de 2 achados** (H1–H4): bug `epoch_time_sec` + `.sif` do tf_base.
6. **Validação das 8** (esta etapa): rodar as 5 abordagens que faltavam.

---

## 2. ✅ O que está PROVADO (validado com evidência)

| item | evidência |
|---|---|
| **Cache PyTorch (F1) é byte-idêntico ao decode** | teste offline: max\|dx\|=0; augment-off idêntico |
| **Cache PyTorch é fork-safe** | ndarray único contíguo, view C-contiguous; rodou com 8 workers |
| **Cache acelera (F1)** | A/B no GH200: **3255 vs 2535 img/s** em regime (**+28%**) |
| **Cache não contamina métricas** | construído no `__init__`, ANTES de `start_time`/`EnergyScope`/`t_start` |
| **F2 (tf_opt cache+shuffle+XLA)** | rodou no GH200: ep0 74s → regime 9s, sem OOM |
| **F3 (tf_base cache+reorder)** | rodou no GH200: ep0 45s → regime 11s, sem OOM |
| **torch.compile compila** | 99→3255 img/s (ep0→ep1, 33×) no hardware |
| **Sem OOM** | PyTorch 9,6 GB / TF 5,5–13 GB de **480 GB** |
| **Dataset v2 correto** | 9350/1650/4816, **0 vazamento por paciente**, byte-idêntico, 160 idx |
| **best-ckpt na val, reporta na test** | confirmado nas 8 (pytorch L1832, retfound L661, TF via P1) |
| **Bug `epoch_time_sec` corrigido** | re-validado: ep0=72,7s ep1=6,5s (era timestamp 1,78e9) |
| **Coluna canônica `train_throughput_img_s` sempre esteve correta** | callback separado (L1210-1220) usa perf_counter nos 2 lados |
| **`.sif` tf_base** | symlink NFS→tf_opt (Apptainer.def diferem só em comentário) |

---

## 3. ✅ FECHADO — as 8 rodaram diretamente no GH200

As 5 que faltavam rodaram 2 épocas cada (job 2046469, hydra-2), **todas rc=0, sem OOM**:

| abordagem | throughput regime | AUC (2 ép) |
|---|---|---|
| pytorch_base | 1250 img/s | val 0.949 |
| hybrid_simple | 3078 img/s | 0.833 |
| hybrid_token_reduction | 3247 img/s | 0.812 |
| hybrid_token_reduction_opt | ~3000 img/s | 0.807 |
| **retfound_green (392px)** | 2298 img/s | **0.957** (sem OOM apesar do cache maior) |

Somadas às 3 anteriores (pytorch_opt, tf_opt, tf_base) → **as 8 abordagens rodam
limpo com o pipeline nivelado**. O retfound a 392px (cache ~7GB) não estourou memória.

---

## 4. ❌ O que continua ABERTO (não foi feito nesta sessão)

Sendo honesto — isto NÃO está resolvido:

1. **A re-rodada final 8×10×200 NÃO foi feita.** Esta sessão validou a **máquina**
   (pipeline + fixes), não produziu os **resultados científicos** finais. Todos os
   números anteriores (GH200 antigos + prod200) estão **superados** e precisam ser
   refeitos com o pipeline nivelado. Isso é uma janela grande (noite inteira).
2. **P10 — domínio de energia no GH200: CARACTERIZADO nesta sessão (2026-07-11).**
   Probe pynvml na GH200 com só a CPU Grace no talo (72 cores, GPU ociosa): o contador
   `nvmlDeviceGetTotalEnergyConsumption` saltou **+195W** → **inclui a CPU Grace**.
   Veredito: na GH200 `train_energy_j` = energia do **MÓDULO (Grace+Hopper)**, NÃO
   GPU-only. O A100 era GPU-only (idle 58W) → **joules GH200 vs A100 têm domínios
   DIFERENTES, não comparáveis diretamente**. O `nvmlDeviceGetPowerUsage` (instant →
   `train_avg_power_w` amostrado) NÃO subiu com a CPU (GPU-orientado). DECISÃO PENDENTE
   p/ o paper: (a) rotular energia GH200 como "módulo" e não comparar com A100 GPU-only,
   ou (b) integrar a potência amostrada p/ isolar o Hopper. Probe: `p10_probe.py`.
3. **`epoch_time_sec` não auditado no PyTorch/híbridos.** Corrigi no tf_opt e conferi
   o tf_base (limpo). O throughput do PyTorch parece são (3255 img/s), mas não auditei
   a coluna `epoch_time_sec` deles com o mesmo rigor.
4. **RNG dos workers (numpy) — pré-existente, não corrigido.** `_apply_augmentations`
   usa `np.random` global sem `worker_init_fn` → workers herdam o mesmo estado. Igual
   nas 6 PyTorch (não quebra comparação entre elas), mas reduz diversidade de augment.
5. **`DIAGNOSTICO_DESEMPENHO.md`** ficou com o D1/D2 errado + banner de correção; não
   foi reescrito (o `AUDITORIA_PIPELINE_NIVELADO.md` tem a versão correta).
6. **Telemetria de memória dos híbridos — FALSO ALARME (só cosmético).** O
   `Avg GPU memory: 0.0` era apenas o print do terminal (`compute_avg_gpu_memory_mb`);
   o CSV de métricas tem `train_gpu_mem_peak_mb` correto por época (7385/7788 MB), além
   de train_energy_j, train_elapsed_s, train_busy_time_s, train_throughput_img_s todos
   preenchidos. **Dados OK**; corrigir o print é opcional.
7. **`epoch_time_sec`/timing do PyTorch e híbridos — CONFERIDO, limpo.** Usam
   `perf_counter` nos dois lados (hybrid L210/L460, pytorch L1513/1627/1675). O bug de
   relógio era exclusivo do tf_opt (já corrigido).

---

## 5. Veredito honesto

**"Agora está tudo correto?"**

- **O pipeline nivelado e os 2 fixes: SIM, corretos e validados em hardware.** As 8
  abordagens agora são comparáveis no pipeline de dados (cache + shuffle uniformes),
  as métricas de tempo estão coerentes, não há OOM, e o cache comprovadamente acelera.
- **A comparação científica final: AINDA NÃO EXISTE.** Validamos que a máquina está
  correta; falta rodar o 8×10×200 para produzir os números do paper.
- **Duas incertezas reais permanecem**: o domínio de energia no GH200 (P10) — que pode
  mudar todos os joules — e o RNG dos workers (menor, uniforme).

Ou seja: **o que consertamos nesta sessão está certo e provado**; o estudo final
ainda depende de (a) rodar o 8×10×200 e (b) resolver o P10 antes de reportar energia.
