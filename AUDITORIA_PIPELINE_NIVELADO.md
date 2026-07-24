# Auditoria — nivelamento de pipeline + revisão fresca das 8 abordagens

> Rodada 2026-07-11. Objetivo do usuário: **comparação justa** (nivelar o pipeline
> de dados) + **conferir que nada mais passou**. Estado: correções implementadas
> (F1–F4); smoke em hardware **adiado** para a próxima janela de GPU (a reserva
> chuc-2 expirava às 08:24 CEST e não valia rushar às 2h). Cache PyTorch validado
> offline byte-a-byte.

---

## A. O que mudou (nivelamento do pipeline)

A "justiça" foi separada em dois eixos:

- **Pipeline de dados = infraestrutura compartilhada** → nivelado nas 8.
- **`compile`/XLA/AMP/`channels_last`/TF32 = eixo base-vs-opt do estudo** → NÃO
  mexido (nivelar isso apagaria a comparação que o paper mede).

| # | mudança | arquivos | efeito |
|---|---|---|---|
| F1 | **Cache in-memory** da imagem decodificada+redimensionada (uint8), fork-safe (1 ndarray contíguo herdado pelos workers via CoW) | `pytorch_opt/dr_hcpa_v2_2024.py` (`RetinaTFRecordDataset`, `build_torch_loader`) | as **6 abordagens PyTorch** param de redecodificar JPEG toda época |
| F2 | Liga o cache já existente (sentinel `--cache_dir memory`) + **shuffle 2048→12000** | `tensorflow_opt/dr_hcpa_v2_2024.py`, `run_g5k_hydra.sh` | `tensorflow_opt` passa a cachear (estava OFF por `--cache_dir none`) e a embaralhar completo |
| F3 | Adiciona `.cache()` + reordena `decode→cache→shuffle(12000)→augment` | `tensorflow_base/dr_hcpa_v2_2024.py` | `tensorflow_base` (que **não tinha cache**) passa a cachear e a embaralhar completo |

Resultado: **todas as 8 cacheiam a imagem decodificada e embaralham o dataset
inteiro** (buffer ≥ train=9350). A diferença que resta entre frameworks passa a
ser modelo/compute, não acidente de I/O.

---

## B. Correção de um erro do MEU diagnóstico anterior

O `DIAGNOSTICO_DESEMPENHO.md` (D1/D2) afirmava *"o TF cacheia, o PyTorch não"*.
**Errado.** A verdade que esta auditoria encontrou:

- `tensorflow_opt` **tinha** `.cache()` no código, mas o run script passava
  `--cache_dir none` → `CACHE_BASE_DIR=None` → `.cache()` **nunca era chamado**.
- `tensorflow_base` **não tinha `.cache()` nenhum**.
- Ou seja, **nas rodadas reais ninguém cacheava** — todos redecodificavam.

O "21% de cpu_wait" que a CUPTI mediu no PyTorch **não era falta de cache** (o TF
também redecodificava); era a **eficiência do pipeline** (`tf.data` C++/AUTOTUNE
esconde o decode melhor que os workers Python do PyTorch). Com o cache ligado nas
8, o decode acontece 1× (época 1) e some — o que isola melhor o modelo.

---

## C. Revisão fresca das 8 — cobertura verificada

| abordagem | loader | DALI | cache agora | shuffle | best-ckpt |
|---|---|---|---|---|---|
| pytorch_opt | `build_torch_loader` | off (`set_defaults False`) | ✅ F1 | full (sampler) | val→test (L1832) |
| pytorch_base | `build_torch_loader` | off (default) | ✅ F1 | full | val→test |
| tensorflow_opt | `tf.data` | — | ✅ F2 (`memory`) | full 12000 | val→test (P1) |
| tensorflow_base | `tf.data` | — | ✅ F3 | full 12000 | val→test (P1) |
| hybrid_simple | bridge→`build_torch_loader` | off (`--disable_dali`) | ✅ F1 | full | val→test |
| hybrid_token_reduction | idem | off | ✅ F1 | full | val→test |
| hybrid_token_reduction_opt | idem | off | ✅ F1 | full | val→test |
| retfound_green | idem | off (`enable_dali=False`) | ✅ F1 | full | val→test (L661) |

**Verificações-chave:**
1. **Nenhuma usa DALI** nas rodadas (todas `enable_dali=0`) → todas passam pelo
   `build_torch_loader` → todas pegam o cache F1. *(Se algum dia ligar DALI, o
   cache NÃO se aplica — o `build_dali_loader` é outro caminho.)*
2. **Cache não contamina as métricas**: no PyTorch o cache é construído no
   `__init__` (antes de `start_time` L1675, antes do `EnergyScope.start()` L1695 e
   do `t_start` por época L1513) → **fora** de `train_elapsed_s`, `total_train_time_s`
   e do escopo de energia. No TF o preenchimento do cache cai na **época 1**
   (já excluída do regime).
3. **Best-checkpoint** em todas: seleciona pela **val**, reporta na **test**.
4. **Val e test cacheados** nas 8 (PyTorch: os 3 loaders; TF: `load_dataset`/
   `build_dataset` aplica `.cache()` a train/valid/test).

---

## D. Assimetrias que RESTAM — bug ou de propósito?

| item | valores | veredito |
|---|---|---|
| **LR** | 5e-4 (CNN) / 1e-4 (híbridos) / 5e-5 (retfound) | ⚙️ **de propósito** — cada arquitetura precisa de LR próprio (modelo de fundação exige LR pequeno). Documentar que cada uma foi ajustada; não é bug. |
| **AMP** | base OFF (`--disable_amp`), resto ON | ⚙️ eixo base-vs-opt. Correto. |
| **cosine** | base OFF, resto ON | ⚙️ eixo base-vs-opt. Correto (P4 já alinhou o hybrid_opt). |
| **compile/XLA** | só nos `_opt` | ⚙️ é a definição de "opt". Correto. |
| **channels_last** | só `hybrid_token_reduction_opt` (flag própria) | ⚙️ parte da identidade "opt" dele; os outros não usam. Aceitável, mas **documentar**. |
| **img_size** | retfound 392, resto 299 | ⚙️ arquitetura (ViT de fundação). Correto. |
| **freeze** | 0 (CNN/retfound), 3 (híbridos = `freeze_backbone_epochs`) | ⚙️ de propósito; híbridos aquecem a cabeça antes. |
| **batch** | 96 nas 8 | ✅ uniforme. |

**Nenhum bug novo.** As assimetrias que restam são todas ou (a) o eixo base-vs-opt
que o estudo mede, ou (b) hiperparâmetros por-arquitetura. Recomendação para o
paper: uma tabela declarando explicitamente o que é igual (pipeline, batch, split,
métricas, seleção) e o que é por-arquitetura (LR, img_size, freeze).

---

## E. Riscos residuais — verificar no smoke de hardware (próxima janela)

Lógica verificada, execução em GPU ainda não:
1. **TF `--cache_dir memory`** dispara `.cache()` de fato (lógica confere:
   `CACHE_BASE_DIR='memory'` é truthy → `.cache()`; bloco de worker-subdir
   desviado). Confirmar em runtime.
2. **RAM**: TF cacheia float32 (~17GB train+val+test); PyTorch uint8 (~4–7GB).
   Node chuc-2 tem 512GB → folga. Confirmar sem OOM.
3. **Época 1 mais lenta** (preenchimento do cache) nas 8 → o cálculo de throughput
   de regime **deve continuar excluindo a época 1**.
4. **Startup do PyTorch** +~40s (decode serial de 15.816 imgs no `__init__`) — não
   entra em métrica, mas aparece no wall-clock.

---

## F. Coerência das coletas de métricas — veredito

**Mantida.** As mudanças de pipeline não tocaram a instrumentação:
- energia (contador NVML, janela por fase), tempo (`perf_counter`+sync), AUC exata
  por época, `test_auc` honesto (val seleciona / test reporta) — **inalterados**.
- O cache foi posicionado **fora** de todos os timers e do escopo de energia (item
  C.2), então os números continuam medindo treino, não I/O de setup.

⚠️ **Consequência**: todos os resultados anteriores (GH200 + prod200 na A100) foram
medidos SEM o pipeline nivelado → estão **superados**. A comparação final exige
re-rodar as 8 com o pipeline nivelado (8×10×200), o que só cabe na janela de
segunda (cota diurna) ou nas próximas noites/fim de semana livres.
