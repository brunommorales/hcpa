# Análise GH200 — energia, tempo e o salto do tensorflow_opt

Base: `nvidia-gh200-480gb_g5k_hydra`, 10 runs por abordagem. Todas as métricas por época
vêm de `InceptionV3-*.csv` (TF) / `inception_v3-*.csv` (PyTorch); agregados em
`gh200_all_runs.csv`. Instrumentação em [gpu_energy.py](../gpu_energy.py).

## 1. A energia é GPU-only e o contador captura o kernel corretamente

- `nvmlDeviceGetTotalEnergyConsumption` é um **acumulador de hardware** (mJ desde o load do
  driver). Lido em `on_epoch_begin`/`on_epoch_end`; o delta = ∫P·dt = energia exata que a
  **GPU** gastou na janela. Um integrador **não perde** kernel dentro da janela nem inclui
  kernel fora — o medo de "amostrar entre kernels" vale só para a **potência pontual** (200 ms),
  nunca para o contador de energia. Na GH200, é a energia do módulo Hopper, separada da Grace.
- **Ressalva (sincronização):** não há `synchronize` antes das leituras. Como CUDA é assíncrono,
  a fronteira pode escorregar **no máximo ~1 step/época = 1/115 ≈ 0,87 %**, e **~0 no total**
  (a cauda da época N cai na N+1 e se cancela). Na prática é menor, pois o Keras já sincroniza ao
  consolidar as métricas da época. → **energia confiável, não precisa re-rodar para confiar.**
- **Ressalva de rótulo:** a janela inclui a validação → `train_energy_j` = energia (treino+val);
  `val_energy_j` fica `None`.

## 2. O salto foi nos 10 runs, no epoch 120, e é o EMA (não é térmico nem erro)

Figura: [graficos/fig_energy_per_epoch_smoothed.png](nvidia-gh200-480gb_g5k_hydra/graficos/fig_energy_per_epoch_smoothed.png)

| regime | batch | energia/época | potência |
|---|---|---|---|
| estável (ep 2–119) | 24,8 ms | 1788 J | 303 W |
| pós-salto (ep 120–199) | 57,0 ms (×2,3) | 2585 J (**+45 %**) | 218 W (**−28 %**) |

- Onset em **exatamente o epoch 120 nos 10 runs**.
- **Causa:** `ema_start_epoch = int(0.6*EPOCHS) = 120` (dr_hcpa_v2_2024.py). A `EMACallback`
  atualizava, **a cada batch**, todas as ~centenas de variáveis treináveis num **for-loop eager
  em Python** (`shadow.assign(0.999*shadow + 0.001*weight)`), **fora do grafo/XLA** → centenas de
  ops minúsculas despachadas no host → GPU ociosa → step mais lento, potência menor, energia maior.
- **Não é térmico/DVFS:** na mesma máquina, `tensorflow_base` (104→104 ms) e `pytorch_opt`
  (37,6→37,7 ms) seguem **planos**. Se fosse do nó, os três cairiam.
- **Impacto:** os últimos 80 epochs do tensorflow_opt estão inflados (~+45 % energia, ~+130 %
  tempo), contaminando `total_energy_kj` e `train_compute_s` dessa abordagem.
- **Nota:** no teste final, se o best.ckpt não for carregado, aplicam-se os pesos EMA — candidato
  a investigar junto do efeito AMP/XLA na especificidade baixa (0,788).

## 3. Há influência de CPU no tempo — e dá pra mitigar já

Figura: [graficos/fig_cpu_gpu_time.png](nvidia-gh200-480gb_g5k_hydra/graficos/fig_cpu_gpu_time.png)

- `train_avg_batch_time_ms = train_elapsed/train_steps` é **wall-clock** (`time.time()`), sem
  sincronização — inclui dataloader, dispatch de Python e host.
- tensorflow_opt: **train_compute_s = 902 s vs total_time_s = 1486 s → 39 % do tempo é não-GPU**
  (val, exact-eval a cada 5, IO, compilação XLA, TTA final).
- Mesmo dentro do compute a GPU está **faminta**: 268–303 W, contra **560 W do vit_pure** e 471 W
  do pytorch_base na *mesma* GPU. Potência baixa = GPU ociosa esperando o host por step.
- **Mitigação sem re-rodar:** usar `train_compute_s`/`steady_epoch_time_s` (tira val/eval/IO/
  compilação) no lugar de `total_time_s` para eficiência energética.
- **Só com re-run:** separar a ociosidade *dentro* do step (tempo GPU-busy real) exige
  `nvmlDeviceGetUtilizationRates` ou CUDA events — não coletado hoje.

## 4. Correção do EMA (aplicada)

`EMACallback` agora compila o update num **único `tf.function`** (grafo), capturando shadow/weights
por closure. Matemática idêntica; troca centenas de ops eager/batch por 1 launch → o salto do
epoch 120 deve desaparecer. Rebuild recompila em cada transição de estágio. Valida por
`py_compile` (OK); validação funcional exige GPU (hydra).
