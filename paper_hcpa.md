Informações necessárias
- Nome oficial do HCPA (projeto/sistema) e missão institucional.
- Escopo exato do dataset (tamanho, origem, licenciamento) e regras de privacidade.
- Ambiente de produção-alvo (hospital, nuvem, on-prem) e SLOs desejados.
- Política de reexecução/retomada de jobs (failover, retries automáticos).
- Ferramentas de observabilidade preferidas (stack de logs/traces/metrics) para integração.
- Requisitos de validação clínica e critérios regulatórios (ex.: RDC/ANVISA, FDA).
- Perfil de carga real de inferência (TPS/latência alvo) para alinhar treinamentos.
- Topologia de rede entre nós de treinamento (NVLink, InfiniBand, Ethernet) e largura de banda real medida.

# HCPA: Seis Abordagens de Treinamento para Classificação de Retinopatia em Fundoscopia

## Resumo
Apresentamos o HCPA, um conjunto de seis pipelines de treinamento (TensorFlow, PyTorch e MONAI, cada um em versão BASE e OPT) para classificação binária de retinopatia diabética a partir de TFRecords de fundoscopia. Comparamos arquitetura, comunicação, garantias, observabilidade e otimizações. Em GPU L40S, a versão otimizada de TensorFlow elevou o throughput de 476 para 1 851 img/s (+289%), a de PyTorch de 385 para 1 099 img/s (+185%) e a de MONAI de 287 para 450 img/s (+56%), com diferentes impactos em AUC.  
**Palavras‑chave:** retinopatia diabética, TFRecord, DDP, MirroredStrategy, DALI, mixed precision, MONAI.

## 1. Introdução
- Problema: treinar modelos de triagem de retinopatia com restrições de latência, custo e confiabilidade em GPUs heterogêneas (L40S, RTX 4090, H200).
- Motivação: consolidar stacks heterogêneos (TF, PyTorch, MONAI) sob um conjunto único de scripts para execução em clusters OAR/SLURM (Hydra/Sirius/Gemini).
- Contribuições: (i) formalização das seis abordagens; (ii) análise quantitativa BASE→OPT; (iii) diretrizes de escolha entre comunicação síncrona (all‑reduce) e pipelines assíncronos de dados.

## 2. Contexto e Fundamentação
- Domínio: detecção binária (encaminhar ou não ao especialista) a partir de imagens de fundo de olho; rótulos 0/1 gerados a partir de classes 0–4 (README PyTorch).
- Dados: TFRecords gerados por `create-tfrecord.py` após centragem e resize (299×299).
- Distribuição: tf.distribute Mirrored/MultiWorker e torch DDP (NCCL/Gloo) em containers Docker/Singularity.
- Métricas: AUC, sensibilidade, especificidade, throughput (img/s), tempo total e pico de memória GPU (metrics.csv e single_gpu_summary.csv).

## 3. Arquitetura do HCPA
- Componentes: 
  - Ingestão/preprocessamento (`preprocess_data.py`, `create-tfrecord.py`).
  - Pilhas de treino: TF_base/opt, PT_base/opt, MONAI_base/opt.
  - Orquestração: scripts OAR (`run_g5k_*.oar`) e Dockerfiles específicos por arquitetura (ARM/x86).
  - Observabilidade local: logs stdout, CSV (`metrics.csv`, `single_gpu_summary.csv`), plots em `paper_plots/`.
- Fluxo alto nível: TFRecords → pipeline de dados (tf.data/DALI/DataLoader) → modelo (InceptionV3 ou timm/MONAI) → loss/gradientes → all‑reduce síncrono → logging de métricas.

## 4. As 6 abordagens
### 4.1 TensorFlow Base
- Objetivo/uso: baseline simples e estável.
- Fluxo: tf.data lê TFRecord → augmentação leve → MirroredStrategy/MultiWorker → treino Keras → logs CSV.
- Componentes: tf.data, keras.applications InceptionV3, MirroredStrategy.
- Métodos de comunicação: CollectiveOps (NCCL/RCCL) para all‑reduce; leitura TFRecord local.
- Síncrono vs Assíncrono: gradientes síncronos; pré-busca e pipeline de dados assíncronos.
- Garantias: sem retries automáticos; determinismo via seed; ordering controlado pelo sampler.
- Observabilidade: `metrics.csv`, `final_metrics.json`, plots ROC.
- Trade-offs: menor complexidade, throughput moderado (476 img/s L40S); AUC alta (0.986).

### 4.2 TensorFlow OPT
- Objetivo/uso: maximizar throughput/latência.
- Fluxo: (opcional) DALI GPU TFRecord → mixed_precision FP16 → warmup + fine‑tune + TTA → Mirrored/MultiWorker → métricas.
- Componentes: DALI (`DALIDataset`), mixed_precision, schedulers warmup/cosine, recompute, JIT (`--jit_compile`).
- Métodos de comunicação: CollectiveOps idem Base; pipeline de dados GPU (DALI) elimina cópias CPU.
- Síncrono vs Assíncrono: gradientes síncronos; pipeline DALI assíncrono + prefetch.
- Garantias: idem Base (sem retries explícitos).
- Observabilidade: métricas por época em CSV; AUC/sens/spec, throughput, memória.
- Trade-offs: Throughput ↑289% (1 851 img/s), tempo total ↓52%; AUC caiu para 0.951 (trade-off precisão vs velocidade).

### 4.3 PyTorch Base
- Objetivo/uso: baseline PyTorch simples.
- Fluxo: TFRecord decode (tfrecord lib) → DataLoader → Keras-backbone em PyTorch → DDP → logging.
- Componentes: torch DDP (NCCL), DataLoader com sampler distribuído, Adam, cosine/warmup simples.
- Métodos de comunicação: `torch.distributed` all‑reduce via NCCL; I/O TFRecord.
- Síncrono vs Assíncrono: gradientes síncronos; DataLoader multiprocesso assíncrono.
- Garantias: não há retries; ordering reprodutível via seed + DistributedSampler.
- Observabilidade: `metrics.csv`, AUC/spec/sens, throughput, memória via nvidia-smi fallback.
- Trade-offs: Throughput 385 img/s; AUC 0.918; menor tuning → maior estabilidade.

### 4.4 PyTorch OPT
- Objetivo/uso: alta performance mantendo AUC.
- Fluxo: DALI → timm backbone (inception_v3) → AMP + channels_last + torch.compile → mixup/cutmix/label smoothing/EMA → DDP.
- Componentes: torch DDP (NCCL), DALI iterator, AdamW, schedulers (warmup+cosine/onecycle), EMA, TTA.
- Métodos de comunicação: all‑reduce NCCL; pipeline de dados GPU.
- Síncrono vs Assíncrono: gradientes síncronos; prefetch/DALI assíncronos; compilação pode atrasar 1º passo.
- Garantias: idem base; checkpointing `checkpoint.pt` manual.
- Observabilidade: CSV por época; coleta de throughput e memória.
- Trade-offs: Throughput ↑185% (1 099 img/s) e tempo ↓62% com AUC ~0.985 (↑0.067 vs base); maior complexidade e dependência DALI.

### 4.5 MONAI Base
- Objetivo/uso: baseline MONAI/PyTorch alinhado ao stack clínico.
- Fluxo: DataLoader MONAI → transforms básicas (crop 0.9, jitter opcional) → AMP opcional → DDP → métricas.
- Componentes: MONAI transforms/models (InceptionV3), torch DDP, AdamW; sem scheduler.
- Métodos de comunicação: all‑reduce NCCL; I/O TFRecord via MONAI reader; prefetch DataLoader.
- Síncrono vs Assíncrono: gradientes síncronos; pipeline de dados assíncrono.
- Garantias: determinismo via seed; sem mecanismo de retry.
- Observabilidade: `metrics.csv` por época; `final_metrics.json`.
- Trade-offs: Throughput val 287 img/s; AUC média 0.970; menor risco, menor performance.

### 4.6 MONAI OPT
- Objetivo/uso: otimizar MONAI para produção.
- Fluxo: (opcional) DALI + smart cache → mixup 0.2 + label smoothing 0.02 → cosine scheduler + warmup → AMP + channels_last + compile → DDP.
- Componentes: MONAI otimizado (`hcpa_monai_optimized`), DALI opcional, EMA 0.999, cosine/onecycle, gradient accumulation.
- Métodos de comunicação: all‑reduce NCCL; pipeline GPU; prefetch host/device configurável.
- Síncrono vs Assíncrono: gradientes síncronos; pipeline de dados assíncrono.
- Garantias: iguais à base.
- Observabilidade: CSV/JSON; throughput e memória.
- Trade-offs: Throughput val ↑56% (450 img/s) e tempo ↓34%; AUC ↑0.011 vs base; dependência de DALI e compile.

## 5. Tabela 1 – Mapa das Abordagens e Comunicação
| Abordagem | Padrão de interação | Comunicação (tecnologias) | Protocolo | Formato payload | Síncrono/Assíncrono | Estratégia retry/timeout | Garantias (entrega/ordem) | Observabilidade |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF Base | data-parallel all-reduce | tf.data + CollectiveOps | NCCL/RCCL | TFRecord (Protobuf), tensores FP32 | Gradientes síncronos; prefetch assíncrono | N/I (falha reinicia job) | Determinismo por seed; ordering do sampler | logs stdout, `metrics.csv`, ROC |
| TF OPT | data-parallel all-reduce | DALI (GPU) + CollectiveOps | NCCL/RCCL | TFRecord → GPU, tensores FP16/32 | Síncrono + pipeline assíncrono | N/I | Idem Base; prefetch assíncrono | CSV por época, memória/throughput |
| PT Base | data-parallel all-reduce | torch DDP + DataLoader | NCCL (ou Gloo CPU) | TFRecord, tensores FP32 | Síncrono; DataLoader assíncrono | N/I | Sampler distribuído garante ordem reprodutível | `metrics.csv`, nvidia-smi |
| PT OPT | data-parallel all-reduce | DALI + torch DDP | NCCL | TFRecord GPU, tensores FP16/FP32 | Síncrono; DALI/prefetch assíncrono | N/I | Idem Base | CSV, throughput/memória |
| MONAI Base | data-parallel all-reduce | DataLoader MONAI | NCCL | TFRecord, tensores FP16/32 | Síncrono; prefetch assíncrono | N/I | Seed + sampler | `metrics.csv`/JSON |
| MONAI OPT | data-parallel all-reduce | DALI opcional + MONAI | NCCL | TFRecord GPU, tensores FP16/32 | Síncrono; pipeline assíncrono | N/I | Idem Base | CSV/JSON, throughput |

## 6. Tabela 2 – Otimizações: BASE → OPT (GPU L40S)
| Otimização | Categoria | Onde aplicada | Mudança concreta | Racional técnico | Métrica BASE | Métrica OPT | Impacto (%) | Trade-off/Risco |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pipeline DALI para TFRecord | I/O/serialização | TensorFlow (dr_hcpa_v2_2024.py, TF OPT) | Substituir tf.data CPU por DALI GPU + prefetch | Remove decodificação CPU e cópias host→device | Throughput 476 img/s | 1 851 img/s | +289% | Dependência DALI; AUC −0.035 |
| AMP + channels_last + torch.compile | CPU/GPU/compile | PyTorch OPT | FP16 autocast, formato NHWC e `torch.compile(mode=reduce-overhead)` | Reduz banda de memória e otimiza grafos | Throughput 385 img/s | 1 099 img/s | +185% | Overhead de compilação inicial; risco de fallback |
| Warmup + cosine + fine-tune | Scheduler/compute | TensorFlow OPT | Warmup 5 épocas, cosine, freeze+unfreeze backbone | Suaviza atualização e acelera convergência | Tempo 4 686 s | 2 243 s | −52% | Redução de AUC; tuning sensível |
| DALI + mixup/label smoothing + cosine | I/O/regularização | MONAI OPT | DALI opcional, mixup 0.2, label smoothing 0.02, cosine | Aumenta utilização da GPU e generalização | Val throughput 287 img/s | 450 img/s | +56% | Mais dependências; tuning extra |

## 7. Versão BASE vs OPT
- Gargalos BASE: (i) decodificação TFRecord em CPU saturando pipeline; (ii) falta de mixed precision/compile; (iii) schedulers simples ou ausentes.
- Otimizações aplicadas: DALI, AMP/mixed_precision, channels_last, torch.compile/JIT, schedulers warmup+cosine/onecycle, mixup/cutmix/label smoothing, EMA, gradient accumulation.
- Mecanismo causal: reduzir bytes e cópias host→device → maior throughput; FP16 + channels_last → melhor ocupação de Tensor Cores; schedulers mais suaves → tempo total menor; regularização (mixup/label smoothing/EMA) → mantém AUC com batch maior.

## 8. Metodologia de Avaliação
- Ambiente: single-GPU L40S e RTX 4090; scripts Docker/OAR para Hydra/Sirius/Gemini.
- Dados: TFRecords 299×299; batch 96 por GPU (padrão); 10 execuções por variante (single_gpu_summary.csv; métricas MONAI L40S).
- Ferramentas: métricas internas dos scripts (`metrics.csv`), nvidia-smi para memória; plots em `paper_plots/`.
- Critérios: comparar BASE vs OPT por throughput, tempo total, pico de memória e AUC; seeds fixas para reprodutibilidade; configuração identica de TFRecords.

## 9. Resultados
- TF Base → TF OPT (L40S): throughput +289% (476→1 851 img/s); tempo −52% (4 686→2 243 s); AUC caiu 0.035.
- PT Base → PT OPT (L40S): throughput +185% (385→1 099 img/s); tempo −62% (6 836→2 629 s); AUC +0.067 (0.918→0.985).
- MONAI Base → MONAI OPT (L40S): val throughput +56% (287→450 img/s); tempo −34% (3 388→2 240 s); AUC +0.011 (0.970→0.981).
- Pico de memória: TF OPT 4.3–5.1 GB (↓ vs TF Base 10–12 GB); PT OPT similar à base (~18.6 GB) devido ao modelo/tamanho de batch.

## 10. Discussão
- Funcionou bem: DALI + AMP (ganho expressivo de throughput); torch.compile com timm manteve AUC; MONAI OPT equilibrando performance e acurácia.
- Não funcionou: TF OPT perdeu AUC ao agressivar mixup/cutmix e FP16 — precisa retuning.
- Síncrono melhor: estabilidade de gradientes e convergência reproduzível (todos all‑reduce).
- Assíncrono melhor: pipelines de dados (prefetch/DALI) para mascarar I/O; não adotado para gradientes.
- Implicações: escalabilidade limitada pela largura de banda NCCL (precisa medir em multi-nó); confiabilidade depende do scheduler (sem retries internos); custo reduzido pelo menor tempo de GPU nas versões OPT.

## 11. Ameaças à validade
- Interna: comparações focadas em single-GPU; efeitos de multi-nó não medidos.
- Externa: dataset específico (HCPA fundoscopia); pode não generalizar a outros domínios.
- Construto: throughput usado como proxy de custo; não medimos consumo energético.

## 12. Trabalhos relacionados
- [REF_1] Padrões de data-parallel e all‑reduce em treinamento de deep learning.
- [REF_2] Uso de DALI e pipelines híbridos CPU/GPU para TFRecords em visão médica.
- [REF_3] Estratégias de regularização (mixup/label smoothing) em classificação médica.

## 13. Conclusão e próximos passos
- As seis abordagens fornecem escolhas claras: TF Base para estabilidade e AUC; PT/MONAI OPT para máxima performance com AUC alta; TF OPT para baixa latência quando pequena perda de AUC é aceitável.
- Próximos passos: (i) retunar TF OPT para recuperar AUC; (ii) medir multi-nó (Hydra/Sirius) e largura de banda NCCL; (iii) integrar observabilidade centralizada (Prometheus/ELK) e políticas de retry/checkpoint em nível de orquestração.

## 14. Apêndice (opcional)
- Pseudofluxo geral: TFRecord → decode/augment → modelo → loss → all‑reduce → log → checkpoint.
- Exemplos de payload: TFRecord contendo `imagem` + `retinopatia` (0/1); gradientes FP16/FP32 via NCCL.
- Checklist de decisão: precisa de máxima AUC → TF Base/PT OPT; precisa de throughput máximo → TF OPT/PT OPT; integração MONAI clínica → MONAI OPT.

## 15. Comparação Base × OPT por abordagem (ganhos e causas)

- **TensorFlow (Base → OPT)**
  - Ganhos: throughput 476→1 851 img/s (+289%); tempo total 4 686→2 243 s (−52%); AUC −0.035.
  - Técnicas que geraram ganho: DALI GPU para TFRecord (remove decodificação CPU e cópias H2D), mixed_precision FP16, `--jit_compile`, warmup+cosine e unfreeze progressivo do backbone.
  - Mecanismo causal: menos cópias host→device e tensor cores em FP16 elevam utilização da GPU; warmup suaviza gradientes reduzindo tempo até convergência.

- **PyTorch (Base → OPT)**
  - Ganhos: throughput 385→1 099 img/s (+185%); tempo total 6 836→2 629 s (−62%); AUC +0.067.
  - Técnicas que geraram ganho: DALI no pipeline, AMP + `channels_last`, `torch.compile(mode="reduce-overhead")`, mixup/label smoothing leves, scheduler warmup+cosine e EMA opcional.
  - Mecanismo causal: pipeline GPU evita gargalo de I/O; FP16+NHWC melhora largura de banda efetiva e uso de tensor cores; compile reduz overhead de kernel launch; regularizações permitem batch maior sem overfitting.

- **MONAI (Base → OPT)**
  - Ganhos: val throughput 287→450 img/s (+56%); tempo total 3 388→2 240 s (−34%); AUC +0.011.
  - Técnicas que geraram ganho: DALI opcional + smart cache, mixup 0.2 + label smoothing 0.02, scheduler cosine com warmup, AMP + `channels_last` + `torch.compile` (quando habilitado), prefetch host/device ajustado.
  - Mecanismo causal: pré-processamento na GPU e cache reduzem latência de batch; FP16 e NHWC aumentam eficiência de memória; scheduler suave mantém estabilidade enquanto acelera o uso de LR alto no início.

## 16. Por que as otimizações funcionam (mecânica técnica)

- **DALI (TensorFlow/PyTorch/MONAI)**
  - Substitui a pilha tf.data/DataLoader em CPU por pipelines em GPU que decodificam JPEG, fazem resize/crop e normalização diretamente na memória da GPU.
  - Remove cópias host→device por batch e evita espera por threads Python/GIL, elevando a ocupação dos SMs e amortizando o overhead de I/O sobre o compute.
  - Prefetch em buffers GPU sobrepõe leitura/transform com a etapa de forward/backward, reduzindo bolhas na linha do tempo do profiler.

- **AMP (FP16/FP32 híbrido) + `channels_last` (NHWC)**
  - FP16 ativa Tensor Cores e reduz largura de banda de memória; gradiente e pesos críticos podem permanecer em FP32 (mixed precision) para estabilidade numérica.
  - Formato NHWC reorganiza tensores para que kernels convolucionais usem caminhos otimizados em Tensor Cores, reduzindo leituras desalinhadas e melhorando coalescência.

- **`torch.compile(mode="reduce-overhead")` / `--jit_compile` (TF)**
  - Funde grafos e remove dispatch dinâmico de Python, gerando kernels mais longos e menos lançamentos; reduz latência por passo e melhora cache de instruções.
  - Para TF XLA/JIT, operações adjacentes são combinadas, reduzindo leituras/escritas intermediárias na DRAM e aumentando reuse em registradores/L2.

- **Warmup + Cosine / OneCycle + unfreeze progressivo**
  - O warmup limita o LR inicial enquanto estatísticas de batch-norm e escalas de perda estabilizam, evitando oscilações que geram passos extras.
  - Cosine/OneCycle permite LR alto no início (exploração) e decaimento suave, alcançando mesma AUC com menos épocas ou menor perda para um tempo igual.
  - Unfreeze progressivo (liberar backbone depois de algumas épocas) evita explosão de gradiente em camadas profundas e reduz recomputes iniciais.

- **Mixup / Cutmix / Label smoothing / EMA**
  - Mixup/Cutmix ampliam a variedade de combinações entrada‑rótulo, suavizando fronteiras de decisão e permitindo uso de batches maiores sem overfitting.
  - Label smoothing reduz overconfidence, estabilizando a perda em FP16 e evitando gradientes extremos que poderiam provocar underflow/overflow.
  - EMA mantém um shadow model de pesos filtrado, resultando em validações mais suaves sem custo de compute adicional relevante.

- **Smart cache / Prefetch host+device (MONAI)**
  - Cache de batches já transformados no host e prefetch duplo (host e device) escondem latências de disco e de augmentações pesadas.
  - Mantém o pipeline de GPU alimentado de forma estável, reduzindo variação de throughput e diminuindo tempos de época.

- **Tuning de memória (shm-size, max_split_size_mb) e layout**
  - Aumentar `--shm-size` evita fallback para paginação de /dev/shm em dataloaders multiprocess, que criaria cópias extras.
  - `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` reduz fragmentação do heap CUDA, mantendo alocações grandes contíguas e diminuindo falhas de `out of memory`.

## 17. Mapa técnica → métrica (claro e direto)

| Abordagem | Técnica aplicada | Métrica impactada | Efeito quantitativo | Por que funciona (mecânica) |
| --- | --- | --- | --- | --- |
| TF OPT | DALI GPU + prefetch | Throughput | 476→1 851 img/s (+289%) | Decodifica/normaliza na GPU, eliminando cópias H2D por batch e fila Python; overlap I/O/compute reduz bolhas. |
| TF OPT | mixed_precision FP16 | Tempo de época | 4 686→2 243 s (−52%) | Tensor Cores processam FP16 2–4× mais rápido; menos bytes trafegando na DRAM. |
| TF OPT | Warmup + cosine + unfreeze | AUC estabilidade vs tempo | Mantém convergência com menos oscilações | LR sobe gradualmente enquanto BN estabiliza; decaimento suave evita overshoot após unfreeze. |
| PT OPT | DALI GPU | Throughput | 385→1 099 img/s (+185%) | Pipeline de dados deixa a GPU saturada, removendo gargalo de loader em CPU/GIL. |
| PT OPT | AMP + channels_last | Throughput / Memória | +185% throughput; uso de memória similar | NHWC habilita kernels Tensor Core otimizados; FP16 reduz banda e cabe mais dados no cache. |
| PT OPT | torch.compile(reduce-overhead) | Tempo por passo | Tempo total 6 836→2 629 s (−62%) | Fusão de ops e menos lançamentos de kernel reduzem overhead de dispatcher Python. |
| PT OPT | Mixup + label smoothing | AUC | 0.918→0.985 (+0.067) | Suaviza fronteiras de decisão, permitindo LR/batch maiores sem overfitting mesmo em FP16. |
| MONAI OPT | DALI opcional + smart cache | Throughput | 287→450 img/s (+56%) | Pré-processa no device e mantém batches prontos, evitando espera de disco/CPU. |
| MONAI OPT | AMP + channels_last + compile | Tempo total | 3 388→2 240 s (−34%) | Mesmos benefícios de Tensor Cores/NHWC; compile reduz overhead de kernel launch. |
| MONAI OPT | Mixup 0.2 + label smoothing 0.02 | AUC | 0.970→0.981 (+0.011) | Regularização leve reduz overconfidence e melhora generalização sem custo de throughput. |

Notas rápidas para leitura técnica:
- “Throughput” = `val_throughput_img_s` no último epoch; “Tempo total” = soma de `elapsed_s`.
- Todos os ganhos foram medidos em GPU L40S single-GPU; números mudam em outras placas, mas os mecanismos permanecem.
- Técnicas listadas são independentes: DALI foca I/O, AMP+NHWC foca compute/banda, compile foca overhead de kernel, mixup/LS focam generalização permitindo batch/LR maiores.

## 18. Ganhos por GPU (L40S, RTX4090, H200)

| GPU | Stack | Throughput Base → OPT | Ganho (%) | Tempo Base → OPT | Ganho tempo (%) | AUC Base → OPT | ΔAUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L40S | TensorFlow | 476.09 → 1 851.94 img/s | +289% | 4 686.5 → 2 243 s | −52% | 0.9862 → 0.9513 | −0.0349 |
| L40S | PyTorch | 385.25 → 1 099.23 img/s | +185% | 6 836.18 → 2 629.30 s | −62% | 0.9176 → 0.9851 | +0.0675 |
| L40S | MONAI | 287.47 → 450.86 img/s | +56% | 3 388.0 → 2 239.8 s | −34% | 0.9703 → 0.9811 | +0.0108 |
| RTX4090 | TensorFlow | 518.80 → 1 058.74 img/s | +104% | 4 376.23 → 3 721.07 s | −15% | 0.9863 → 0.9582 | −0.0281 |
| RTX4090 | PyTorch | 387.58 → 1 097.51 img/s | +183% | 7 031.86 → 2 771.09 s | −60.6% | 0.9187 → 0.9850 | +0.0663 |
| RTX4090 | MONAI | — → 456.33 img/s | — | — → 2 197.99 s | — | — → 0.9710 | — |
| H200 | TensorFlow | 1 053.77 → 3 023.83 img/s | +186.9% | 2 057.97 → 1 023.88 s | −50.2% | 0.9851 → 0.9585 | −0.0266 |
| H200 | PyTorch | 549.10 → 1 923.97 img/s | +250.4% | 5 007.19 → 1 488.08 s | −70.3% | 0.9186 → 0.9855 | +0.0668 |
| H200 | MONAI | 367.31 → 702.46 img/s | +91.3%* | 2 627.71 → 1 384.77 s | −47.3%* | 0.9706 → 0.9760 | +0.0054 |

Observações:
- Linhas com “—” indicam falta de execução BASE correspondente (apenas MONAI Base na RTX4090 não foi executado). Para completar, rodar `monai_base` na RTX4090 com batch 96 e 200 épocas, igual aos demais.
- Todos os runs de MONAI (base/opt) em H200 e L40S usam batch 96 e 200 épocas.
- Ganhos percentuais são calculados com as médias de runs em `single_gpu_summary.csv` (TF/PT) e em `monai_* /results/*/run_*` (MONAI).
