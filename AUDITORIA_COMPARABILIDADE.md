# Auditoria de comparabilidade e validade — 8 abordagens HCPA

> Escopo: as 8 abordagens do estudo (`vit_pure` fora). Auditoria do **desenho
> experimental**, não da instrumentação (essa foi corrigida: ver
> `PLANO_ESTUDO_ENERGIA.md`, 12 bugs).
>
> **Veredito: a coleta está correta. O desenho experimental não está.**
> Um problema é fatal para um artigo clínico. Todos são corrigíveis.

---

## Resumo executivo

| # | Problema | Severidade | Afeta |
|---|---|---|---|
| **P1** | **Não existe conjunto de validação: validação == teste** | 🔴 **FATAL** | Toda métrica clínica; Q2 inteira |
| **P2** | Só o TF grava `patient_id`? Não — **ninguém grava**. Split por paciente não verificável | 🔴 **FATAL se não resolvido** | Validade clínica |
| **P3** | TF alterna dois estimadores de AUC na mesma curva | 🟠 grave | best-epoch, early stopping, Q2 |
| **P4** | `hybrid_token_reduction_opt` roda **sem cosine scheduler** | 🟠 grave | curva de convergência dele |
| **P5** | `drop_last` divergente: PyTorch 114 batches, TF 115 | 🟡 médio | energia/época (−0,5 %) |
| **P6** | Determinismo só no `retfound_green` | 🟡 médio | tempo, energia e variância dele |
| **P7** | Receitas heterogêneas (LR 5e-5 a 5e-4, freeze, wd, augmentation) | 🟡 declarar | enquadramento do paper |
| **P8** | `retfound_green` em 392 px (as outras em 299) | 🟡 declarar | não é o mesmo workload |
| **P9** | Sem plano estatístico (n=10 cobre só a seed) | 🟡 médio | força das conclusões |
| **P10** | Domínio de energia do NVML no GH200 nunca verificado | 🟡 aberto | magnitude absoluta dos joules |

**O que já é sólido e NÃO é afetado por P1/P2:** a comparação relativa de
**energia e tempo** entre abordagens, o `fig_speed_vs_energy` (Q1), o piano roll
e todo o profiling de kernel. Esses medem custo computacional, não desfecho clínico.

---

## P1 — Não existe conjunto de validação. Validação = Teste. 🔴

### Evidência

O dataset tem **dois** splits, só:

```
data/all-tfrec/  ->  train*.tfrec  (110 arquivos, 11.000 imagens)
                     test*.tfrec   ( 49 arquivos,  4.816 imagens, 30,5 %)
```

E as três famílias fazem, todas, a mesma coisa:

```python
# tensorflow_opt/dr_hcpa_v2_2024.py:2704
VALID_FILENAMES = tf.io.gfile.glob(TFREC_DIR + '/test*.tfrec')   # <-- teste como validação

# pytorch_opt: --val_split default "val"; não existe.
#   default_validation_fallbacks() -> ["valid", "test"]  -> cai em test*
#   test_files = resolve_split_files(dir, "test")        -> os MESMOS arquivos

# hybrid_simple / hybrid_token_reduction* / retfound_green:
#   eval_split_name="test"   (mesmo conjunto para selecionar e para reportar)
```

### Por que isso é fatal

1. **O best-checkpoint é escolhido maximizando a AUC no conjunto de teste.**
   O `test_auc` reportado (≈0,98) não é uma estimativa honesta de generalização:
   é o **máximo de 200 tentativas** avaliadas no próprio conjunto de teste.
   Viés otimista, e a magnitude cresce com o número de épocas.

2. **A Q2 inteira depende dessa curva.** O `best_auc_epoch`, o early stopping
   simulado, o `es_auc_penalty` e a "energia desperdiçada" são todos calculados
   sobre `val_auc` — que é a AUC no teste. Um early stopping que olha o teste
   não é um early stopping; é seleção de modelo no teste.

3. `test_spec_at_sens95` — a métrica clínica central do paper — sofre do mesmo
   viés, e o threshold de 95 % de sensibilidade também é escolhido nesse conjunto.

### Nota importante

Isto **não invalida** as medições de energia, tempo, `busy_time`, `cpu_wait` nem
o piano roll. Todas as abordagens sofrem o mesmo viés clínico, então a **ordem
relativa** de energia continua válida. O que cai é qualquer afirmação clínica
absoluta e a Q2 na forma atual.

---

## P2 — Nenhum `patient_id` no dado. Vazamento por paciente não verificável. 🔴

```python
# pytorch_base/create-tfrecord.py:40
            # 'patient_id': _int64_feature(patient_id),    <-- COMENTADO
            'image_name': _string_feature(name),
```

O TFRecord carrega `imagem`, `retinopatia` e `image_name`. **`patient_id` está
comentado.** O split train/test vem de dois CSVs prontos
(`labels_train.csv`, `labels_test.csv`), e não há como saber, a partir dos
TFRecords, se um mesmo paciente teve o olho direito no treino e o esquerdo no
teste.

Em retinopatia diabética isso é o vazamento clássico: os dois olhos de um
paciente são altamente correlacionados. Se ocorreu, a AUC está inflada por cima
do viés do P1.

**Isto precisa ser respondido antes de qualquer artigo**, e a resposta está nos
CSVs originais do HCPA, não no código.

---

## P3 — O TensorFlow mede `val_auc` com dois estimadores diferentes, alternando 🟠

`--exact_eval_interval 5` faz o `ExactEvalMetricsLogger` rodar
`roc_auc_score` (exato) e **sobrescrever** `logs["val_AUC"]` e `logs["val_auc"]`
apenas quando `(epoch+1) % 5 == 0` ou na última época.

Nas outras 4 de cada 5 épocas, o valor que vai ao CSV é a métrica Keras
`AUC(num_thresholds=200)` — uma **soma de Riemann binada**.

Enquanto isso:
- `pytorch_opt` / `pytorch_base`: `roc_auc_score` (exato) **toda** época.
- hybrids / retfound: `compute_auc` = rank-sum de Mann-Whitney (exato) **toda** época.

**Só o TF tem a curva serrilhada por estimador.** E `SaveBestLastCallback`
monitora `val_AUC`, comparando épocas medidas por métodos diferentes.

### Magnitude (simulado com n=3200, prevalência 30 %)

```
AUC exato (rank-sum)      : 0.935886
AUC binado Keras (200 th) : 0.935789      diferença = 0.000097

Em 400 pares de épocas quase-empatadas, o estimador binado
inverteu o vencedor 70 vezes  ->  18 %
```

No platô, épocas vizinhas diferem por 0,001–0,003 de AUC. Um viés sistemático de
1e-4 entre estimadores já é suficiente para trocar qual época "ganha" — e o
best-epoch alimenta a Q2 inteira.

---

## P4 — `hybrid_token_reduction_opt` roda sem cosine scheduler 🟠

```
hybrid_simple              ENABLE_COSINE=1
hybrid_token_reduction     ENABLE_COSINE=1
hybrid_token_reduction_opt ENABLE_COSINE=0    <-- unico
retfound_green             (cosine)
tensorflow_*, pytorch_*    (cosine)
```

Sete abordagens usam warmup + cosine annealing; uma não. O scheduler determina a
forma inteira da curva de convergência — e portanto o best-epoch, a época de
early stopping e a energia "desperdiçada". Muito provavelmente não intencional.

Observação de apoio: no `fig_energy_early_stopping`, `hybrid_token_reduction_opt`
foi o que atingiu o pico mais tarde (ep. 169 com p=10) e teve o maior ΔAUC do
early stopping (0,0150). Consistente com LR constante.

---

## P5 — `drop_last` divergente 🟡

```python
pytorch_opt: DataLoader(..., drop_last=shuffle)   # True no treino -> 114 batches
tensorflow_opt: dataset.batch(GLOBAL_BATCH_SIZE)  # sem drop_remainder -> 115 batches
```

11.000 / 96 = 114,58. O PyTorch descarta as 56 imagens do último batch parcial;
o TF as processa.

**Confirmado independentemente pelo CUPTI:** contamos exatamente **114**
transferências H2D grandes no `pytorch_opt` e **115** no `tensorflow_opt`.

Efeito: o TF vê 0,5 % mais amostras por época. Pequeno, mas entra direto em
energia/época, throughput e tempo — as métricas do estudo.

---

## P6 — Determinismo só no `retfound_green` 🟡

```python
retfound_green/train.py:78-80
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
```

As outras 7 não fazem isso. Consequências para o `retfound_green`:
- kernels determinísticos são tipicamente **10–20 % mais lentos** → tempo e
  energia dele estão inflados;
- a **variância entre runs** dele é artificialmente menor (só a seed varia),
  então a barra de erro não é comparável com as demais.

O `retfound_green` é justamente o que aparece com a maior potência média (516 W)
e o maior desperdício (93 %). Parte disso pode ser artefato.

---

## P7 — Receitas heterogêneas (declarar, não corrigir) 🟡

| abordagem | LR | freeze | weight decay | img | mixup/cutmix |
|---|---|---|---|---|---|
| tensorflow_base | 5e-4 | — | 1e-5 | 299 | — |
| tensorflow_opt | 5e-4 | 0 | 1e-4 / 1e-5 | 299 | 0 / 0 |
| pytorch_base | 5e-4 | 0 | 1e-4 / 1e-5 | 299 | — |
| pytorch_opt | 5e-4 | 0 | 1e-4 / 1e-5 | 299 | 0 / 0 |
| hybrid_simple | **1e-4** | **3** | — | 299 | — |
| hybrid_token_reduction | **1e-4** | **3** | — | 299 | — |
| hybrid_token_reduction_opt | **1e-4** | **3** | — | 299 | **sim** |
| retfound_green | **5e-5** | ? | **5e-4** | **392** | — |

Isto é legítimo como "melhor esforço por abordagem". Mas então **o paper não pode
afirmar que compara frameworks**. Só o par `tensorflow_opt` ↔ `pytorch_opt`
isola o runtime (receita comum via `tools/common_recipe.sh`).

O enquadramento correto é: *comparamos oito **abordagens** (arquitetura + receita
+ runtime), cada uma no seu melhor esforço; e, separadamente, isolamos o efeito
do runtime num par controlado.*

---

## P8 — `retfound_green` em 392 px 🟡

1,7× mais pixels que 299². Não é o mesmo workload computacional. Inerente à
abordagem (RETFound-Green tem patch 14 e foi pré-treinado nessa resolução), mas
precisa aparecer em toda tabela de energia.

---

## P9 — Sem plano estatístico 🟡

`n = 10` runs cobre variância de **seed**, não de nó (~5 %, já medido) nem de
inicialização de dados. Nenhum teste de significância está previsto. Para
afirmar "A gasta menos energia que B" é preciso um intervalo de confiança e um
teste pareado.

---

## P10 — Domínio de energia do NVML no GH200 🟡

`tools/g5k_energy_domain_check.py` existe e **nunca rodou**. Não sabemos se
`nvmlDeviceGetTotalEnergyConsumption` no GH200 mede só a Hopper ou o módulo
Grace+Hopper. Não muda nenhuma comparação relativa, mas muda a magnitude
absoluta de **todos** os joules do paper.

---

# PLANO DE CORREÇÃO

## Fase 0 — Bloqueadores clínicos (antes de qualquer re-rodada)

### 0.1 Criar um conjunto de validação de verdade

Origem do problema: só existem `labels_train.csv` e `labels_test.csv`.

1. **Verificar `patient_id` nos CSVs originais do HCPA.** Se existir, o split tem
   de ser **por paciente** (`GroupShuffleSplit`), não por imagem.
2. Verificar se o split train/test **atual** já respeita paciente. Se não
   respeitar, o teste está contaminado e precisa ser refeito.
3. Dividir o `train` (11.000 imagens) em **train (80 %) / val (20 %)**,
   estratificado por rótulo **e agrupado por paciente**, com seed fixa.
   Resultado: ~8.800 train / ~2.200 val / 4.816 test.
4. Regravar os TFRecords com prefixos `train*`, `val*`, `test*` e **incluir
   `patient_id`** como feature (descomentar `create-tfrecord.py:40`).

### 0.2 Fazer as 8 abordagens usarem `val*` para selecionar e `test*` para reportar

| abordagem | mudança |
|---|---|
| `tensorflow_opt`, `tensorflow_base` | `VALID_FILENAMES = glob('val*.tfrec')`; adicionar `TEST_FILENAMES = glob('test*.tfrec')` e avaliar o teste **uma vez**, no fim, com o best-ckpt |
| `pytorch_opt`, `pytorch_base` | já suportam `--val_split val`; basta o split existir. **Remover o fallback `-> test`** e falhar alto se `val*` não existir |
| 3 hybrids + `retfound_green` | trocar `eval_split_name="test"` por `val` na seleção; teste só no fim |

**Regra a codificar e testar:** `val_files ∩ test_files == ∅`. Um `assert` no
início de cada script, e um check no `tools/smoke_check.py`.

### 0.3 Recalcular a Q2 sobre a validação

Com `val` disjunto do teste, `best_auc_epoch`, o early stopping simulado e o
`es_auc_penalty` passam a ser honestos. O `test_auc` vira uma medida única,
sem seleção.

---

## Fase 1 — Consistência de medição (barato, alto impacto)

### 1.1 Um único estimador de AUC nas 8 (P3)

- **`--exact_eval_interval 1`** nos dois TF (custa uma passada de `predict` por
  época — pequena; e o `val_energy_j` já a captura).
- Alternativa mais barata: manter `interval=5` para as métricas clínicas caras,
  mas **calcular a AUC exata toda época** (é só um `roc_auc_score` sobre as probs
  que a validação já produziu).
- Padronizar o estimador: `roc_auc_score` (sklearn) ou o rank-sum do
  `hybrid_shared/metrics.py`. Os dois são exatos e equivalentes; escolher um e
  usar nas 8.
- `SaveBestLastCallback` passa a monitorar a AUC exata.

### 1.2 `drop_last` igual nas 8 (P5)

Decidir e aplicar. Recomendo **`drop_last=True` em todas** (batches de tamanho
fixo, `steps_per_epoch` constante, menos ruído em throughput):
- TF: `dataset.batch(GLOBAL_BATCH_SIZE, drop_remainder=True)` no treino.
- PyTorch: já é `drop_last=shuffle`.
- Hybrids: verificar.
- **Nunca** no val/test (todas as amostras devem ser avaliadas).

Validação: o CUPTI deve passar a contar **114** H2D grandes nas duas famílias.

### 1.3 Determinismo uniforme (P6)

Escolher um dos dois e aplicar às 8:
- **(a) Determinístico** — `cudnn.deterministic=True`,
  `use_deterministic_algorithms(True)`. Reprodutível, mas 10–20 % mais lento e
  **muda a energia medida**. Se escolher isto, todas as 8, e declarar.
- **(b) Não determinístico** (recomendado para um estudo de energia) —
  `cudnn.benchmark=True` em todas, seed só para inicialização de pesos e shuffle.
  Mede o que um usuário real obteria. **Remover o `deterministic` do
  `retfound_green`.**

Recomendo **(b)**, porque o objeto do estudo é o custo real, e o determinismo
distorce exatamente a variável medida. E declarar que a variância entre runs
inclui não-determinismo de kernel.

### 1.4 Ligar o cosine no `hybrid_token_reduction_opt` (P4)

`ENABLE_COSINE=0` → `1`. Investigar se foi intencional (git log). Se foi, virar
ablação explícita, não um asterisco escondido.

---

## Fase 2 — Enquadramento e estatística

### 2.1 Reescrever a pergunta do paper (P7, P8)

Duas comparações, declaradas separadamente:

- **(A) Comparação de abordagens** (as 8): arquitetura + receita + runtime, cada
  uma no seu melhor esforço. É o que interessa clinicamente: *"qual pipeline
  entrega spec@95 %sens por joule?"*
- **(B) Comparação de runtime** (`tensorflow_opt` ↔ `pytorch_opt`): receita
  idêntica, só o framework muda. Isola XLA vs TorchInductor, tf.data vs
  DataLoader.

Tabela de receitas (P7) e resolução (P8) **em todo** artefato de energia.

### 2.2 Plano estatístico (P9)

- n = 10 runs por abordagem, **mesmo nó físico** por abordagem (efeito de nó ~5 %).
- Reportar média ± IC 95 % (bootstrap), não desvio-padrão.
- Comparações pareadas por seed quando possível.
- Para "A gasta menos que B": teste de Wilcoxon pareado ou bootstrap da diferença.
- Declarar o efeito de nó como componente de variância não controlada.

### 2.3 Verificar o domínio de energia (P10)

Rodar `tools/g5k_energy_domain_check.py` no GH200. Se o contador cobrir
Grace+Hopper, **todos** os joules do paper mudam de significado (passam a incluir
a CPU). Declarar o resultado explicitamente.

---

## Fase 3 — Re-rodada final

Só depois das Fases 0 e 1:

1. Smoke-test das **4 abordagens que nunca rodaram** pós-reforma
   (`tensorflow_base`, `hybrid_simple`, `hybrid_token_reduction`,
   `hybrid_token_reduction_opt`) + `retfound_green`.
2. **Fase A** — efeito observador (`HCPA_GPU_SAMPLE_MS ∈ {0, 200}`).
3. **Fase B** — `busy_time` do CUPTI vs `elapsed × util%` do NVML, mesma época.
   Se o erro for grande, aposentar o `util%` das figuras.
4. Varredura de LR para o par controlado (`tools/lr_sweep.sh`).
5. Re-rodada 8 × 10 runs (~52 h de GPU).

---

## O que muda em cada abordagem (checklist)

| | val≠test | AUC exata/época | drop_last | determinismo | cosine |
|---|---|---|---|---|---|
| `tensorflow_base` | 🔧 trocar glob | 🔧 `exact_eval_interval=1` | 🔧 `drop_remainder=True` | 🔧 uniformizar | ok |
| `tensorflow_opt` | 🔧 trocar glob | 🔧 `exact_eval_interval=1` | 🔧 `drop_remainder=True` | 🔧 uniformizar | ok |
| `pytorch_base` | 🔧 remover fallback→test | ok | ok | 🔧 uniformizar | ok |
| `pytorch_opt` | 🔧 remover fallback→test | ok | ok | 🔧 uniformizar | ok |
| `hybrid_simple` | 🔧 `eval_split_name` | ok (rank-sum) | 🔍 verificar | 🔧 uniformizar | ok |
| `hybrid_token_reduction` | 🔧 `eval_split_name` | ok (rank-sum) | 🔍 verificar | 🔧 uniformizar | ok |
| `hybrid_token_reduction_opt` | 🔧 `eval_split_name` | ok (rank-sum) | 🔍 verificar | 🔧 uniformizar | 🔴 **ligar** |
| `retfound_green` | 🔧 `eval_split_name` | ok (rank-sum) | 🔍 verificar | 🔴 **remover** | ok |

Mais, para todas: `assert val_files ∩ test_files == ∅` e um check novo no
`tools/smoke_check.py`.

---

## Resposta direta

**"Podemos criar um artigo bem validado sobre o custo clínico-energético com
esses dados?"**

- **Com os dados de hoje: não.** P1 (validação == teste) invalida toda afirmação
  clínica, e P2 (sem `patient_id`) deixa em aberto um vazamento possivelmente
  maior. Um revisor rejeita no primeiro parágrafo da seção de métodos.
- **Com a Fase 0 + Fase 1 feitas: sim.** O eixo energético já é sólido — a
  instrumentação foi auditada, os 12 bugs corrigidos, e o CUPTI dá `busy_time` e
  `cpu_wait` exatos. O que falta é o eixo clínico ser honesto.
- **O eixo puramente computacional (Q1, piano roll, `fig_speed_vs_energy`)
  sobrevive intacto** e já dá um artigo de HPC/eficiência por si só, sem depender
  do P1.

---

# REVISÃO DA AUDITORIA (após inspecionar o dado)

## Escopo real desta auditoria — o que NÃO foi verificado

Fui dirigido por hipóteses, não li as ~12.000 linhas das 8 abordagens.
**Verificado:** instrumentação (12 bugs, 3 abordagens com smoke-test em GPU),
splits, estimador de AUC, `drop_last`, scheduler, determinismo, resolução,
receitas, e o dado cru.
**NÃO verificado:** correção da arquitetura dos modelos, da função de perda, das
augmentações, do TTA, da lógica de checkpoint em todas as 8, do `token_reduction`
em si, e do pipeline de pré-processamento das imagens.

---

## P2 REBAIXADO: não há evidência de vazamento por paciente 🟢

Os TFRecords **guardam `image_name`** (o parser do treino simplesmente não o lê).
Extraí os 15.816 nomes e testei:

```
imagens idênticas em train E test          : 0
pacientes (prefixo numérico 18 díg.)       : 459 train | 180 test | ∩ = 0
estudos DICOM (prefixo de 9 campos do UID) : 294 train |  84 test | ∩ = 0
img*.jpg (16 % do dataset, 2.579 imgs)     : sequenciais, não verificável
```

**O split train/test original respeita paciente.** P2 deixa de ser fatal.
Fica em aberto apenas a coorte `img*` (renomeada sequencialmente, perdeu o vínculo).

### E a Fase 0 é viável sem os CSVs originais

O Example carrega `imagem`, `image_name` e `retinopatia`:

```
train: 11.000 exemplos | {0: 8.066, 1: 2.934} | prevalência 26,7 %
test :  4.816 exemplos | {0: 3.559, 1: 1.257} | prevalência 26,1 %
```

Rótulo **binário**, e o split está bem estratificado. Dá para reconstruir tudo a
partir dos próprios TFRecords e regravar `train*/val*/test*` agrupando por
paciente. **Não precisamos dos `labels_*.csv`.**

⚠️ Mas: **98 % dos pacientes estão espalhados por vários `.tfrec`**. Não dá para
criar o `val` movendo arquivos inteiros — é preciso regravar.

### O que foi desabilitado no `create-tfrecord.py`

```python
'imagem':      _bytes_feature(img),
# 'patient_id': _int64_feature(patient_id),      <-- comentado
#'side':       _int64_feature(side),  # 0,1 left,right   <-- comentado
'image_name':  _string_feature(name),
'retinopatia': _int64_feature(label)
```

Ao regravar, **descomentar `patient_id` e `side`**.

---

## P11 (NOVO) 🔴 — Não está provado que o `torch.compile` compila

```python
# pytorch_opt/dr_hcpa_v2_2024.py:1279
torch._dynamo.config.suppress_errors = True   # não derruba o treino
model = torch.compile(model, mode="reduce-overhead")
```

O próprio comentário do código admite:

> *"Falhas nesse momento (ex.: `libcuda.so cannot found!`) são suprimidas pelo
> dynamo, que cai em EAGER silenciosamente. Para verificar se a compilação
> realmente valeu, confira a AUSÊNCIA de `BackendCompilerFailed` / `libcuda`."*

Ausência de erro **não é prova de sucesso**. No smoke-test da A100:

| | evidência |
|---|---|
| `tensorflow_opt` | **112 menções a XLA** no log → compilou |
| `pytorch_opt` | **0 menções a Inductor/Triton** · 1.633 kernels/batch (CUPTI) |

`mode="reduce-overhead"` usa CUDA graphs. Com graphs ativos, o número de launches
por batch cairia drasticamente. 1.633 kernels/batch sugere **eager ou compilação
parcial**.

**Se `pytorch_opt` roda em eager, a comparação não é "XLA vs TorchInductor" — é
"XLA vs eager".** É a afirmação central do par controlado. Precisa ser provado
com `TORCH_LOGS=graph_breaks,recompiles` e um A/B contra `PT_DISABLE_COMPILE=1`.

---

## P6 CORRIGIDO — o determinismo do `retfound_green` fomos NÓS

```
commit afe06d7 (original, 9 abordagens) : 0 menções a cudnn.deterministic
commit 1445ee1 (nossa instrumentação)  : 1 menção     <-- introduzido aqui
```

`torch.backends.cudnn.deterministic = True` e
`torch.use_deterministic_algorithms(True)` **não estavam** no `retfound_green`
original. Foram adicionados no *nosso* commit de coleta de energia — num estudo
cujo objeto de medida é justamente tempo e energia.

**Remover restaura o comportamento original**, não descaracteriza a abordagem.

---

## P8 CORRIGIDO — os 392 px não são escolha nossa

O backbone é `vit_small_patch14_reg4_dinov2`: **patch 14**.

```
224 / 14 = 16.00   OK    256 tokens
294 / 14 = 21.00   OK    441 tokens   <-- o mais próximo de 299
299 / 14 = 21.36   IMPOSSÍVEL
392 / 14 = 28.00   OK    784 tokens   <-- o que está no commit ORIGINAL
```

`retfound_green/README.md:102`: *"O `img_size` deve ser divisível por 14."*
E `IMG_SIZE=392` está no commit `afe06d7`, o original.

**Não rodamos o RETFound em 299 px.** Ele nunca poderia rodar em 299.

Opções:
- **(a)** manter 392 (config dos autores) e declarar 1,72× mais pixels — recomendado;
- **(b)** rodar 294 px como **ablação** (0,97× os pixels de 299) para separar o
  efeito de resolução do efeito de arquitetura. Isso *é* mexer na abordagem, então
  entra como ablação, não como a linha principal.

### Regra proposta para o `retfound_green`

Ele entra como **baseline importado**. Só é lícito tocar em:
1. **instrumentação** (obrigatório, senão não se mede nada);
2. **parâmetros de entrada** (`batch_size`, épocas, caminhos dos splits, seed).

**Proibido:** arquitetura, perda, scheduler, resolução, augmentação, determinismo.
Tudo o mais deve ser o upstream. Hoje há **uma** violação: o determinismo, que nós
introduzimos — e ela deve ser desfeita.

---

# P11 RESOLVIDO — o `torch.compile` COMPILA de fato 🟢

A suspeita era legítima (`suppress_errors=True` esconde falhas), mas a **evidência
dos logs de produção da GH200 a refuta**. Três assinaturas independentes:

### 1. Salto de compilação na 1ª época
```
                        ep0      ep1     salto
pytorch_opt (compile)  95,0 s   4,3 s   22,2x     <-- 1a passada compila os kernels
pytorch_base (eager)   17,8 s   8,6 s    2,1x     <-- so cudnn.benchmark
```
22× vs 2× é a diferença entre "compilou" e "só aqueceu o cuDNN".

### 2. Ganho em regime estável (mesma GPU)
```
tensorflow_opt (XLA)          : 2,86 s/epoca
pytorch_opt    (TorchInductor): 4,31 s/epoca   <-- 1,77x mais rapido que...
pytorch_base   (eager)        : 7,64 s/epoca   <-- ...o eager
```
Se o `pytorch_opt` rodasse em eager, seria ~7,6 s como o base. Roda a 4,3 s.

### 3. TorchInductor gera Triton neste ambiente
Um log com `mode=max-autotune` mostra `triton_convolution2d_*` sendo autotunado
(`ALLOW_TF32=True, BLOCK_K=16, ...`). O backend Triton funciona na GH200.

### Por que os logs não mencionam "triton/inductor"
Os 10 runs finais usaram `mode=reduce-overhead`, que **não imprime** o autotuning
(diferente de `max-autotune`). Ausência de log ≠ ausência de compilação. Meu
argumento anterior dos "1.633 kernels/batch" também estava errado: CUDA graphs
reduz o *overhead de lançamento*, não o número de kernels que o CUPTI conta.

**Conclusão:** o par controlado é legitimamente **XLA vs TorchInductor**, não
"XLA vs eager". P11 encerrado.

### Nota de consistência (verificar antes da re-rodada)
Os logs mostram DOIS modos usados ao longo do projeto: `max-autotune` (1 run
antigo) e `reduce-overhead` (os 10 finais). O código atual está em
`reduce-overhead`. A re-rodada final deve fixar UM modo para todos — recomendo
`max-autotune` no par controlado (compilação mais agressiva, melhor isola o teto
do runtime) e declarar a escolha.

---

# CORRECOES APLICADAS (2026-07-10, fim de semana)

| # | Problema | Status | O que foi feito |
|---|---|---|---|
| P1 | val == test | ✅ | `tools/make_val_split.py` cria `data/all-tfrec-v2`: train 9350 / val 1650 / test 4816. Split POR PACIENTE, prevalencia identica (0,267), 0 vazamento. Examples preservados byte-a-byte (blocos brutos, CRC original). |
| P1b | as 8 usarem val* | ✅ | `tensorflow_opt`: VALID=val*, TEST=test* separados (validacao por epoca vs teste final). Os outros 7 ja separavam via bridge/flags — so faltava o val* existir. |
| P2 | vazamento por paciente | ✅ (nao era) | Verificado nos TFRecords via `image_name`: 0 imagens repetidas, 0 pacientes em train∩test (numericos e DICOM). O split original ja respeitava paciente. |
| P3 | AUC binada no TF | ✅ | `tensorflow_opt`: `exact_eval_interval 5 -> 1` (AUC exata toda epoca). Os outros ja usavam AUC exata sempre. |
| P4 | cosine desligado no htr_opt | ✅ | `ENABLE_COSINE 0 -> 1`. |
| P5 | drop_last divergente | ✅ | `tensorflow_opt`: steps `ceil -> floor` (114 batches). TF base ja usava drop_remainder; pytorch+hybrids ja usavam drop_last=shuffle. Todas: 114 batches cheios. |
| P6 | determinismo so no retfound | ✅ | Removido `cudnn.deterministic`/`use_deterministic_algorithms` (nao estava no upstream; era nosso). `cudnn.benchmark=True` como as demais. |
| P8 | retfound em 392px | ✅ (nao mexer) | 392 = 28×14 e a config ORIGINAL (patch 14; 299 nem seria divisivel). Manter e declarar 1,72x pixels. |
| P11 | torch.compile inativo | ✅ (compila) | Salto ep0/ep1 = 22x, regime 1,77x < eager, Triton no log max-autotune. E XLA-vs-TorchInductor de fato. |
| P7 | receitas heterogeneas | 📋 declarar | Enquadrar: comparacao de ABORDAGENS (nao de frameworks, exceto o par _opt). |
| P9 | sem plano estatistico | 📋 declarar | n=10, mesmo no por abordagem, IC bootstrap, teste pareado. |
| P10 | dominio de energia NVML GH200 | ⏳ | `tools/g5k_energy_domain_check.py` — rodar na chuc-2. |

**Falta validar em GPU (chuc-2, 12h):** smoke-test das 8 com o codigo+dataset
corrigidos (val* para selecao, test* no fim, AUC exata, drop_last, cosine) + P10.

**NAO rodado ainda:** a re-rodada final 8×10 (precisa de bloco grande, so cabe a
partir de segunda 10:04 no cluster — e horario diurno, consome cota).
