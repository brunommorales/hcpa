# HCPA Autotuner

Wrapper de autoajuste online para treino de classificação de Retinopatia Diabética (DR/HCPA).

Suporta 6 variantes existentes (3 stacks × 2 modos):

| Stack      | Base                          | Optimized                           |
|------------|-------------------------------|-------------------------------------|
| PyTorch    | Keras-on-torch, Adam, manual  | timm, AdamW, DALI, EMA, compile     |
| TensorFlow | tf.data, Adam, Keras          | DALI, mixed-precision, freeze/unfreeze |
| MONAI      | torchvision, no scheduler     | timm, cosine, DALI, EMA, compile     |

## Princípio fundamental

> **Não inventa knobs.** O autotuner só pode ajustar parâmetros que já existam
> nas 6 variantes base/opt. Nenhum hiperparâmetro novo é introduzido.

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                       main.py (CLI)                     │
├───────────┬──────────────┬──────────────┬───────────────┤
│  ETAPA 0  │   ETAPA 1    │  ETAPA 2-3   │   ETAPA 4    │
│   GPU     │    Audit &   │  Controller  │   Report     │
│ Discovery │ Derived Space│ + Backends   │   & CSV      │
├───────────┼──────────────┼──────────────┼───────────────┤
│gpu_discov.│  audit.py    │ controller.py│ logging_csv  │
│gpu_monitor│derived_space │ safety.py    │              │
│           │              │ backends/*   │              │
└───────────┴──────────────┴──────────────┴───────────────┘
```

## Módulos

| Módulo              | Responsabilidade |
|---------------------|------------------|
| `gpu_discovery.py`  | Detecta GPUs via nvidia-smi; lê driver, memória, utilização, temperatura, potência |
| `gpu_monitor.py`    | Monitor assíncrono (thread) com pynvml ou fallback nvidia-smi |
| `audit.py`          | Auditoria completa dos parâmetros de todas as 6 variantes |
| `derived_space.py`  | Espaço de Configuração Derivado — `ParamSpec` com range, tipo, tunable_online |
| `controller.py`     | AutoTuneController — ajustes epoch-by-epoch baseados em sinais |
| `safety.py`         | NaN detection, OOM detection, loss spike, checkpoint/rollback |
| `logging_csv.py`    | Logger CSV com 25+ campos (métricas, GPU, ações, config) |
| `backends/`         | PyTorchBackend, TensorFlowBackend, MonaiBackend |

## Instalação

```bash
# Nenhuma dependência além do Python 3.9+
# pynvml é opcional (melhora GPU monitoring)
pip install pynvml   # opcional
```

---

## Integração com SLURM — Como funciona na prática

O fluxo atual de execução no cluster (Grid5000 / tupi) é:

```
sbatch distributed_run_x86.slurm
  └─→ SLURM aloca nós + GPUs
       └─→ srun lança Docker container por nó
            └─→ torchrun train.py --batch_size 96 --epochs 200 ...
```

Com o autotuner, o fluxo passa a ser:

```
sbatch autotuner_run.slurm
  └─→ SLURM aloca nós + GPUs  (mesmos #SBATCH de antes)
       └─→ srun lança Docker container por nó
            └─→ python -m src.main --stack monai --mode opt --path /workspace
                  ├─ ETAPA 0: detecta GPU (nvidia-smi)
                  ├─ ETAPA 1: auditoria + espaço derivado
                  └─ ETAPA 2-3: lança "torchrun train.py ..." como subprocess
                       ├─ parseia stdout epoch-by-epoch
                       ├─ consulta GPU monitor (memória, utilização, temperatura)
                       ├─ aplica ajustes online (LR, batch_size, mixup, etc.)
                       └─ loga tudo em CSV + checkpoint
```

### O que muda e o que NÃO muda

| Aspecto                      | Antes (sem autotuner)         | Agora (com autotuner)                    |
|------------------------------|-------------------------------|------------------------------------------|
| Headers `#SBATCH`            | Mantém                        | **Idênticos** — mesmos nós, GPUs, tempo  |
| Docker setup                 | Mantém                        | **Idêntico** — mesma imagem, volumes     |
| Rede NCCL/Gloo               | Mantém                        | **Idêntico** — mesma detecção de iface   |
| Código de treino (`train.py`)| Executado diretamente          | Executado **pelo autotuner** como subprocess |
| Parâmetros do treino         | Fixos no `.slurm`             | Gerenciados pelo autotuner (com overrides) |
| Monitoramento GPU            | Manual / nenhum               | **Automático** (pynvml + nvidia-smi)     |
| Ajuste mid-training          | Impossível                    | **Automático** (5 políticas de ajuste)   |
| Logs estruturados            | Apenas stdout                 | **CSV epoch-by-epoch** + controller state |

### Como adaptar seu `.slurm` existente

A mudança é **mínima**: apenas a linha final de comando dentro do Docker. O resto do script (SBATCH, Docker setup, rede) permanece igual.

**Antes** (monai_opt/distributed_run_x86.slurm):
```bash
    torchrun \
      --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
      --node_rank=${NODE_RANK} \
      --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
      train.py \
        --results '${RUN_DIR_CONT}' \
        --tfrec_dir '${CONT_TFREC_DIR}' \
        --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} \
        --learning_rate ${LR} ...
```

**Depois** (com autotuner):
```bash
    python -m src.main \
      --stack monai --mode opt \
      --path /workspace \
      --output '${RUN_DIR_CONT}/autotuner_output' \
      --override \
        tfrec_dir='${CONT_TFREC_DIR}' \
        results_dir='${RUN_DIR_CONT}' \
        epochs=${EPOCHS} \
        batch_size=${BATCH_SIZE} \
        learning_rate=${LR}
```

O autotuner internamente vai montar o comando `torchrun train.py ...` com todos os argumentos, parsear a saída, e ajustar o que for possível em tempo real.

### Template SLURM completo

Um script SLURM de referência está disponível em `slurm/autotuner_run.slurm`. Para usar:

```bash
# MONAI opt no tupi4
cd /home/users/bmmorales/projects/hcpa/autotuner
sbatch slurm/autotuner_run.slurm

# Com overrides de variáveis de ambiente
STACK=pytorch MODE=opt VARIANT_DIR=/home/users/bmmorales/projects/hcpa/pytorch_opt \
  sbatch slurm/autotuner_run.slurm

# Sem autotuning (apenas logging + monitoramento)
ENABLE_TUNING=0 sbatch slurm/autotuner_run.slurm

# Apenas auditoria (não treina, gera relatório)
AUDIT_ONLY=1 sbatch slurm/autotuner_run.slurm
```

### Fluxo detalhado dentro do container

```
┌──────────────────────────────────────────────────────────────────────┐
│  Docker container (mesma imagem: monai_opt:latest, etc.)             │
│                                                                      │
│  1. python -m src.main --stack monai --mode opt --path /workspace    │
│     │                                                                │
│     ├─ GPU Discovery: nvidia-smi → detecta GPU, VRAM, driver        │
│     │                                                                │
│     ├─ Audit: carrega espaço derivado (31 params para MONAI)        │
│     │                                                                │
│     ├─ Controller: inicializa config (base ou opt values)            │
│     │   └─ aplica --override se fornecidos                           │
│     │                                                                │
│     ├─ GPU Monitor: thread em background a cada 5s                   │
│     │                                                                │
│     ├─ Subprocess: torchrun --nnodes=1 --nproc_per_node=1 train.py  │
│     │   │                                                            │
│     │   ├─ Epoch 1: train_loss=0.85 val_auc=0.72 → OK, sem ajuste  │
│     │   ├─ Epoch 2: train_loss=0.71 val_auc=0.78 → GPU mem=22GB OK │
│     │   ├─ ...                                                       │
│     │   ├─ Epoch 30: val_auc estagnado 5 épocas → LR ×0.5 (plateau)│
│     │   ├─ Epoch 45: loss spike 5× → label_smoothing +0.01          │
│     │   ├─ Epoch 80: GPU mem 95% → batch_size 96→64 (OOM risk)     │
│     │   └─ Epoch 200: fim                                            │
│     │                                                                │
│     ├─ CSV log: autotuner_output/autotuner_log.csv                   │
│     ├─ Controller state: autotuner_output/checkpoints/               │
│     └─ Relatório final: sumário de todas ações tomadas               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Nota sobre ajustes que requerem restart

Alguns parâmetros (ex: `model`, `img_size`, `use_dali`, `compile`) são marcados como
`requires_restart=True` no espaço derivado. Estes **não podem ser alterados mid-training**
porque o script de treino os lê apenas no início. O controller pode recomendar mudanças
para esses parâmetros, mas elas só serão aplicadas se o treino for relançado com `--resume`.

Parâmetros `tunable_online=True` (ex: learning_rate, batch_size, mixup_alpha,
label_smoothing) são os únicos ajustados automaticamente durante o treino.

---

## Uso local (sem SLURM)

### Execução completa

```bash
cd /path/to/autotuner

# PyTorch base
python -m src.main --stack pytorch --mode base --path /path/to/pytorch_base

# PyTorch opt
python -m src.main --stack pytorch --mode opt --path /path/to/pytorch_opt

# TensorFlow base
python -m src.main --stack tensorflow --mode base --path /path/to/tensorflow_base

# TensorFlow opt com tuning desligado (dry)
python -m src.main --stack tensorflow --mode opt --path /path/to/tensorflow_opt --disable-tuning

# MONAI opt com overrides
python -m src.main --stack monai --mode opt --path /path/to/monai_opt \
    --override batch_size=64 learning_rate=1e-4

# Retomar de checkpoint
python -m src.main --stack monai --mode opt --path /path/to/monai_opt --resume
```

### Apenas auditoria (sem treinar)

```bash
python -m src.main --stack pytorch --mode base --path /path/to/pytorch_base --audit-only
```

Gera:
- `autotuner_output/audit_report.txt` — relatório de auditoria completo
- `autotuner_output/derived_space_schema.json` — schema JSON do espaço derivado

### Opções CLI

| Flag                  | Default            | Descrição |
|-----------------------|--------------------|-----------|
| `--stack`             | (obrigatório)      | `pytorch`, `tensorflow` ou `monai` |
| `--mode`              | (obrigatório)      | `base` ou `opt` |
| `--path`              | (obrigatório)      | Diretório raiz da variante |
| `--output`            | `./autotuner_output` | Diretório de saída |
| `--enable-tuning`     | `True`             | Habilita autoajuste online |
| `--disable-tuning`    | —                  | Desabilita autoajuste |
| `--monitor-interval`  | `5.0`              | Intervalo GPU monitor (s) |
| `--audit-only`        | `False`            | Só audita, não treina |
| `--resume`            | `False`            | Retoma de checkpoint |
| `--override`          | —                  | Overrides: `key=value ...` |

## Espaço de Configuração Derivado

Cada parâmetro no espaço derivado tem:

```python
@dataclass
class ParamSpec:
    name: str                      # ex: "batch_size", "lrate"
    param_type: str                # "int", "float", "bool", "str"
    source_variants: List[str]     # ex: ["pytorch_base", "pytorch_opt"]
    base_value: Any                # valor na variante base
    opt_value: Any                 # valor na variante opt
    range: Optional[Tuple]         # (min, max) se numérico
    choices: Optional[List]        # valores discretos permitidos
    tunable_online: bool           # pode ser ajustado mid-training?
    requires_restart: bool         # requer restart para aplicar?
    description: str               # descrição do parâmetro
```

### Estratégias de autoajuste online

O controller aplica 5 políticas baseadas em sinais reais:

1. **ROLLBACK on Divergence** — Se NaN ou divergência detectada, reverte para último checkpoint estável
2. **Batch Size Reduction on OOM** — Se memória GPU > 90%, reduz batch_size
3. **LR Reduction on Plateau** — Se val_AUC estagna por 5 épocas, reduz LR por 0.5×
4. **Mixup Activation** — Se GPU utilization < 30%, ativa mixup para regularização
5. **Label Smoothing Increase** — Se loss spike detectado (5× do baseline), aumenta label_smoothing

## Saída

```
autotuner_output/
├── audit_report.txt            # Relatório de auditoria (ETAPA 1)
├── derived_space_schema.json   # Schema JSON do espaço derivado
├── autotuner_log.csv           # Log CSV epoch-by-epoch
├── train_output.log            # Saída stdout do treino
├── results/                    # Resultados do treino
└── checkpoints/
    └── controller_state.json   # Estado do controller p/ resume
```

## Como adicionar uma nova variante

1. Crie um backend em `src/backends/my_backend.py` que herda de `BackendBase`
2. Implemente: `get_entry_point()`, `build_command()`, `config_to_cli_args()`, `parse_epoch_metrics()`, `validate()`
3. Adicione os parâmetros em `audit.py` e `derived_space.py`
4. Registre no `backends/__init__.py`

## Estrutura de arquivos

```
autotuner/
├── README.md
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py               # CLI + orquestrador principal
│   ├── gpu_discovery.py       # ETAPA 0: detecção de GPU
│   ├── gpu_monitor.py         # Monitor assíncrono de GPU
│   ├── audit.py               # ETAPA 1: auditoria das 6 variantes
│   ├── derived_space.py       # Espaço de configuração derivado
│   ├── controller.py          # AutoTuneController (ajuste online)
│   ├── safety.py              # NaN/OOM/spike detection + rollback
│   ├── logging_csv.py         # Logger CSV estruturado
│   └── backends/
│       ├── __init__.py
│       ├── base.py            # BackendBase (ABC)
│       ├── pytorch_backend.py # Backend PyTorch (base & opt)
│       ├── tensorflow_backend.py  # Backend TensorFlow (base & opt)
│       └── monai_backend.py   # Backend MONAI (base & opt)
```
