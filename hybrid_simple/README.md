# Hybrid Simple - CNN + Transformer

Modelo híbrido que combina um backbone CNN pré-treinado com um encoder Transformer para classificação de retinopatia diabética.

## Arquitetura

```
Imagem → CNN Backbone → Feature Map [B, C, H, W]
                      ↓
               Flatten Espacial [B, N, D]
                      ↓
          Positional Encoding + Transformer (4 camadas)
                      ↓
         Agregação (Mean Pooling ou CLS Token)
                      ↓
          Linear Classification Head
                      ↓
                 Logits [B, 1]
```

## Componentes Principais

### Model (`model.py`)
- **`HybridSimple`**: Classe principal do modelo
- **`MultiHeadAttention`**: Implementação de atenção multi-head
- **`TransformerEncoderLayer`**: Camada individual de Transformer
- **`TransformerEncoder`**: Stack de camadas Transformer
- **`PositionalEncoding`**: Encoding posicional dos tokens

### Características

- **Backbone pré-treinado**: Reutiliza modelo CNN do `pytorch_opt`
- **Transformer leve**: 4 camadas, 4 heads, configurável
- **Agregação flexível**: Mean pooling ou CLS token
- **AMP opcional**: Suporte a Automatic Mixed Precision (FP16), desativado por padrão
- **DDP support**: Treinamento distribuído multi-GPU

## Como Usar

### 1. Treinamento

```bash
cd /home/users/bmmorales/projects/hcpa/hybrid_simple

python train.py \
  --tfrec_dir /home/users/bmmorales/projects/hcpa/data/all-tfrec \
  --batch_size 32 \
  --epochs 50 \
  --backbone inception_v3 \
  --num_transformer_layers 4 \
  --num_heads 4 \
  --enable_ema
```

#### Argumentos Principais

- `--tfrec_dir`: Diretório com TFRecords
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Número de épocas (default: 50)
- `--backbone`: Backbone CNN (inception_v3, resnet50, efficientnet_b0, etc.)
- `--num_transformer_layers`: Camadas Transformer (default: 4)
- `--num_heads`: Heads de atenção (default: 4)
- `--img_size`: Tamanho da imagem (default: 299)
- `--lrate`: Learning rate (default: 1e-4)
- `--optimizer`: adamw, adam, sgd, rmsprop (default: adamw)
- `--enable_amp`: Ativar Automatic Mixed Precision
- `--disable_amp`: Garantir AMP desligado (padrão do hybrid_simple)
- `--enable_ema`: Ativar Exponential Moving Average
- `--enable_dali`: Usar NVIDIA DALI pipeline (recomendado)
- `--use_cls_token`: Usar CLS token em vez de mean pooling
- `--freeze_backbone_epochs`: Épocas com backbone congelado (default: 3)
- `--warmup_epochs`: Warmup linear (default: 5)
- `--label_smoothing`: Label smoothing factor (default: 0.0)
- `--focal_gamma`: Focal loss gamma (default: 0.0)
- `--mixup_alpha`: Mixup alpha (default: 0.0)
- `--cutmix_alpha`: CutMix alpha (default: 0.0)
- `--seed`: Random seed (default: 42)

### 2. Avaliação

```bash
python evaluate.py \
  --tfrec_dir /home/users/bmmorales/projects/hcpa/data/all-tfrec \
  --batch_size 64 \
  --backbone inception_v3 \
  --exec_id 0
```

Gera arquivo `eval_results_exec0.txt` com AUC, Sensibilidade, Especificidade.

### 3. Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=2 train.py \
  --tfrec_dir /home/users/bmmorales/projects/hcpa/data/all-tfrec \
  --batch_size 32
```

## Hiperparâmetros Recomendados

### Para Quick Training (Debug)
```bash
--batch_size 16 --epochs 5 --num_transformer_layers 2 --num_heads 2
```

### Para Produção
```bash
--batch_size 32 --epochs 100 --num_transformer_layers 4 --num_heads 4 \
--enable_ema --enable_dali --warmup_epochs 5 --freeze_backbone_epochs 3
```

## Estrutura de Arquivos

```
hybrid_simple/
├── __init__.py           # Export principal
├── model.py              # Implementação do modelo
├── train.py              # Script de treinamento
├── evaluate.py           # Script de avaliação
├── config.py             # Argumentos CLI
├── utils.py              # Utilitários
├── README.md             # Este arquivo
└── results/              # Diretório de saída (criado automaticamente)
    ├── best_model_exec0.pt
    ├── summary_exec0.txt
    └── eval_results_exec0.txt
```

## Saídas

### Durante Treinamento
- **Logs no console**: Loss, AUC, Sensibilidade, Especificidade, Throughput a cada época
- **Checkpoint**: `best_model_exec{id}.pt` - Melhor modelo baseado em validação AUC
- **Summary**: `summary_exec{id}.txt` - Resumo final de treinamento

### Após Avaliação
- **Resultados**: `eval_results_exec{id}.txt` - Métricas clínicas em validação e teste

## Performance

### Benchmark Esperado (com Inception-V3 backbone)

| Métrica | Valor Típico |
|---------|-------------|
| Train Throughput | ~80-100 img/s (GPU) |
| AUC (Validação) | ~0.87-0.92 |
| Sensibilidade | ~0.80-0.85 |
| Especificidade | ~0.88-0.93 |
| Tempo/Época | ~30-50s (batch 32) |
| Memória Pico | ~4-6 GB (batch 32) |

## Integração com pytorch_opt

Este modelo **reutiliza**:
- Dataset em formato TFRecord
- DALI pipeline para decodificação GPU
- Backbone CNN timm
- Normalização automática
- Loss functions
- Metrics

**Não modifica** o projeto `pytorch_opt` - funciona como extensão independente.

## Troubleshooting

### DALI não disponível
Se DALI não funcionar, o código automaticamente cai para PyTorch DataLoader:
```bash
--enable_dali  # Remova esta flag se tiver problemas
```

### Erro de memória
Reduza batch size:
```bash
--batch_size 16  # ou menor
```

### Divergência de loss
Reduza learning rate ou ative warmup:
```bash
--lrate 5e-5 --warmup_epochs 10
```

## Comparação de Variantes

Veja `../compare_models.py` para comparar:
- Baseline CNN (pytorch_opt)
- Hybrid Simple (este modelo)
- Hybrid Token Reduction (com seleção de tokens)

## Referências

- ViT (Vision Transformer): https://arxiv.org/abs/2010.11929
- Transformer Architecture: https://arxiv.org/abs/1706.03762
- Timm Models: https://github.com/huggingface/pytorch-image-models
