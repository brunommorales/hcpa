# monai_opt — performance máxima (PyTorch/MONAI)

Versão otimizada para throughput/latência, com pipeline agressivo de dados e treino.

## Destaques
- MONAI + timm (`inception_v3` por default) com `channels_last`, AMP e `torch.compile` habilitados.
- DALI opcional para leitura de TFRecords em GPU (instale manualmente `nvidia-dali-cuda120` se quiser usar `--use_dali`).
- Mixup/Cutmix, label smoothing, grad clip, EMA, OneCycle/Cosine schedulers.
- Métricas AUC/sens/espec/acurácia e throughput por época; checkpoints em `checkpoint.pt`.

## Treino recomendado
```bash
python3 train.py \
  --tfrec_dir ./data/all-tfrec \
  --results results/opt_monai \
  --batch_size 96 \
  --epochs 200 \
  --model inception_v3 \
  --use_dali
```

## Avaliação
```bash
python3 eval.py --results results/opt_monai
```

## Benchmark (usa dados sintéticos por padrão)
```bash
python3 benchmark.py --batch_size 96 --warmup_steps 20 --measure_steps 200
```

## Notas
- Execute com `torchrun --nproc_per_node=NUM_GPU train.py ...` para DDP.
- Ajuste `--mixup_alpha`/`--cutmix_alpha` para agressividade; use `--no_dali` se DALI não estiver disponível.
- `--gradient_accumulation` permite batches globais maiores sem estourar memória.
