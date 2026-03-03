# monai_base — MONAI puro (PyTorch)

Clonamos a versão otimizada e a simplificamos: nada de DALI, nada de timm, sem scheduler de LR. Apenas MONAI + InceptionV3 com batch 96 e 200 épocas.

## Destaques
- MONAI + torchvision InceptionV3 (sem timm, sem DALI).
- Batch 96, 200 épocas, LR constante (scheduler desativado).
- Mixup, Cutmix, label smoothing e grad clip mantidos.
- Métricas AUC/sens/espec/acurácia e throughput por época; checkpoints em `checkpoint.pt`.

## Treino recomendado
```bash
python3 train.py \
  --tfrec_dir ./data/all-tfrec \
  --results results/monai_puro \
  --batch_size 96 --epochs 200 \
  --model inception_v3
```

## Avaliação
```bash
python3 eval.py --results results/monai_puro
```

## Benchmark (usa dados sintéticos por padrão)
```bash
python3 benchmark.py --batch_size 96 --warmup_steps 20 --measure_steps 200
```

## Notas
- Execute com `torchrun --nproc_per_node=NUM_GPU train.py ...` para DDP.
- Ajuste `--mixup_alpha`/`--cutmix_alpha` para agressividade.
- `--gradient_accumulation` permite batches globais maiores sem estourar memória.
