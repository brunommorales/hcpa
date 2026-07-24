# Pure ViT - Raw Image Patches

Modelo `Vision Transformer` puro para classificação de retinopatia diabética.

Aqui os tokens **não** vêm de uma CNN. Eles vêm diretamente de **patches crus da imagem**:

```text
Imagem [B, 3, H, W]
  -> PatchEmbedding Conv2d(kernel=patch_size, stride=patch_size)
  -> Tokens [B, N, D]
  -> CLS token + positional embedding
  -> Transformer encoder
  -> Head linear
  -> Logit [B, 1]
```

## Origem dos tokens

Cada token representa um patch espacial da imagem original.

- Se `img_size=320` e `patch_size=16`
- A grade é `20 x 20`
- O número de patches é `400`
- A sequência do Transformer é `401` com `CLS token`

No código isso acontece em [model.py](/home/users/bmmorales/projects/hcpa/vit_pure/model.py):

- `PatchEmbedding.proj = Conv2d(..., kernel_size=patch_size, stride=patch_size)`
- Cada janela da convolução corresponde a um patch cru
- Depois o mapa é achatado para `[B, N, D]`

## Treinamento

```bash
cd /home/users/bmmorales/projects/hcpa/vit_pure

python train.py \
  --tfrec_dir /home/users/bmmorales/projects/hcpa/data/all-tfrec \
  --img_size 320 \
  --patch_size 16 \
  --batch_size 32 \
  --gradient_accumulation_steps 3 \
  --epochs 200 \
  --enable_amp \
  --enable_ema \
  --disable_flash_attention
```

Ou via SLURM:

```bash
cd /home/users/bmmorales/projects/hcpa/vit_pure
sbatch train.slurm
```

Exemplo com overrides:

```bash
IMG_SIZE=320 PATCH_SIZE=16 BATCH_SIZE=32 GRADIENT_ACCUMULATION_STEPS=3 EPOCHS=200 sbatch train.slurm
```

Por padrão o `vit_pure` usa atenção manual estável. Para testar SDPA/Flash explicitamente:

```bash
ENABLE_FLASH_ATTENTION=1 sbatch distributed_run_x86.slurm
```

## Avaliação

```bash
python evaluate.py \
  --tfrec_dir /home/users/bmmorales/projects/hcpa/data/all-tfrec \
  --img_size 320 \
  --patch_size 16 \
  --batch_size 32
```

A análise completa de patches/atenções fica desativada por padrão para evitar alto consumo de memória. Para habilitar:

```bash
python evaluate.py --collect_patch_analysis ...
```

## Relação com os outros modelos

- `pytorch_opt`: CNN puro
- `hybrid_simple`: CNN + Transformer
- `hybrid_token_reduction`: CNN + seleção de tokens + Transformer
- `vit_pure`: Transformer puro com tokens extraídos de patches crus

## Observações

- O `img_size` precisa ser divisível por `patch_size`
- O pipeline de dados, métricas, profiler e scheduler reaproveitam `hybrid_shared`
- A normalização usada é a padrão ImageNet, adequada para ViT
