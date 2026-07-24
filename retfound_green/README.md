# RETFound-Green Fine-Tuning

Baseline rapido para testar o backbone/pesos do **RETFound-Green** no pipeline atual do `hcpa`, sem portar o pre-treino self-supervised deles.

## O que este modulo faz

- carrega o backbone publico `vit_small_patch14_reg4_dinov2`
- aplica os pesos RETFound-Green da release oficial
- adiciona uma head binaria `Linear(384 -> 1)`
- reaproveita o pipeline de dados, metricas, profiler e scheduler do projeto

## Pesos

O caminho esperado por padrao e:

`/home/users/bmmorales/projects/hcpa/retfound_green/weights/retfoundgreen_statedict.pth`

Baixe os pesos com:

```bash
mkdir -p /home/users/bmmorales/projects/hcpa/retfound_green/weights
wget https://github.com/justinengelmann/RETFound_Green/releases/download/v0.1/retfoundgreen_statedict.pth \
  -O /home/users/bmmorales/projects/hcpa/retfound_green/weights/retfoundgreen_statedict.pth
```

## Treinamento

Slurm direto pela home, sem staging para SSD/scratch:

```bash
cd /home/users/bmmorales/projects/hcpa/retfound_green
sbatch train.slurm
```

Isso executa o array `0-9%1` e grava artefatos em:

`/home/users/bmmorales/projects/hcpa/retfound_green/results/result<job>_<node>_<gpu>_bs<batch>/run_<exec_id>/`

Para um teste curto de apenas um run:

```bash
cd /home/users/bmmorales/projects/hcpa/retfound_green
EPOCHS=5 BATCH_SIZE=64 sbatch --array=0-0 train.slurm
```

Slurm distribuido x86/ARM, tambem direto pela home por padrao:

```bash
cd /home/users/bmmorales/projects/hcpa/retfound_green
sbatch distributed_run_x86.slurm
sbatch distributed_run_arm.slurm
```

Para fixar um no especifico, passe no submit:

```bash
sbatch --nodelist=tupi5 distributed_run_x86.slurm
sbatch --nodelist=grace1 distributed_run_arm.slurm
```

```bash
cd /home/users/bmmorales/projects/hcpa/retfound_green

python train.py \
  --tfrec_dir /home/users/bmmorales/projects/hcpa/data/all-tfrec \
  --backbone_checkpoint /home/users/bmmorales/projects/hcpa/retfound_green/weights/retfoundgreen_statedict.pth \
  --img_size 392 \
  --batch_size 64 \
  --epochs 100 \
  --lrate 5e-5 \
  --enable_amp
```

Treino apenas da head:

```bash
python train.py \
  --backbone_checkpoint /home/users/bmmorales/projects/hcpa/retfound_green/weights/retfoundgreen_statedict.pth \
  --freeze_backbone \
  --enable_amp
```

## Avaliacao

```bash
python evaluate.py \
  --results_dir /home/users/bmmorales/projects/hcpa/retfound_green/results \
  --img_size 392 \
  --batch_size 32 \
  --enable_amp
```

## Normalizacao

O RETFound-Green usa media e desvio `0.5` por canal. Para manter o modulo isolado, eu nao alterei o `pytorch_opt`: o `retfound_green` reaproveita internamente a normalizacao ja existente de `inception_v3`, que e numericamente identica:

- `(x - 127.5) / 127.5`
- equivalente a normalizar com `mean=0.5` e `std=0.5` em imagens no intervalo `[0, 1]`

## Observacoes

- O `img_size` deve ser divisivel por `14`
- O checkpoint salvo pelo treino contem backbone + head, entao a avaliacao nao precisa recarregar o peso original
- O modulo compara de forma justa com o `vit_pure`, porque reutiliza o mesmo loader e as mesmas metricas
