# HCPA no Grid5000 — alvos multi-site (bronze)

Estudo cross-GPU no G5K. **3 GPUs bronze-acessíveis** (prod=NO). H100 ficou de fora:
só existe em clusters `production=YES` (Rennes/Sophia), sem acesso na fila default da conta bronze.

| Alvo (`g5k_target.sh`) | GPU | Cluster | Site | Arch | exotic | Deploy image |
|---|---|---|---|---|---|---|
| `lyon-hydra` | GH200 480GB (96GB) | hydra | Lyon | ARM aarch64 | sim | `ubuntugh2404-arm64-big` |
| `lille-chuc` | A100-SXM4-40GB | chuc | Lille | x86_64 | não | `debiannvopen11-big` |
| `lille-chicoree` | H200 NVL 140GB | chicoree | Lille | x86_64 | sim | `debiannvopen11-big` |

Mais L40S + RTX4090 do GPPD = **5 GPUs** no estudo final.

## Arquitetura: containers NGC são multi-arch
As imagens base (`nvcr.io/nvidia/pytorch:24.11-py3`, `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`)
têm manifest amd64 **e** arm64 — o Singularity puxa a arquitetura certa automaticamente. Os 9
`Apptainer.def` agora também garantem `pynvml` (via `nvidia-ml-py`) p/ a coleta de energia/memória.

## Fluxo completo

### 1. No GPPD (este box) — enviar código + dados
```bash
bash tools/g5k_target.sh lyon-hydra     send   # GH200 (dataset já estava lá)
bash tools/g5k_target.sh lille-chuc     send   # A100 + H200 compartilham o home de Lille
# (não precisa enviar p/ lille-chicoree separado: mesmo home de Lille)
```

### 2. No frontend do site — alocar nó + build das imagens (janela bronze!)
```bash
# ssh -J bmorales@access.grid5000.fr bmorales@flyon.lyon.grid5000.fr   (ou flille...)
WALLTIME=08:00:00 NIGHT_JOB=1 bash tools/g5k_target.sh lyon-hydra setup
# idem lille-chuc / lille-chicoree no frontend de Lille
```
`setup` faz: oarsub (`-t deploy` + `-t exotic` quando aplicável) → kadeploy → instala
Singularity → builda `hcpa.sif` das 9 abordagens → grava `~/.g5k_hcpa/state.env`.

### 3. No frontend — rodar (0..9 por abordagem)
```bash
bash tools/g5k_target.sh lille-chuc run pytorch_opt   # uma abordagem
bash tools/g5k_target.sh lille-chuc run-all           # as 9 abordagens
```
Cada run mantém **apenas `*.csv` e `*.pdf`** em
`<approach>/results/<gpu>_g5k_<cluster>/run_<id>/` (whitelist no nó).

### 4. No GPPD — puxar resultados de volta
```bash
bash tools/g5k_target.sh lille-chuc fetch         # traz csv+pdf p/ <approach>/results/ no GPPD
bash tools/g5k_target.sh lille-chuc fetch-purge   # idem + libera quota no G5K
```
> Rede: o G5K **não** alcança o GPPD (firewall UFRGS), então o "mandar de volta" é
> sempre **pull** disparado daqui. Rode `fetch` quando as 0..9 terminarem.

## Bronze (lembretes)
- Janela noturna/FDS em **hora de Paris**; conta `bmorales` é morta no fechamento. Use
  `NIGHT_JOB=1` e `WALLTIME` que caiba na janela. Ver memory `reference-grid5000-bronze`.
- `chuc` (A100) **não** precisa de exotic; `hydra` e `chicoree` precisam (o preset já cuida).
- SSH é jump único pelo `access.grid5000.fr` (resolve o DNS interno dos frontends).
