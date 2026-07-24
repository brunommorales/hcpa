"""
Data Loader Bridge - Wrapper para integração com pytorch_opt dataloaders.

Facilita uso do DALI pipeline ou PyTorch DataLoader do pytorch_opt.
"""

import importlib
import os
import sys
from functools import lru_cache
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _iter_pytorch_opt_candidates():
    seen = set()
    raw_candidates = []

    env_candidate = os.environ.get("PYTORCH_OPT_DIR")
    if env_candidate:
        raw_candidates.append(Path(env_candidate).expanduser())

    raw_candidates.append(PROJECT_ROOT / "pytorch_opt")
    raw_candidates.append(Path.cwd() / "pytorch_opt")

    for entry in sys.path:
        if not entry:
            continue
        entry_path = Path(entry).expanduser()
        raw_candidates.append(entry_path)
        raw_candidates.append(entry_path / "pytorch_opt")

    raw_candidates.extend([
        Path("/workspace/pytorch_opt"),
        Path("/home/users/bmmorales/projects/hcpa/pytorch_opt"),
    ])

    for candidate in raw_candidates:
        try:
            candidate = candidate.resolve(strict=False)
        except OSError:
            continue

        candidate_key = str(candidate)
        if candidate_key in seen:
            continue

        seen.add(candidate_key)
        yield candidate


@lru_cache(maxsize=1)
def _load_pytorch_opt_module():
    attempted_locations = []
    import_errors = []

    for candidate in _iter_pytorch_opt_candidates():
        attempted_locations.append(str(candidate))
        module_file = candidate / "dr_hcpa_v2_2024.py"

        if not module_file.is_file():
            continue

        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

        try:
            return importlib.import_module("dr_hcpa_v2_2024")
        except Exception as exc:
            sys.modules.pop("dr_hcpa_v2_2024", None)
            import_errors.append(f"{candidate}: {exc}")

    searched = "\n".join(f"- {path}" for path in attempted_locations) or "- nenhum caminho candidato"
    error_details = "\n".join(f"- {error}" for error in import_errors)
    message = (
        "Não foi possível carregar o módulo pytorch_opt `dr_hcpa_v2_2024`.\n"
        "Copie `pytorch_opt/` para o mesmo root do projeto em execução ou defina `PYTORCH_OPT_DIR`.\n"
        f"Caminhos pesquisados:\n{searched}"
    )
    if error_details:
        message += f"\nErros de import:\n{error_details}"

    raise RuntimeError(message)


def unpack_batch(batch):
    """
    Normaliza batches vindos de PyTorch DataLoader ou DALI.

    Formatos suportados:
    - tuple/list simples: (images, labels)
    - dict: {"image": ..., "label": ...}
    - list com um único dict, como retornado pelo DALIGenericIterator
    """
    if isinstance(batch, dict):
        return batch["image"], batch["label"]

    if isinstance(batch, (list, tuple)):
        if len(batch) == 1 and isinstance(batch[0], dict):
            return batch[0]["image"], batch[0]["label"]
        if len(batch) == 2 and not isinstance(batch[0], dict):
            return batch
        if len(batch) > 0 and isinstance(batch[0], dict):
            return batch[0]["image"], batch[0]["label"]

    raise TypeError(f"Unsupported batch structure: {type(batch)}")


def get_data_loaders(
    tfrec_dir: str,
    batch_size: int,
    img_size: int,
    dataset_name: str = "all",
    use_dali: bool = True,
    augment: bool = True,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    enable_amp: bool = True,
    model_name: str = "inception_v3",
):
    """
    Carrega dataloaders usando infraestrutura pytorch_opt.

    Esta função tenta importar do pytorch_opt e reutiliza:
    - DALI pipeline (se use_dali=True)
    - PyTorch DataLoader (fallback)
    - Normalização automática baseada no backbone

    Args:
        tfrec_dir: Diretório com TFrecords
        batch_size: Batch size por GPU
        img_size: Tamanho das imagens
        dataset_name: Nome do dataset (ex: "all")
        use_dali: Usar DALI (True) ou PyTorch (False)
        augment: Ativar augmentações
        num_workers: Workers para PyTorch dataloader
        rank: Rank para DDP
        world_size: World size para DDP
        enable_amp: Enable automatic mixed precision

    Returns:
        (train_loader, val_loader, test_loader)
    """
    try:
        pytorch_opt_module = _load_pytorch_opt_module()

        train_files = pytorch_opt_module.list_split_files(tfrec_dir, "train")
        val_files = pytorch_opt_module.list_split_files(tfrec_dir, "val")
        test_files = pytorch_opt_module.list_split_files(tfrec_dir, "test")

        if not train_files:
            raise ValueError(f"Nenhum arquivo train*.tfrec encontrado em {tfrec_dir}")

        # NAO reutilizar test como validacao: o best-checkpoint e selecionado pela
        # val, entao val==test escolheria o modelo no proprio teste (vazamento P1).
        if not test_files:
            test_files = val_files

        if not val_files or not test_files:
            raise ValueError(
                f"Não foi possível resolver os splits de validação (val*) e teste (test*) "
                f"em {tfrec_dir}. Sem split val* proprio, use data/all-tfrec-v2 (vazamento P1)."
            )
        # GUARDA P1: val e test disjuntos.
        if set(val_files) & set(test_files):
            raise ValueError(
                f"val e test compartilham TFRecords em {tfrec_dir}: vazamento P1. "
                "Use data/all-tfrec-v2."
            )

        # Parâmetros DALI/Torch
        device_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
        shuffle_seed = 42

        if use_dali:
            train_loader = pytorch_opt_module.build_dali_loader(
                tfrec_files=train_files,
                batch_size=batch_size,
                image_size=img_size,
                device_id=device_id,
                shard_id=rank,
                num_shards=world_size,
                augment=augment,
                seed=shuffle_seed,
                fundus_crop_ratio=1.0,
                random_shuffle=True,
                model_name=model_name,
            )
            val_loader = pytorch_opt_module.build_dali_loader(
                tfrec_files=val_files,
                batch_size=batch_size,
                image_size=img_size,
                device_id=device_id,
                shard_id=rank,
                num_shards=world_size,
                augment=False,
                seed=shuffle_seed + 999,
                fundus_crop_ratio=1.0,
                random_shuffle=False,
                model_name=model_name,
            )
            test_loader = pytorch_opt_module.build_dali_loader(
                tfrec_files=test_files,
                batch_size=batch_size,
                image_size=img_size,
                device_id=device_id,
                shard_id=rank,
                num_shards=world_size,
                augment=False,
                seed=shuffle_seed + 1999,
                fundus_crop_ratio=1.0,
                random_shuffle=False,
                model_name=model_name,
            )
        else:
            train_loader, _, _, _ = pytorch_opt_module.build_torch_loader(
                tfrec_files=train_files,
                batch_size=batch_size,
                image_size=img_size,
                world_size=world_size,
                rank=rank,
                normalize_mode="preprocess",
                augment=augment,
                fundus_crop_ratio=1.0,
                model_name=model_name,
                use_cuda=torch.cuda.is_available(),
                shuffle=True,
            )
            val_loader, _, _, _ = pytorch_opt_module.build_torch_loader(
                tfrec_files=val_files,
                batch_size=batch_size,
                image_size=img_size,
                world_size=world_size,
                rank=rank,
                normalize_mode="preprocess",
                augment=False,
                fundus_crop_ratio=1.0,
                model_name=model_name,
                use_cuda=torch.cuda.is_available(),
                shuffle=False,
            )
            test_loader, _, _, _ = pytorch_opt_module.build_torch_loader(
                tfrec_files=test_files,
                batch_size=batch_size,
                image_size=img_size,
                world_size=world_size,
                rank=rank,
                normalize_mode="preprocess",
                augment=False,
                fundus_crop_ratio=1.0,
                model_name=model_name,
                use_cuda=torch.cuda.is_available(),
                shuffle=False,
            )

        return train_loader, val_loader, test_loader

    except Exception as e:
        raise RuntimeError(f"Não foi possível carregar dataloaders do pytorch_opt: {e}") from e


def get_backbone(
    model_name: str = "inception_v3",
    img_size: int = 299,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> torch.nn.Module:
    """
    Carrega backbone CNN do pytorch_opt retornando features espaciais.

    Args:
        model_name: Nome do modelo (ex: "inception_v3", "resnet50")
        img_size: Tamanho da imagem
        pretrained: Usar pesos ImageNet
        freeze_backbone: Congelar backbone (treinar apenas head)

    Returns:
        Modelo PyTorch que retorna features [B, C, H, W] em vez de logit
    """
    try:
        pytorch_opt_module = _load_pytorch_opt_module()

        # Usar features_only=True para obter mapa de features, não logit
        backbone = pytorch_opt_module.create_backbone(
            model_name=model_name,
            img_size=img_size,
            features_only=True,  # Retorna features espaciais
        )

        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False

        return backbone

    except Exception as e:
        raise RuntimeError(f"Não foi possível carregar backbone do pytorch_opt: {e}") from e
