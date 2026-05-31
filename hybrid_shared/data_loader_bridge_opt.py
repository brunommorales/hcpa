"""
Opt-specific data loader bridge.

Reusa a infraestrutura do bridge base, mas fixa defaults mais conservadores
para as variantes otimizadas sem alterar as variantes estaveis.
"""

from __future__ import annotations

from hybrid_shared.data_loader_bridge import get_backbone, get_data_loaders as get_data_loaders_base, unpack_batch


def get_data_loaders(
    tfrec_dir: str,
    batch_size: int,
    img_size: int,
    dataset_name: str = "all",
    use_dali: bool = False,
    augment: bool = True,
    num_workers: int = 4,
    rank: int = 0,
    world_size: int = 1,
    enable_amp: bool = True,
    model_name: str = "inception_v3",
):
    """
    Wrapper do bridge base com defaults pensados para as variantes opt.
    """
    return get_data_loaders_base(
        tfrec_dir=tfrec_dir,
        batch_size=batch_size,
        img_size=img_size,
        dataset_name=dataset_name,
        use_dali=use_dali,
        augment=augment,
        num_workers=num_workers,
        rank=rank,
        world_size=world_size,
        enable_amp=enable_amp,
        model_name=model_name,
    )


__all__ = ["get_backbone", "get_data_loaders", "unpack_batch"]
