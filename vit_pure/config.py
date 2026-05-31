"""CLI configuration for Pure ViT."""

from __future__ import annotations

import argparse


FIXED_TFREC_DIR = "/home/users/bmmorales/projects/hcpa/data/all-tfrec"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento de Vision Transformer puro (patches crus da imagem)"
    )

    parser.add_argument(
        "--tfrec_dir",
        type=str,
        default=FIXED_TFREC_DIR,
        help="Diretório com TFRecords",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/users/bmmorales/projects/hcpa/vit_pure/results",
        help="Diretório para resultados",
    )
    parser.add_argument(
        "--exec_id",
        type=int,
        default=0,
        help="ID da execução",
    )

    parser.add_argument("--dataset", type=str, default="all", help="Nome do dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size por GPU")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Número de passos de accumulation de gradientes (útil para simular batches maiores)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=320,
        help="Tamanho da imagem; deve ser divisível por patch_size",
    )
    parser.add_argument("--num_workers", type=int, default=4, help="Workers do DataLoader")

    parser.add_argument("--patch_size", type=int, default=16, help="Tamanho do patch")
    parser.add_argument("--embed_dim", type=int, default=768, help="Dimensão do embedding")
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=12,
        help="Número de camadas Transformer (ViT-B/16 usa 12)",
    )
    parser.add_argument("--num_heads", type=int, default=12, help="Número de heads (ViT-B/16 usa 12)")
    parser.add_argument(
        "--mlp_ratio",
        type=float,
        default=4.0,
        help="Multiplicador do bloco feed-forward",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout do Transformer e embeddings",
    )
    parser.add_argument(
        "--use_cls_token",
        dest="use_cls_token",
        action="store_true",
        help="Usar CLS token para agregação",
    )
    parser.add_argument(
        "--no_cls_token",
        dest="use_cls_token",
        action="store_false",
        help="Usar mean pooling em vez de CLS token",
    )
    parser.set_defaults(use_cls_token=True)

    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas")
    parser.add_argument("--lrate", type=float, default=1e-4, help="Learning rate inicial")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "rmsprop"],
        help="Otimizador",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Warmup epochs")

    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Peso da classe positiva")
    parser.add_argument(
        "--auto_pos_weight",
        dest="auto_pos_weight",
        action="store_true",
        help="Estimar automaticamente neg/pos no treino quando pos_weight <= 1",
    )
    parser.add_argument(
        "--disable_auto_pos_weight",
        dest="auto_pos_weight",
        action="store_false",
        help="Desativar estimação automática de pos_weight",
    )
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Focal loss gamma")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping")

    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Ativar augmentações",
    )
    parser.add_argument("--mixup_alpha", type=float, default=0.0, help="Reservado")
    parser.add_argument("--cutmix_alpha", type=float, default=0.0, help="Reservado")

    parser.add_argument(
        "--enable_amp",
        dest="enable_amp",
        action="store_true",
        help="Ativar AMP",
    )
    parser.add_argument(
        "--disable_amp",
        dest="enable_amp",
        action="store_false",
        help="Desativar AMP",
    )
    parser.add_argument(
        "--enable_ema",
        dest="enable_ema",
        action="store_true",
        help="Ativar EMA",
    )
    parser.add_argument(
        "--disable_ema",
        dest="enable_ema",
        action="store_false",
        help="Desativar EMA",
    )
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
    parser.add_argument(
        "--enable_dali",
        dest="enable_dali",
        action="store_true",
        help="Usar pipeline DALI",
    )
    parser.add_argument(
        "--disable_dali",
        dest="enable_dali",
        action="store_false",
        help="Usar DataLoader PyTorch sem DALI",
    )
    parser.add_argument(
        "--enable_cosine",
        dest="enable_cosine",
        action="store_true",
        help="Ativar warmup + cosine scheduler",
    )
    parser.add_argument(
        "--disable_cosine",
        dest="enable_cosine",
        action="store_false",
        help="Desativar warmup + cosine scheduler",
    )
    parser.add_argument(
        "--enable_performance_profiler",
        dest="enable_performance_profiler",
        action="store_true",
        help="Ativar profiler de batch/epoca",
    )
    parser.add_argument(
        "--disable_performance_profiler",
        dest="enable_performance_profiler",
        action="store_false",
        help="Desativar profiler de batch/epoca",
    )
    parser.add_argument(
        "--enable_flash_attention",
        dest="enable_flash_attention",
        action="store_true",
        help="Ativar Flash Attention para otimizar memória (requer hardware compatível)",
    )
    parser.add_argument(
        "--disable_flash_attention",
        dest="enable_flash_attention",
        action="store_false",
        help="Desativar Flash Attention",
    )
    parser.add_argument(
        "--collect_patch_analysis",
        dest="collect_patch_analysis",
        action="store_true",
        help="Coletar análise de patches/atenções durante evaluate.py",
    )
    parser.add_argument(
        "--disable_patch_analysis",
        dest="collect_patch_analysis",
        action="store_false",
        help="Desativar análise de patches/atenções durante evaluate.py",
    )
    parser.set_defaults(
        enable_amp=True,
        enable_ema=False,
        enable_dali=False,
        enable_cosine=True,
        enable_performance_profiler=False,
        enable_flash_attention=False,
        collect_patch_analysis=False,
        auto_pos_weight=True,
    )

    parser.add_argument("--local_rank", type=int, default=0, help="Local rank")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Reduzir logs por batch",
    )
    parser.set_defaults(verbose=False)

    return parser.parse_args()
