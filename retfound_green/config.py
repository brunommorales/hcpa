"""CLI configuration for RETFound-Green fine-tuning."""

from __future__ import annotations

import argparse


FIXED_TFREC_DIR = "/home/users/bmmorales/projects/hcpa/data/all-tfrec"
DEFAULT_RESULTS_DIR = "/home/users/bmmorales/projects/hcpa/retfound_green/results"
DEFAULT_BACKBONE_CHECKPOINT = (
    "/home/users/bmmorales/projects/hcpa/retfound_green/weights/retfoundgreen_statedict.pth"
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tuning do backbone RETFound-Green para classificacao binaria"
    )

    parser.add_argument(
        "--tfrec_dir",
        type=str,
        default=FIXED_TFREC_DIR,
        help="Diretorio com TFRecords",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Diretorio para resultados",
    )
    parser.add_argument("--exec_id", type=int, default=0, help="ID da execucao")

    parser.add_argument("--dataset", type=str, default="all", help="Nome do dataset")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size por GPU")
    parser.add_argument("--img_size", type=int, default=392, help="Tamanho de entrada da imagem")
    parser.add_argument("--num_workers", type=int, default=4, help="Workers do DataLoader")
    parser.add_argument(
        "--backbone_model",
        type=str,
        default="vit_small_patch14_reg4_dinov2",
        help="Nome do backbone timm",
    )
    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default=DEFAULT_BACKBONE_CHECKPOINT,
        help="Caminho para os pesos RETFound-Green",
    )
    parser.add_argument("--head_dropout", type=float, default=0.0, help="Dropout antes da head")
    parser.add_argument(
        "--freeze_backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Congelar backbone e treinar apenas a head",
    )
    parser.add_argument(
        "--unfreeze_backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Treinar backbone e head",
    )
    parser.set_defaults(freeze_backbone=False)

    parser.add_argument("--epochs", type=int, default=200, help="Numero de epocas")
    parser.add_argument("--lrate", type=float, default=5e-5, help="Learning rate inicial")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "rmsprop"],
        help="Otimizador",
    )
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
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
        help="Desativar estimacao automatica de pos_weight",
    )
    parser.add_argument("--focal_gamma", type=float, default=0.0, help="Focal loss gamma")
    parser.add_argument("--clip_grad_norm", type=float, default=1.0, help="Gradient clipping")

    parser.add_argument("--augment", action="store_true", default=True, help="Ativar augmentacoes")
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
    parser.set_defaults(
        enable_amp=True,
        enable_ema=False,
        enable_dali=False,
        enable_cosine=True,
        enable_performance_profiler=False,
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
