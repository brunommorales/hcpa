"""
Configuração de argumentos para Hybrid Token Reduction.
"""

import argparse


FIXED_TFREC_DIR = "/home/users/bmmorales/projects/hcpa/data/all-tfrec"


def get_args() -> argparse.Namespace:
    """
    Parse de argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Treinamento de Modelo Híbrido com Redução de Tokens"
    )

    # ========== Paths ==========
    parser.add_argument(
        "--tfrec_dir",
        type=str,
        default=FIXED_TFREC_DIR,
        help="Diretório com TFRecords",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/home/users/bmmorales/projects/hcpa/hybrid_token_reduction/results",
        help="Diretório para salvar resultados",
    )
    parser.add_argument(
        "--exec_id",
        type=int,
        default=0,
        help="ID da execução",
    )

    # ========== Dataset ==========
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Nome do dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=96,
        help="Batch size por GPU",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=299,
        help="Tamanho da imagem",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers",
    )

    # ========== Model ==========
    parser.add_argument(
        "--backbone",
        type=str,
        default="inception_v3",
        choices=["inception_v3", "resnet50", "efficientnet_b0", "efficientnet_b4"],
        help="Backbone CNN",
    )
    parser.add_argument(
        "--backbone_dim",
        type=int,
        default=2048,
        help="Dimensão do backbone (canais de saída da CNN)",
    )
    parser.add_argument(
        "--transformer_dim",
        type=int,
        default=256,
        help="Dimensão de trabalho do Transformer. A projeção comprime backbone_dim -> "
             "transformer_dim antes da atenção, reduzindo o custo por "
             "(backbone_dim/transformer_dim)^2. Use --transformer_dim 2048 para o modelo antigo.",
    )
    parser.add_argument(
        "--num_transformer_layers",
        type=int,
        default=4,
        help="Número de camadas Transformer",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Número de heads",
    )
    parser.add_argument(
        "--use_cls_token",
        action="store_true",
        help="Usar CLS token",
    )

    # ========== Token Reduction (CHAVE) ==========
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=0.5,
        help="Proporção de tokens a manter (0.0-1.0). "
             "Keep_ratio=0.5 → redução de 4x em complexidade Transformer (O(n²)/O(k²))",
    )
    parser.add_argument(
        "--keep_k",
        type=int,
        default=None,
        help="Número absoluto de tokens a manter (sobrescreve keep_ratio se fornecido)",
    )

    # ========== Training ==========
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Número de épocas",
    )
    parser.add_argument(
        "--lrate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "adam", "sgd", "rmsprop"],
        help="Otimizador",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=3,
        help="Épocas com backbone congelado",
    )
    parser.add_argument(
        "--fine_tune_lr_factor",
        type=float,
        default=0.1,
        help="Fator LR no fine-tuning",
    )

    # ========== Loss ==========
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Peso classe positiva",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma",
    )

    # ========== Regularization ==========
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping",
    )
    parser.add_argument(
        "--transformer_dropout",
        type=float,
        default=0.1,
        help="Transformer dropout",
    )

    # ========== Augmentation ==========
    parser.add_argument(
        "--augment",
        action="store_true",
        default=True,
        help="Ativar augmentações",
    )
    parser.add_argument(
        "--mixup_alpha",
        type=float,
        default=0.0,
        help="Mixup alpha",
    )
    parser.add_argument(
        "--cutmix_alpha",
        type=float,
        default=0.0,
        help="CutMix alpha",
    )

    # ========== Optimization ==========
    parser.add_argument(
        "--enable_amp",
        dest="enable_amp",
        action="store_true",
        help="Enable AMP",
    )
    parser.add_argument(
        "--disable_amp",
        dest="enable_amp",
        action="store_false",
        help="Disable AMP",
    )
    parser.set_defaults(enable_amp=False)   # variante isola SO' token reduction
    parser.add_argument(
        "--enable_ema",
        dest="enable_ema",
        action="store_true",
        help="Enable EMA",
    )
    parser.add_argument(
        "--disable_ema",
        dest="enable_ema",
        action="store_false",
        help="Disable EMA",
    )
    parser.set_defaults(enable_ema=False)
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay",
    )
    parser.add_argument(
        "--enable_dali",
        dest="enable_dali",
        action="store_true",
        help="Enable DALI",
    )
    parser.add_argument(
        "--disable_dali",
        dest="enable_dali",
        action="store_false",
        help="Disable DALI",
    )
    parser.set_defaults(enable_dali=False)
    parser.add_argument(
        "--enable_cosine",
        dest="enable_cosine",
        action="store_true",
        help="Enable warmup + cosine LR scheduler",
    )
    parser.add_argument(
        "--disable_cosine",
        dest="enable_cosine",
        action="store_false",
        help="Disable warmup + cosine LR scheduler",
    )
    parser.set_defaults(enable_cosine=False)  # sem otimizacao alem do token reduction
    parser.add_argument(
        "--enable_performance_profiler",
        dest="enable_performance_profiler",
        action="store_true",
        help="Enable batch/epoch performance profiler",
    )
    parser.add_argument(
        "--disable_performance_profiler",
        dest="enable_performance_profiler",
        action="store_false",
        help="Disable batch/epoch performance profiler",
    )
    parser.set_defaults(enable_performance_profiler=False)

    # ========== Distributed ==========
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank",
    )

    # ========== Other ==========
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
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
        help="Reduce batch-level logging",
    )
    parser.set_defaults(verbose=False)

    return parser.parse_args()
