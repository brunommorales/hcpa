"""
derived_space.py — Espaço de Configuração Derivado.

Construído EXCLUSIVAMENTE a partir da auditoria das 6 variantes.
Cada parâmetro ajustável é derivado do que EXISTE nas variantes base/opt.
Nenhum hiperparâmetro é inventado — somente os que aparecem nos argparse,
dataclasses ou defaults das variantes fornecidas.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union

# Tipos para limites
NumericRange = Tuple[float, float]  # (min, max)
CategoricalValues = List[Any]


@dataclasses.dataclass
class ParamSpec:
    """Especificação de um parâmetro ajustável do espaço derivado."""
    name: str
    description: str
    param_type: str  # "int", "float", "bool", "categorical"
    source_variants: List[str]  # quais variantes expõem este parâmetro
    base_value: Any  # valor na variante base
    opt_value: Any  # valor na variante opt
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    choices: Optional[List[Any]] = None
    tunable_online: bool = True  # se pode ser ajustado durante execução
    requires_restart: bool = False  # se precisa recriar loader/modelo

    @property
    def range(self) -> Optional[NumericRange]:
        if self.range_min is not None and self.range_max is not None:
            return (self.range_min, self.range_max)
        return None


@dataclasses.dataclass
class DerivedConfigSpace:
    """Espaço de configuração derivado — contém SOMENTE parâmetros encontrados nas variantes."""
    stack: str  # "pytorch", "tensorflow", "monai"
    params: Dict[str, ParamSpec] = dataclasses.field(default_factory=dict)

    def add(self, spec: ParamSpec):
        self.params[spec.name] = spec

    def get(self, name: str) -> Optional[ParamSpec]:
        return self.params.get(name)

    def tunable_params(self) -> Dict[str, ParamSpec]:
        return {k: v for k, v in self.params.items() if v.tunable_online}

    def summary(self) -> str:
        lines = [f"=== Espaço Derivado: {self.stack} ({len(self.params)} parâmetros) ==="]
        for name, spec in sorted(self.params.items()):
            flag = "[online]" if spec.tunable_online else "[static]"
            if spec.param_type in ("int", "float") and spec.range is not None:
                rng = f"[{spec.range_min}, {spec.range_max}]"
            elif spec.choices:
                rng = str(spec.choices)
            else:
                rng = "—"
            lines.append(
                f"  {flag} {name} ({spec.param_type}): base={spec.base_value} opt={spec.opt_value} range={rng}"
            )
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# Funções que constroem o espaço derivado para cada stack
# TODOS OS VALORES foram extraídos da auditoria das 6 variantes.
# ═══════════════════════════════════════════════════════════════

def _build_pytorch_space() -> DerivedConfigSpace:
    """Derivado de pytorch_base/dr_hcpa_v2_2024.py e pytorch_opt/dr_hcpa_v2_2024.py."""
    space = DerivedConfigSpace(stack="pytorch")

    # --- Parâmetros que diferem entre base e opt ---
    space.add(ParamSpec(
        name="batch_size", description="Batch por GPU",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=32, opt_value=96,
        range_min=8, range_max=256,
        tunable_online=True, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="epochs", description="Épocas totais (FIXO: 200)",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=200, opt_value=200,
        range_min=200, range_max=200,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="lrate", description="Learning rate inicial",
        param_type="float", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=5e-4, opt_value=5e-4,
        range_min=1e-6, range_max=1e-2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="freeze_epochs", description="Épocas com backbone congelado",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=3, opt_value=3,
        range_min=0, range_max=20,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="fine_tune_lr_factor", description="Fator de LR no fine-tune",
        param_type="float", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=0.01, opt_value=0.1,
        range_min=0.001, range_max=1.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="fine_tune_lr", description="LR absoluto de fine-tune",
        param_type="float", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=-1.0, opt_value=5e-4,
        range_min=1e-6, range_max=1e-2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="warmup_epochs", description="Épocas de warmup do scheduler",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=3, opt_value=5,
        range_min=0, range_max=20,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="min_lr", description="LR mínimo para cosine annealing",
        param_type="float", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=1e-6, opt_value=1e-6,
        range_min=0.0, range_max=1e-4,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="clip_grad_norm", description="Clipping de norma de gradiente",
        param_type="float", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=1.0, opt_value=1.0,
        range_min=0.0, range_max=10.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="label_smoothing", description="Label smoothing",
        param_type="float", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=0.0, opt_value=0.0,
        range_min=0.0, range_max=0.2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="model", description="Backbone do modelo",
        param_type="categorical", source_variants=["pytorch_base", "pytorch_opt"],
        base_value="InceptionV3", opt_value="inception_v3",
        choices=["InceptionV3", "inception_v3", "ResNet50", "resnet50",
                 "EfficientNetB0", "efficientnet_b0"],
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="normalize", description="Normalização de imagem",
        param_type="categorical", source_variants=["pytorch_base", "pytorch_opt"],
        base_value="preprocess", opt_value="preprocess",
        choices=["preprocess", "raw255", "unit"],
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="augment", description="Habilita augmentações",
        param_type="bool", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=True, opt_value=True,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="img_sizes", description="Tamanho da imagem (quadrada)",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=299, opt_value=299,
        range_min=224, range_max=512,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="seed", description="Seed de reprodutibilidade",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=42, opt_value=42,
        range_min=0, range_max=2**31,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="cores", description="OMP_NUM_THREADS",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=0, opt_value=0,
        range_min=0, range_max=128,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="verbose", description="Verbosidade",
        param_type="int", source_variants=["pytorch_base", "pytorch_opt"],
        base_value=1, opt_value=1,
        range_min=0, range_max=2,
        tunable_online=False,
    ))

    # --- Parâmetros que só existem no opt (entram como opções) ---
    space.add(ParamSpec(
        name="mixup_alpha", description="Alpha para mixup (<=0 desativa)",
        param_type="float", source_variants=["pytorch_opt"],
        base_value=0.0, opt_value=0.0,
        range_min=0.0, range_max=1.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="cutmix_alpha", description="Alpha para cutmix (<=0 desativa)",
        param_type="float", source_variants=["pytorch_opt"],
        base_value=0.0, opt_value=0.0,
        range_min=0.0, range_max=1.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="focal_gamma", description="Gamma da focal loss (0 desativa)",
        param_type="float", source_variants=["pytorch_opt"],
        base_value=0.0, opt_value=0.0,
        range_min=0.0, range_max=5.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="pos_weight", description="Peso da classe positiva",
        param_type="float", source_variants=["pytorch_opt"],
        base_value=1.0, opt_value=1.0,
        range_min=0.1, range_max=10.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="tta_views", description="Vistas TTA na avaliação final",
        param_type="int", source_variants=["pytorch_opt"],
        base_value=1, opt_value=1,
        range_min=1, range_max=8,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="fundus_crop_ratio", description="Crop central antes do resize",
        param_type="float", source_variants=["pytorch_opt"],
        base_value=1.0, opt_value=0.9,
        range_min=0.5, range_max=1.0,
        tunable_online=False, requires_restart=True,
    ))

    return space


def _build_tensorflow_space() -> DerivedConfigSpace:
    """Derivado de tensorflow_base/dr_hcpa_v2_2024.py e tensorflow_opt/dr_hcpa_v2_2024.py."""
    space = DerivedConfigSpace(stack="tensorflow")

    space.add(ParamSpec(
        name="batch_size", description="Batch por réplica/GPU",
        param_type="int", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=32, opt_value=96,
        range_min=8, range_max=256,
        tunable_online=True, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="epochs", description="Épocas totais (FIXO: 200)",
        param_type="int", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=200, opt_value=200,
        range_min=200, range_max=200,
        tunable_online=False,
    ))
    # O LR opt padrão (3e-3) mostrou risco de NaN na RTX 4070;
    # reduzimos opt_value e range_max para limitar saltos agressivos.
    space.add(ParamSpec(
        name="lrate", description="Learning rate",
        param_type="float", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=1e-3, opt_value=1.5e-3,
        range_min=5e-5, range_max=3e-3,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="warmup_epochs", description="Épocas de warmup",
        param_type="int", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=3, opt_value=4,
        range_min=0, range_max=10,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="min_lr", description="LR mínimo do scheduler",
        param_type="float", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=1e-6, opt_value=1e-5,
        range_min=0.0, range_max=5e-5,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="label_smoothing", description="Label smoothing",
        param_type="float", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=0.0, opt_value=0.1,
        range_min=0.0, range_max=0.2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="model", description="Backbone keras.applications",
        param_type="categorical", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value="InceptionV3", opt_value="InceptionV3",
        choices=["InceptionV3", "Xception", "ResNet50", "EfficientNetB0", "DenseNet121",
                 "MobileNetV2", "ConvNeXtTiny"],
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="normalize", description="Normalização de imagem",
        param_type="categorical", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value="preprocess", opt_value="preprocess",
        choices=["preprocess", "raw255", "unit"],
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="augment", description="Habilita augmentações",
        param_type="bool", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=True, opt_value=True,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="img_sizes", description="Tamanho da imagem (quadrada)",
        param_type="int", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=299, opt_value=299,
        range_min=224, range_max=512,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="seed", description="Seed para reprodutibilidade",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=42, opt_value=2024,
        range_min=0, range_max=2**31,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="cores", description="CPU threads (OMP_NUM_THREADS)",
        param_type="int", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=0, opt_value=0,
        range_min=0, range_max=128,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="verbose", description="Verbosidade do Keras",
        param_type="int", source_variants=["tensorflow_base", "tensorflow_opt"],
        base_value=1, opt_value=1,
        range_min=0, range_max=2,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="clip_grad_norm", description="Gradient clipping norm",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=0.0, opt_value=1.0,
        range_min=0.0, range_max=10.0,
        tunable_online=True,
    ))

    # --- Parâmetros que só existem no tensorflow_opt ---
    space.add(ParamSpec(
        name="num_classes", description="Número de classes",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=2, opt_value=2,
        range_min=2, range_max=10,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="freeze_epochs", description="Épocas com backbone congelado",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=0, opt_value=1,
        range_min=0, range_max=20,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="fine_tune_lr_factor", description="Multiplicador de LR no fine-tune",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=1.0, opt_value=0.1,
        range_min=0.001, range_max=1.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="fine_tune_at", description="Layer index para fine-tune",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=0, opt_value=-200,
        range_min=-500, range_max=0,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="fine_tune_lr", description="LR abs de fine-tune",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=0.0, opt_value=2e-4,
        range_min=1e-6, range_max=1e-2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="scheduler", description="Estratégia de LR scheduler",
        param_type="categorical", source_variants=["tensorflow_opt"],
        base_value="cosine", opt_value="cosine",
        choices=["cosine", "plateau"],
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="mixup_alpha", description="Alpha para mixup",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=0.0, opt_value=0.4,
        range_min=0.0, range_max=0.6,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="cutmix_alpha", description="Alpha para cutmix",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=0.0, opt_value=0.6,
        range_min=0.0, range_max=0.8,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="focal_gamma", description="Gamma focal loss",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=0.0, opt_value=2.0,
        range_min=0.0, range_max=3.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="pos_weight", description="Peso da classe positiva",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=1.0, opt_value=2.0,
        range_min=0.1, range_max=3.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="fundus_crop_ratio", description="Crop central",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=1.0, opt_value=0.9,
        range_min=0.5, range_max=1.0,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="freeze_bn", description="Congela BatchNorm no fine-tune",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=False, opt_value=True,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="mixed_precision", description="Mixed precision fp16",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=False, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="use_dali", description="Usar NVIDIA DALI",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=False, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="dali_threads", description="Threads do DALI pipeline",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=4, opt_value=4,
        range_min=1, range_max=16,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="channels_last", description="Data format channels last",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=True, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="h2d_uint8", description="Manter uint8 até normalização",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=False, opt_value=True,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="tta_views", description="Vistas TTA na avaliação",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=1, opt_value=2,
        range_min=1, range_max=4,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="wait_epochs", description="Épocas de espera",
        param_type="int", source_variants=["tensorflow_opt"],
        base_value=30, opt_value=30,
        range_min=5, range_max=100,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="recompute_backbone", description="tf.recompute_grad no backbone",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=False, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="jit_compile", description="JIT compile Keras",
        param_type="bool", source_variants=["tensorflow_opt"],
        base_value=False, opt_value=False,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="auc_target", description="AUC alvo para registro de tempo",
        param_type="float", source_variants=["tensorflow_opt"],
        base_value=0.95, opt_value=0.95,
        range_min=0.5, range_max=1.0,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="cache_dir", description="Diretório de cache de TF datasets",
        param_type="categorical", source_variants=["tensorflow_opt"],
        base_value="none", opt_value="none",
        choices=["none"],
        tunable_online=False,
    ))

    return space


def _build_monai_space() -> DerivedConfigSpace:
    """Derivado de monai_base e monai_opt (TrainConfig dataclass + train.py argparse)."""
    space = DerivedConfigSpace(stack="monai")

    space.add(ParamSpec(
        name="batch_size", description="Batch de treino",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=96, opt_value=96,
        range_min=8, range_max=256,
        tunable_online=True, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="epochs", description="Épocas totais (FIXO: 200)",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=200, opt_value=200,
        range_min=200, range_max=200,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="learning_rate", description="Learning rate",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=3e-4, opt_value=3e-4,
        range_min=1e-6, range_max=1e-2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="min_lr", description="LR mínimo do scheduler",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.0, opt_value=3e-5,
        range_min=0.0, range_max=1e-3,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="warmup_epochs", description="Épocas de warmup",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=0, opt_value=10,
        range_min=0, range_max=30,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="weight_decay", description="Weight decay do otimizador",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=1e-4, opt_value=1e-4,
        range_min=0.0, range_max=1e-2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="scheduler", description="Tipo de LR scheduler",
        param_type="categorical", source_variants=["monai_base", "monai_opt"],
        base_value="none", opt_value="cosine",
        choices=["none", "cosine", "onecycle"],
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="grad_clip_norm", description="Gradient clipping norm",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=1.0, opt_value=1.0,
        range_min=0.0, range_max=10.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="mixup_alpha", description="Alpha para mixup",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.0, opt_value=0.2,
        range_min=0.0, range_max=1.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="cutmix_alpha", description="Alpha para cutmix",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.0, opt_value=0.0,
        range_min=0.0, range_max=1.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="label_smoothing", description="Label smoothing",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.0, opt_value=0.02,
        range_min=0.0, range_max=0.2,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="augment", description="Habilita augmentações",
        param_type="bool", source_variants=["monai_base", "monai_opt"],
        base_value=True, opt_value=True,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="model_name", description="Backbone do modelo (timm ou torchvision)",
        param_type="categorical", source_variants=["monai_base", "monai_opt"],
        base_value="inception_v3", opt_value="inception_v3",
        choices=["inception_v3"],
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="normalize", description="Normalização de imagem",
        param_type="categorical", source_variants=["monai_base", "monai_opt"],
        base_value="inception", opt_value="inception",
        choices=["inception", "imagenet", "none"],
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="image_size", description="Tamanho de imagem",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=299, opt_value=299,
        range_min=224, range_max=512,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="fundus_crop_ratio", description="Crop central",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.9, opt_value=0.9,
        range_min=0.5, range_max=1.0,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="channels_last", description="Memory format channels last",
        param_type="bool", source_variants=["monai_base", "monai_opt"],
        base_value=True, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="amp", description="Automatic Mixed Precision",
        param_type="bool", source_variants=["monai_base", "monai_opt"],
        base_value=True, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="compile", description="torch.compile habilitado",
        param_type="bool", source_variants=["monai_base", "monai_opt"],
        base_value=False, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="use_dali", description="Usar NVIDIA DALI",
        param_type="bool", source_variants=["monai_base", "monai_opt"],
        base_value=False, opt_value=True,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="ema_decay", description="EMA decay (0 desativa)",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.0, opt_value=0.999,
        range_min=0.0, range_max=0.9999,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="ema_on_cpu", description="EMA em CPU para economizar VRAM",
        param_type="bool", source_variants=["monai_base", "monai_opt"],
        base_value=False, opt_value=False,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="gradient_accumulation", description="Passos de acumulação de gradiente",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=1, opt_value=1,
        range_min=1, range_max=16,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="pos_weight", description="Peso da classe positiva",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=1.0, opt_value=1.0,
        range_min=0.1, range_max=10.0,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="dropout", description="Dropout rate",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.2, opt_value=0.2,
        range_min=0.0, range_max=0.5,
        tunable_online=False, requires_restart=True,
    ))
    space.add(ParamSpec(
        name="num_workers", description="DataLoader num_workers",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=8, opt_value=8,
        range_min=0, range_max=32,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="seed", description="Seed de reprodutibilidade",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=2026, opt_value=2026,
        range_min=0, range_max=2**31,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="color_jitter", description="Intensidade de color jitter",
        param_type="float", source_variants=["monai_base", "monai_opt"],
        base_value=0.1, opt_value=0.1,
        range_min=0.0, range_max=0.5,
        tunable_online=True,
    ))
    space.add(ParamSpec(
        name="tta_views", description="Vistas TTA na avaliação",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=1, opt_value=1,
        range_min=1, range_max=8,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="patience", description="Paciência para early stop (0 desativa)",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=0, opt_value=0,
        range_min=0, range_max=50,
        tunable_online=False,
    ))
    space.add(ParamSpec(
        name="log_every", description="Frequência de log em steps",
        param_type="int", source_variants=["monai_base", "monai_opt"],
        base_value=50, opt_value=25,
        range_min=1, range_max=200,
        tunable_online=False,
    ))

    return space


def build_derived_space(stack: str) -> DerivedConfigSpace:
    """Constrói o espaço de configuração derivado para um stack."""
    builders = {
        "pytorch": _build_pytorch_space,
        "tensorflow": _build_tensorflow_space,
        "monai": _build_monai_space,
    }
    if stack not in builders:
        raise ValueError(f"Stack desconhecido: {stack}. Opções: {list(builders.keys())}")
    return builders[stack]()


def get_initial_config(space: DerivedConfigSpace, mode: str) -> Dict[str, Any]:
    """Retorna config inicial derivada do modo (base ou opt)."""
    config: Dict[str, Any] = {}
    for name, spec in space.params.items():
        if mode == "base":
            config[name] = spec.base_value
        elif mode == "opt":
            config[name] = spec.opt_value
        else:
            raise ValueError(f"Modo desconhecido: {mode}")
    return config
