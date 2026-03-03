"""
audit.py — ETAPA 1: Auditoria estruturada das 6 variantes.

Extrai entradas configuráveis, compara base vs opt, e gera relatório.
Inclui sumário de desempenho histórico a partir da base de conhecimento offline.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple


@dataclasses.dataclass
class VariantEntry:
    """Uma entrada configurável encontrada em uma variante."""
    name: str
    value: Any
    source: str  # "argparse", "dataclass", "env", "code"
    description: str = ""


@dataclasses.dataclass
class VariantAudit:
    """Resultado da auditoria de uma variante."""
    variant_name: str
    entries: List[VariantEntry]

    def as_dict(self) -> Dict[str, Any]:
        return {e.name: e.value for e in self.entries}


@dataclasses.dataclass
class StackDiff:
    """Diferenças entre base e opt dentro de um stack."""
    stack: str
    only_in_base: List[str]
    only_in_opt: List[str]
    changed: Dict[str, Tuple[Any, Any]]  # name -> (base_value, opt_value)
    same: Dict[str, Any]  # name -> shared value


def _pytorch_base_entries() -> List[VariantEntry]:
    """Entradas extraídas de pytorch_base/dr_hcpa_v2_2024.py parse_args()."""
    return [
        VariantEntry("tfrec_dir", "./data/all", "argparse", "Diretório com TFRecords"),
        VariantEntry("dataset", "all", "argparse", "Nome lógico do dataset"),
        VariantEntry("results", "./results/all", "argparse", "Diretório de saída"),
        VariantEntry("exec", 0, "argparse", "ID da execução"),
        VariantEntry("img_sizes", 299, "argparse", "Tamanho das imagens"),
        VariantEntry("batch_size", 32, "argparse", "Batch por GPU"),
        VariantEntry("epochs", 50, "argparse", "Épocas totais"),
        VariantEntry("lrate", 5e-4, "argparse", "Learning rate"),
        VariantEntry("num_thresholds", 200, "argparse", "Thresholds para ROC"),
        VariantEntry("verbose", 1, "argparse", "Verbosidade"),
        VariantEntry("model", "InceptionV3", "argparse", "Backbone keras.applications"),
        VariantEntry("normalize", "preprocess", "argparse", "Normalização: preprocess|raw255|unit"),
        VariantEntry("augment", True, "argparse", "Augmentações habilitadas"),
        VariantEntry("cores", 0, "argparse", "OMP_NUM_THREADS"),
        VariantEntry("seed", 42, "argparse", "Seed base"),
        VariantEntry("clip_grad_norm", 1.0, "argparse", "Clipping de gradiente"),
        VariantEntry("freeze_epochs", 3, "argparse", "Épocas com backbone congelado"),
        VariantEntry("fine_tune_lr_factor", 0.01, "argparse", "Fator de LR no fine-tune"),
        VariantEntry("fine_tune_lr", -1.0, "argparse", "LR absoluto de fine-tune"),
        VariantEntry("warmup_epochs", 3, "argparse", "Épocas de warmup"),
        VariantEntry("min_lr", 1e-6, "argparse", "LR mínimo cosine annealing"),
        VariantEntry("label_smoothing", 0.0, "argparse", "Label smoothing"),
        # Code-level knobs
        VariantEntry("cudnn_benchmark", True, "code", "torch.backends.cudnn.benchmark"),
        VariantEntry("tf32_matmul", True, "code", "torch.backends.cuda.matmul.allow_tf32"),
        VariantEntry("tf32_cudnn", True, "code", "torch.backends.cudnn.allow_tf32"),
        VariantEntry("optimizer", "Adam", "code", "Otimizador usado"),
        VariantEntry("amp_enabled", True, "code", "AMP autocast habilitado"),
        VariantEntry("gpu_augmentation", True, "code", "Augmentação na GPU (torchvision v2)"),
        VariantEntry("dataloader_num_workers", "auto", "code", "max(4,cores) ou min(8,cpu_count)"),
        VariantEntry("persistent_workers", True, "code", "DataLoader persistent_workers"),
        VariantEntry("prefetch_factor", 4, "code", "DataLoader prefetch_factor"),
        VariantEntry("drop_last", True, "code", "DataLoader drop_last"),
    ]


def _pytorch_opt_entries() -> List[VariantEntry]:
    """Entradas extraídas de pytorch_opt/dr_hcpa_v2_2024.py parse_args()."""
    return [
        VariantEntry("tfrec_dir", "./data/all-tfrec", "argparse", "Diretório com TFRecords"),
        VariantEntry("dataset", "all", "argparse", "Nome lógico do dataset"),
        VariantEntry("results", "./results/all", "argparse", "Diretório de saída"),
        VariantEntry("exec", 0, "argparse", "ID da execução"),
        VariantEntry("img_sizes", 299, "argparse", "Tamanho das imagens"),
        VariantEntry("batch_size", 96, "argparse", "Batch por GPU"),
        VariantEntry("epochs", 200, "argparse", "Épocas totais"),
        VariantEntry("lrate", 5e-4, "argparse", "Learning rate"),
        VariantEntry("verbose", 1, "argparse", "Verbosidade"),
        VariantEntry("model", "inception_v3", "argparse", "Backbone timm"),
        VariantEntry("normalize", "preprocess", "argparse", "Normalização"),
        VariantEntry("augment", True, "argparse", "Augmentações habilitadas"),
        VariantEntry("cores", 0, "argparse", "OMP_NUM_THREADS"),
        VariantEntry("seed", 42, "argparse", "Seed base"),
        VariantEntry("clip_grad_norm", 1.0, "argparse", "Clipping de gradiente"),
        VariantEntry("freeze_epochs", 3, "argparse", "Épocas congeladas"),
        VariantEntry("fine_tune_lr_factor", 0.1, "argparse", "Fator LR fine-tune"),
        VariantEntry("fine_tune_lr", 5e-4, "argparse", "LR absoluto fine-tune"),
        VariantEntry("warmup_epochs", 5, "argparse", "Épocas warmup"),
        VariantEntry("min_lr", 1e-6, "argparse", "LR mínimo"),
        VariantEntry("mixup_alpha", 0.0, "argparse", "Alpha mixup"),
        VariantEntry("cutmix_alpha", 0.0, "argparse", "Alpha cutmix"),
        VariantEntry("label_smoothing", 0.0, "argparse", "Label smoothing"),
        VariantEntry("focal_gamma", 0.0, "argparse", "Gamma focal loss"),
        VariantEntry("pos_weight", 1.0, "argparse", "Peso classe positiva"),
        VariantEntry("tta_views", 1, "argparse", "Vistas TTA"),
        VariantEntry("fundus_crop_ratio", 0.9, "argparse", "Crop central"),
        # Code-level knobs
        VariantEntry("cudnn_benchmark", True, "code", "torch.backends.cudnn.benchmark"),
        VariantEntry("tf32_matmul", True, "code", "allow_tf32"),
        VariantEntry("tf32_cudnn", True, "code", "allow_tf32"),
        VariantEntry("optimizer", "AdamW", "code", "Otimizador AdamW"),
        VariantEntry("amp_enabled", True, "code", "AMP GradScaler"),
        VariantEntry("use_dali", True, "code", "NVIDIA DALI pipeline"),
        VariantEntry("use_timm", True, "code", "timm.create_model"),
        VariantEntry("torch_compile", True, "code", "torch.compile(mode='reduce-overhead')"),
        VariantEntry("ema_decay", 0.999, "code", "EMAManager decay"),
        VariantEntry("ema_start_epoch", "60%", "code", "Start EMA at 60% of epochs"),
        VariantEntry("head_weight_decay", 1e-4, "code", "Weight decay para head"),
        VariantEntry("fine_tune_weight_decay", 1e-5, "code", "Weight decay para fine-tune"),
    ]


def _tensorflow_base_entries() -> List[VariantEntry]:
    """Entradas extraídas de tensorflow_base/dr_hcpa_v2_2024.py."""
    return [
        VariantEntry("tfrec_dir", "./data/all", "argparse", "Diretório TFRecords"),
        VariantEntry("dataset", "all", "argparse", "Nome do dataset"),
        VariantEntry("results", "./results/all", "argparse", "Diretório resultados"),
        VariantEntry("exec", 0, "argparse", "ID execução"),
        VariantEntry("img_sizes", 299, "argparse", "Tamanho imagem"),
        VariantEntry("batch_size", 32, "argparse", "Batch por réplica"),
        VariantEntry("epochs", 50, "argparse", "Épocas"),
        VariantEntry("lrate", 1e-3, "argparse", "Learning rate"),
        VariantEntry("num_thresholds", 200, "argparse", "Thresholds ROC"),
        VariantEntry("verbose", 1, "argparse", "Verbosidade Keras"),
        VariantEntry("model", "InceptionV3", "argparse", "Backbone"),
        VariantEntry("augment", True, "argparse", "Augmentações"),
        VariantEntry("normalize", "preprocess", "argparse", "Normalização"),
        VariantEntry("cores", 0, "argparse", "CPU threads"),
        VariantEntry("log_gpu_mem", True, "argparse", "Log memória GPU"),
        VariantEntry("warmup_epochs", 3, "argparse", "Warmup epochs"),
        VariantEntry("min_lr", 1e-6, "argparse", "LR mínimo"),
        VariantEntry("label_smoothing", 0.0, "argparse", "Label smoothing"),
        # Code-level
        VariantEntry("memory_growth", True, "code", "set_memory_growth"),
        VariantEntry("optimizer", "Adam", "code", "Otimizador"),
        VariantEntry("shuffle_buffer", 8192, "code", "tf.data shuffle buffer"),
        VariantEntry("scheduler_type", "warmup_cosine", "code", "WarmupCosineSchedule"),
    ]


def _tensorflow_opt_entries() -> List[VariantEntry]:
    """Entradas extraídas de tensorflow_opt/dr_hcpa_v2_2024.py."""
    return [
        VariantEntry("tfrec_dir", "./data/all", "argparse", "Diretório TFRecords"),
        VariantEntry("dataset", "all", "argparse", "Nome dataset"),
        VariantEntry("results", "./results/all", "argparse", "Diretório resultados"),
        VariantEntry("exec", 0, "argparse", "ID execução"),
        VariantEntry("img_sizes", 299, "argparse", "Tamanho imagem"),
        VariantEntry("batch_size", 96, "argparse", "Batch por réplica"),
        VariantEntry("epochs", 200, "argparse", "Épocas"),
        VariantEntry("num_classes", 2, "argparse", "Número de classes"),
        VariantEntry("lrate", 0.003, "argparse", "Learning rate"),
        VariantEntry("num_thresholds", 200, "argparse", "Thresholds ROC"),
        VariantEntry("wait_epochs", 30, "argparse", "Épocas de espera"),
        VariantEntry("show_files", False, "argparse", "Mostrar arquivos"),
        VariantEntry("verbose", 1, "argparse", "Verbosidade"),
        VariantEntry("model", "InceptionV3", "argparse", "Backbone"),
        VariantEntry("cache_dir", "none", "argparse", "Cache dir TF dataset"),
        VariantEntry("freeze_epochs", 1, "argparse", "Freeze epochs"),
        VariantEntry("fine_tune_lr_factor", 0.1, "argparse", "Fator LR fine-tune"),
        VariantEntry("fine_tune_at", -200, "argparse", "Layer index fine-tune"),
        VariantEntry("fine_tune_lr", 2e-4, "argparse", "LR abs fine-tune"),
        VariantEntry("scheduler", "cosine", "argparse", "Scheduler strategy"),
        VariantEntry("warmup_epochs", 2, "argparse", "Warmup epochs"),
        VariantEntry("min_lr", 1e-6, "argparse", "LR mínimo"),
        VariantEntry("grad_clip_norm", 1.0, "argparse", "Gradient clipping"),
        VariantEntry("mixup_alpha", 0.3, "argparse", "Alpha mixup"),
        VariantEntry("cutmix_alpha", 0.0, "argparse", "Alpha cutmix"),
        VariantEntry("label_smoothing", 0.01, "argparse", "Label smoothing"),
        VariantEntry("focal_gamma", 0.0, "argparse", "Focal loss gamma"),
        VariantEntry("pos_weight", 1.0, "argparse", "Peso classe positiva"),
        VariantEntry("fundus_crop_ratio", 0.9, "argparse", "Crop central"),
        VariantEntry("augment", True, "argparse", "Augmentações"),
        VariantEntry("freeze_bn", True, "argparse", "Freeze BatchNorm"),
        VariantEntry("fine_tune_schedule", "", "argparse", "Schedule progressivo"),
        VariantEntry("normalize", "preprocess", "argparse", "Normalização"),
        VariantEntry("channels_last", True, "argparse", "Channels Last"),
        VariantEntry("h2d_uint8", False, "argparse", "uint8 HtoD"),
        VariantEntry("tta_views", 1, "argparse", "Vistas TTA"),
        VariantEntry("cores", 0, "argparse", "CPU threads"),
        VariantEntry("mixed_precision", True, "argparse", "Mixed precision fp16"),
        VariantEntry("use_dali", False, "argparse", "Usar DALI"),
        VariantEntry("dali_threads", 4, "argparse", "DALI threads"),
        VariantEntry("dali_layout", "NHWC", "argparse", "DALI layout"),
        VariantEntry("dali_seed", 2024, "argparse", "DALI seed"),
        VariantEntry("recompute_backbone", False, "argparse", "Recompute backbone"),
        VariantEntry("jit_compile", False, "argparse", "JIT compile"),
        VariantEntry("auc_target", 0.95, "argparse", "AUC target"),
        # Code-level
        VariantEntry("memory_growth", True, "code", "set_memory_growth"),
        VariantEntry("dali_available_check", True, "code", "Detecção automática DALI"),
        VariantEntry("auto_shard_policy", "DATA", "code", "AutoShardPolicy"),
    ]


def _monai_base_entries() -> List[VariantEntry]:
    """Entradas de monai_base (TrainConfig dataclass + train.py argparse)."""
    return [
        VariantEntry("results_dir", "./results/monai_puro", "argparse", "Diretório resultados"),
        VariantEntry("tfrec_dir", "./data/all-tfrec", "argparse", "Diretório TFRecords"),
        VariantEntry("image_size", 299, "argparse/dataclass", "Tamanho imagem"),
        VariantEntry("num_classes", 2, "dataclass", "Número de classes"),
        VariantEntry("fundus_crop_ratio", 0.9, "argparse/dataclass", "Crop central"),
        VariantEntry("normalize", "inception", "argparse/dataclass", "Normalização"),
        VariantEntry("augment", True, "argparse/dataclass", "Augmentações"),
        VariantEntry("color_jitter", 0.1, "dataclass", "Color jitter"),
        VariantEntry("mixup_alpha", 0.0, "argparse/dataclass", "Alpha mixup"),
        VariantEntry("cutmix_alpha", 0.0, "argparse/dataclass", "Alpha cutmix"),
        VariantEntry("label_smoothing", 0.0, "argparse/dataclass", "Label smoothing"),
        VariantEntry("use_dali", False, "dataclass", "Usar DALI"),
        VariantEntry("smart_cache", False, "dataclass", "MONAI SmartCache"),
        VariantEntry("cache_rate", 0.0, "dataclass", "Cache rate"),
        VariantEntry("batch_size", 96, "argparse/dataclass", "Batch de treino"),
        VariantEntry("eval_batch_size", None, "dataclass", "Batch de avaliação"),
        VariantEntry("num_workers", 8, "argparse/dataclass", "DataLoader workers"),
        VariantEntry("pin_memory", True, "dataclass", "Pin memory"),
        VariantEntry("persistent_workers", True, "dataclass", "Persistent workers"),
        VariantEntry("prefetch_factor", 2, "dataclass", "Prefetch factor"),
        VariantEntry("drop_last", True, "dataclass", "Drop last batch"),
        VariantEntry("model_name", "inception_v3", "argparse/dataclass", "Backbone"),
        VariantEntry("pretrained", True, "dataclass", "Usar pesos pré-treinados"),
        VariantEntry("dropout", 0.2, "dataclass", "Dropout rate"),
        VariantEntry("channels_last", True, "argparse/dataclass", "Channels last"),
        VariantEntry("epochs", 200, "argparse/dataclass", "Épocas totais"),
        VariantEntry("learning_rate", 3e-4, "argparse/dataclass", "Learning rate"),
        VariantEntry("min_lr", 0.0, "dataclass", "LR mínimo"),
        VariantEntry("warmup_epochs", 0, "dataclass", "Warmup epochs"),
        VariantEntry("weight_decay", 1e-4, "argparse/dataclass", "Weight decay"),
        VariantEntry("optimizer", "adamw", "dataclass", "Otimizador"),
        VariantEntry("scheduler", "none", "argparse/dataclass", "Scheduler"),
        VariantEntry("grad_clip_norm", 1.0, "argparse/dataclass", "Gradient clipping"),
        VariantEntry("pos_weight", None, "argparse/dataclass", "Peso classe positiva"),
        VariantEntry("amp", True, "argparse/dataclass", "AMP habilitado"),
        VariantEntry("compile", False, "argparse/dataclass", "torch.compile"),
        VariantEntry("gradient_accumulation", 1, "argparse/dataclass", "Acumulação de gradientes"),
        VariantEntry("ema_decay", 0.0, "argparse/dataclass", "EMA decay"),
        VariantEntry("ema_on_cpu", False, "argparse/dataclass", "EMA em CPU"),
        VariantEntry("threshold", 0.5, "dataclass", "Limiar de classificação"),
        VariantEntry("tta_views", 1, "dataclass", "Vistas TTA"),
        VariantEntry("patience", 0, "dataclass", "Early stop patience"),
        VariantEntry("target_metric", "auc", "dataclass", "Métrica alvo"),
        VariantEntry("log_every", 50, "argparse/dataclass", "Log a cada N steps"),
        VariantEntry("save_every", 1, "dataclass", "Salvar a cada N épocas"),
        VariantEntry("seed", 2026, "argparse/dataclass", "Seed"),
        VariantEntry("host_prefetch", 2, "dataclass", "Host prefetch"),
        VariantEntry("device_prefetch", 2, "dataclass", "Device prefetch"),
        VariantEntry("use_fake_data", False, "argparse", "Usar dados falsos"),
    ]


def _monai_opt_entries() -> List[VariantEntry]:
    """Entradas de monai_opt (TrainConfig dataclass + train.py argparse)."""
    return [
        VariantEntry("results_dir", "./results/opt_monai", "argparse", "Diretório resultados"),
        VariantEntry("tfrec_dir", "./data/all-tfrec", "argparse", "Diretório TFRecords"),
        VariantEntry("image_size", 299, "argparse/dataclass", "Tamanho imagem"),
        VariantEntry("num_classes", 2, "dataclass", "Número de classes"),
        VariantEntry("fundus_crop_ratio", 0.9, "argparse/dataclass", "Crop central"),
        VariantEntry("normalize", "inception", "argparse/dataclass", "Normalização"),
        VariantEntry("augment", True, "argparse/dataclass", "Augmentações"),
        VariantEntry("color_jitter", 0.1, "dataclass", "Color jitter"),
        VariantEntry("mixup_alpha", 0.2, "argparse", "Alpha mixup"),
        VariantEntry("cutmix_alpha", 0.0, "argparse/dataclass", "Alpha cutmix"),
        VariantEntry("label_smoothing", 0.02, "argparse", "Label smoothing"),
        VariantEntry("use_dali", True, "argparse", "Usar DALI"),
        VariantEntry("smart_cache", False, "dataclass", "MONAI SmartCache"),
        VariantEntry("cache_rate", 0.0, "dataclass", "Cache rate"),
        VariantEntry("batch_size", 96, "argparse", "Batch de treino"),
        VariantEntry("eval_batch_size", None, "dataclass", "Batch de avaliação"),
        VariantEntry("num_workers", 8, "argparse", "DataLoader workers"),
        VariantEntry("pin_memory", True, "dataclass", "Pin memory"),
        VariantEntry("persistent_workers", True, "dataclass", "Persistent workers"),
        VariantEntry("prefetch_factor", 2, "dataclass", "Prefetch factor"),
        VariantEntry("drop_last", True, "dataclass", "Drop last batch"),
        VariantEntry("model_name", "inception_v3", "argparse/dataclass", "Backbone"),
        VariantEntry("pretrained", True, "dataclass", "Pré-treinado"),
        VariantEntry("dropout", 0.2, "dataclass", "Dropout"),
        VariantEntry("channels_last", True, "argparse", "Channels last"),
        VariantEntry("epochs", 200, "argparse/dataclass", "Épocas"),
        VariantEntry("learning_rate", 3e-4, "argparse/dataclass", "Learning rate"),
        VariantEntry("min_lr", 3e-5, "argparse", "LR mínimo"),
        VariantEntry("warmup_epochs", 10, "argparse", "Warmup epochs"),
        VariantEntry("weight_decay", 1e-4, "argparse/dataclass", "Weight decay"),
        VariantEntry("optimizer", "adamw", "dataclass", "Otimizador"),
        VariantEntry("scheduler", "cosine", "argparse", "Scheduler"),
        VariantEntry("grad_clip_norm", 1.0, "argparse/dataclass", "Gradient clipping"),
        VariantEntry("pos_weight", None, "argparse/dataclass", "Peso classe positiva"),
        VariantEntry("amp", True, "argparse/dataclass", "AMP"),
        VariantEntry("compile", True, "argparse", "torch.compile"),
        VariantEntry("gradient_accumulation", 1, "argparse/dataclass", "Grad accumulation"),
        VariantEntry("ema_decay", 0.999, "argparse", "EMA decay"),
        VariantEntry("ema_on_cpu", False, "argparse/dataclass", "EMA em CPU"),
        VariantEntry("threshold", 0.5, "dataclass", "Limiar classificação"),
        VariantEntry("tta_views", 1, "dataclass", "Vistas TTA"),
        VariantEntry("patience", 0, "dataclass", "Early stop patience"),
        VariantEntry("target_metric", "auc", "dataclass", "Métrica alvo"),
        VariantEntry("log_every", 25, "dataclass", "Log a cada N steps"),
        VariantEntry("save_every", 1, "dataclass", "Salvar a cada N"),
        VariantEntry("seed", 2026, "argparse/dataclass", "Seed"),
        VariantEntry("host_prefetch", 2, "dataclass", "Host prefetch"),
        VariantEntry("device_prefetch", 2, "dataclass", "Device prefetch"),
        VariantEntry("use_fake_data", False, "argparse", "Dados falsos"),
    ]


# ── Registry ──
VARIANT_REGISTRY = {
    "pytorch_base": _pytorch_base_entries,
    "pytorch_opt": _pytorch_opt_entries,
    "tensorflow_base": _tensorflow_base_entries,
    "tensorflow_opt": _tensorflow_opt_entries,
    "monai_base": _monai_base_entries,
    "monai_opt": _monai_opt_entries,
}


def audit_variant(variant_name: str) -> VariantAudit:
    """Audita uma variante e retorna suas entradas."""
    factory = VARIANT_REGISTRY.get(variant_name)
    if factory is None:
        raise ValueError(f"Variante desconhecida: {variant_name}")
    return VariantAudit(variant_name=variant_name, entries=factory())


def audit_all() -> Dict[str, VariantAudit]:
    """Audita todas as 6 variantes."""
    return {name: audit_variant(name) for name in VARIANT_REGISTRY}


def diff_base_opt(stack: str) -> StackDiff:
    """Compara base vs opt dentro de um stack."""
    base_name = f"{stack}_base"
    opt_name = f"{stack}_opt"
    base_audit = audit_variant(base_name)
    opt_audit = audit_variant(opt_name)

    base_dict = base_audit.as_dict()
    opt_dict = opt_audit.as_dict()

    all_keys = set(base_dict.keys()) | set(opt_dict.keys())
    only_in_base = [k for k in all_keys if k in base_dict and k not in opt_dict]
    only_in_opt = [k for k in all_keys if k in opt_dict and k not in base_dict]
    changed = {}
    same = {}
    for k in all_keys:
        if k in base_dict and k in opt_dict:
            if base_dict[k] != opt_dict[k]:
                changed[k] = (base_dict[k], opt_dict[k])
            else:
                same[k] = base_dict[k]

    return StackDiff(
        stack=stack,
        only_in_base=sorted(only_in_base),
        only_in_opt=sorted(only_in_opt),
        changed=changed,
        same=same,
    )


def generate_audit_report(include_historical_stats: bool = True) -> str:
    """Gera relatório completo da auditoria (ETAPA 1).

    Args:
        include_historical_stats: Se True, anexa seção com desempenho histórico
            das 6 variantes obtido da base de conhecimento offline.
    """
    lines = ["=" * 80, "RELATÓRIO DE AUDITORIA — ETAPA 1", "=" * 80, ""]

    for stack in ("pytorch", "tensorflow", "monai"):
        lines.append(f"{'─' * 40}")
        lines.append(f"STACK: {stack.upper()}")
        lines.append(f"{'─' * 40}")

        for variant_type in ("base", "opt"):
            name = f"{stack}_{variant_type}"
            audit = audit_variant(name)
            lines.append(f"\n  [{name}] — {len(audit.entries)} entradas configuráveis:")
            for e in audit.entries:
                lines.append(f"    {e.name:30s} = {str(e.value):20s} ({e.source}) — {e.description}")

        diff = diff_base_opt(stack)
        lines.append(f"\n  DIFERENÇAS base vs opt:")
        if diff.only_in_base:
            lines.append(f"    Só no base: {diff.only_in_base}")
        if diff.only_in_opt:
            lines.append(f"    Só no opt:  {diff.only_in_opt}")
        if diff.changed:
            lines.append(f"    Alterados ({len(diff.changed)}):")
            for k, (bv, ov) in sorted(diff.changed.items()):
                lines.append(f"      {k:30s}: {bv!s:20s} -> {ov!s}")
        lines.append("")

    if include_historical_stats:
        lines += _historical_stats_section()

    return "\n".join(lines)


def _historical_stats_section() -> List[str]:
    """Gera seção com desempenho histórico usando a base de conhecimento offline.

    A importação é feita localmente para evitar dependência circular caso
    offline_knowledge.py importe audit.py.
    """
    lines: List[str] = []
    try:
        from .offline_knowledge import get_knowledge_base  # local import to avoid circular deps
        kb = get_knowledge_base()
        lines.append("=" * 80)
        lines.append("DESEMPENHO HISTÓRICO DAS 6 VARIANTES — BASE DE CONHECIMENTO OFFLINE")
        lines.append("=" * 80)
        lines.append(kb.summary_report())
        lines.append("")

        # Ranking by quality
        lines.append("─" * 60)
        lines.append("RANKING POR QUALIDADE (AUC):")
        ranking_q = kb.rank_variants_by_objective(objective="auc")
        for i, (name, score) in enumerate(ranking_q, 1):
            lines.append(f"  {i}. {name:<20s}  score={score:.4f}")
        lines.append("")

        # Ranking by efficiency
        lines.append("─" * 60)
        lines.append("RANKING POR EFICIÊNCIA (throughput + memória):")
        ranking_e = kb.rank_variants_by_objective(objective="efficiency")
        for i, (name, score) in enumerate(ranking_e, 1):
            lines.append(f"  {i}. {name:<20s}  score={score:.4f}")
        lines.append("")

        # Ranking by balanced
        lines.append("─" * 60)
        lines.append("RANKING BALANCEADO (qualidade + eficiência):")
        ranking_b = kb.rank_variants_by_objective(objective="balanced")
        for i, (name, score) in enumerate(ranking_b, 1):
            lines.append(f"  {i}. {name:<20s}  score={score:.4f}")
        lines.append("")

    except Exception as exc:
        lines.append("─" * 60)
        lines.append(f"[AVISO] Não foi possível carregar base de conhecimento offline: {exc}")
        lines.append("")

    return lines
