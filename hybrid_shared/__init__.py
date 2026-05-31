"""
Hybrid Shared - Utilitários compartilhados entre modelos híbridos.

Módulos:
- metrics: AUC, sensibilidade, especificidade
- profiling: Medição de performance (tempo, memória, throughput)
- training_utils: Loss functions, schedulers, EMA
- data_loader_bridge: Wrapper para dataloaders
"""

from .metrics import (
    compute_metrics,
    compute_auc,
    compute_sens_spec,
    compute_specificity_at_sensitivity,
)
from .profiling import PerformanceProfiler
from .training_utils import compute_loss, create_optimizer, create_scheduler, EMAManager

__all__ = [
    "compute_metrics",
    "compute_auc",
    "compute_sens_spec",
    "compute_specificity_at_sensitivity",
    "PerformanceProfiler",
    "compute_loss",
    "create_optimizer",
    "create_scheduler",
    "EMAManager",
]
