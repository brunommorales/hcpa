"""
Performance Profiling - Coleta métricas computacionais (tempo, memória, throughput).
"""

import time
import torch
from typing import Optional, Dict


class PerformanceProfiler:
    """
    Coleta e agrega métricas de performance durante treinamento/validação.

    Métricas capturadas:
    - Tempo por época
    - Tempo médio por batch
    - Throughput (imagens por segundo)
    - Consumo médio de memória GPU durante a época
    - AUC por tempo (eficiência clínica)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.epoch_times = []
        self.batch_times = []
        self.memory_avgs = []
        self._epoch_batch_times = []
        self._epoch_memory_samples = []

        self._epoch_start_time: Optional[float] = None
        self._batch_start_time: Optional[float] = None
        self._total_samples: int = 0

    def _sample_gpu_memory_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return max(allocated, reserved)

    def start_epoch(self) -> None:
        """Marca início de uma época."""
        # Sincroniza GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._epoch_start_time = time.perf_counter()
        self._total_samples = 0
        self._epoch_batch_times = []
        self._epoch_memory_samples = []

    def start_batch(self) -> None:
        """Marca início de um batch."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._batch_start_time = time.perf_counter()

    def end_batch(self, num_samples: int) -> float:
        """
        Marca fim de um batch e retorna tempo decorrido.

        Args:
            num_samples: Número de samples neste batch (para cálculo de throughput)

        Returns:
            Tempo do batch em segundos
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        batch_time = time.perf_counter() - self._batch_start_time
        self.batch_times.append(batch_time)
        self._epoch_batch_times.append(batch_time)
        self._total_samples += num_samples
        self._epoch_memory_samples.append(self._sample_gpu_memory_mb())

        return batch_time

    def end_epoch(self) -> Dict[str, float]:
        """
        Marca fim da época e retorna métricas agregadas.

        Returns:
            Dict com:
            - epoch_time: Tempo total da época (segundos)
            - avg_batch_time: Tempo médio por batch (ms)
            - throughput: Imagens por segundo
            - memory_avg_mb: Consumo médio de memória GPU durante a época (MB)
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - self._epoch_start_time

        memory_avg = (
            sum(self._epoch_memory_samples) / len(self._epoch_memory_samples)
            if self._epoch_memory_samples
            else 0.0
        )

        # Computar métricas
        num_batches = len(self._epoch_batch_times)
        avg_batch_time = (sum(self._epoch_batch_times) / num_batches * 1000) if num_batches > 0 else 0
        throughput = self._total_samples / (epoch_time + 1e-7)  # samples/second

        self.epoch_times.append(epoch_time)
        self.memory_avgs.append(memory_avg)

        return {
            "epoch_time": epoch_time,
            "avg_batch_time_ms": avg_batch_time,
            "throughput": throughput,  # img/s
            "memory_avg_mb": memory_avg,
            "num_samples": self._total_samples,
        }

    def get_summary(self) -> Dict[str, float]:
        """
        Retorna resumo estatístico de todas as épocas.

        Returns:
            Dict com médias, máximos e totais das métricas
        """
        if not self.epoch_times:
            return {}

        total_time = sum(self.epoch_times)
        avg_epoch_time = total_time / len(self.epoch_times)
        avg_memory = sum(self.memory_avgs) / len(self.memory_avgs) if self.memory_avgs else 0.0

        avg_batch_time = sum(self.batch_times) / len(self.batch_times) * 1000 if self.batch_times else 0.0

        return {
            "total_time_s": total_time,
            "avg_epoch_time_s": avg_epoch_time,
            "max_epoch_time_s": max(self.epoch_times),
            "min_epoch_time_s": min(self.epoch_times),
            "avg_batch_time_ms": avg_batch_time,
            "avg_memory_mb": avg_memory,
            "num_epochs": len(self.epoch_times),
        }

    def efficiency_score(self, auc: float) -> Dict[str, float]:
        """
        Computa métricas de eficiência (AUC por unidade de recursos).

        Args:
            auc: AUC final do modelo

        Returns:
            Dict com métricas de eficiência
        """
        summary = self.get_summary()
        if not summary:
            return {}

        total_time = summary["total_time_s"]
        avg_memory = summary["avg_memory_mb"]

        return {
            "auc_per_second": auc / (total_time + 1e-7),
            "auc_per_mb": auc / (avg_memory + 1e-7),
            "total_gpu_memory_seconds": avg_memory * total_time,  # GPU-seconds (crude proxy)
        }

    def reset(self) -> None:
        """Reseta todas as métricas."""
        self.epoch_times.clear()
        self.batch_times.clear()
        self.memory_avgs.clear()
        self._epoch_batch_times.clear()
        self._epoch_memory_samples.clear()
        self._total_samples = 0


def format_profiling_report(
    model_name: str,
    auc: float,
    sensitivity: float,
    specificity: float,
    profiler: PerformanceProfiler,
) -> str:
    """
    Formata relatório legível de profiling.

    Args:
        model_name: Nome do modelo
        auc: AUC final
        sensitivity: Sensibilidade
        specificity: Especificidade
        profiler: Instância de PerformanceProfiler

    Returns:
        String formatada para impressão/logging
    """
    summary = profiler.get_summary()
    efficiency = profiler.efficiency_score(auc)

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║  PROFILING REPORT: {model_name:.<40}  ║
╚══════════════════════════════════════════════════════════════╝

MÉTRICAS CLÍNICAS:
  AUC:              {auc:.4f}
  Sensibilidade:    {sensitivity:.4f}
  Especificidade:   {specificity:.4f}

MÉTRICAS COMPUTACIONAIS (Total de treino):
  Tempo total:      {summary['total_time_s']:.2f}s
  Tempo/época:      {summary['avg_epoch_time_s']:.2f}s (médio)
  Tempo/época:      {summary['min_epoch_time_s']:.2f}s - {summary['max_epoch_time_s']:.2f}s (min-max)
  Tempo/batch:      {summary['avg_batch_time_ms']:.2f}ms (médio)

MEMÓRIA GPU:
  Média:            {summary['avg_memory_mb']:.2f} MB

EFICIÊNCIA:
  AUC / segundo:    {efficiency['auc_per_second']:.6f}
  AUC / MB:         {efficiency['auc_per_mb']:.6f}
  GP-seconds proxy: {efficiency['total_gpu_memory_seconds']:.0f}
"""
    return report
