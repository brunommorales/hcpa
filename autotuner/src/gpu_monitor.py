"""
gpu_monitor.py — Monitoramento contínuo de GPU via NVML (preferencialmente) ou nvidia-smi fallback.

Coleta métricas de utilização, memória, temperatura e potência durante o treino.
"""
from __future__ import annotations

import subprocess
import threading
import time
import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class GPUSnapshot:
    """Snapshot de telemetria GPU coletado em um instante."""
    timestamp: float  # time.time()
    gpu_index: int
    memory_used_mb: float
    memory_total_mb: float
    utilization_gpu_pct: float
    utilization_mem_pct: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None


class GPUMonitor:
    """Monitor assíncrono de GPU — coleta snapshots em intervalo configurável."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 5.0):
        self.gpu_index = gpu_index
        self.interval_s = max(1.0, interval_s)
        self._snapshots: List[GPUSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._nvml_available = False
        self._nvml_handle = None
        self._try_init_nvml()

    def _try_init_nvml(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._nvml_available = True
        except Exception:
            self._nvml_available = False

    def _collect_nvml(self) -> Optional[GPUSnapshot]:
        try:
            import pynvml
            handle = self._nvml_handle
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
            except Exception:
                power = None
            return GPUSnapshot(
                timestamp=time.time(),
                gpu_index=self.gpu_index,
                memory_used_mb=mem.used / (1024 ** 2),
                memory_total_mb=mem.total / (1024 ** 2),
                utilization_gpu_pct=float(util.gpu),
                utilization_mem_pct=float(util.memory),
                temperature_c=float(temp) if temp is not None else None,
                power_draw_w=power,
            )
        except Exception:
            return None

    def _collect_smi(self) -> Optional[GPUSnapshot]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.gpu_index}",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8",
                timeout=10,
            )
            parts = [p.strip() for p in out.strip().split(",")]
            if len(parts) < 4:
                return None

            def _f(s):
                s = s.strip().replace("[N/A]", "").replace("N/A", "").strip()
                return float(s) if s else 0.0

            return GPUSnapshot(
                timestamp=time.time(),
                gpu_index=self.gpu_index,
                memory_used_mb=_f(parts[0]),
                memory_total_mb=_f(parts[1]),
                utilization_gpu_pct=_f(parts[2]),
                utilization_mem_pct=_f(parts[3]),
                temperature_c=_f(parts[4]) if len(parts) > 4 else None,
                power_draw_w=_f(parts[5]) if len(parts) > 5 else None,
            )
        except Exception:
            return None

    def collect_once(self) -> Optional[GPUSnapshot]:
        snap = None
        if self._nvml_available:
            snap = self._collect_nvml()
        if snap is None:
            snap = self._collect_smi()
        if snap is not None:
            self._snapshots.append(snap)
        return snap

    def _loop(self):
        while self._running:
            self.collect_once()
            time.sleep(self.interval_s)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="gpu-monitor")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)
            self._thread = None

    @property
    def snapshots(self) -> List[GPUSnapshot]:
        return list(self._snapshots)

    @property
    def latest(self) -> Optional[GPUSnapshot]:
        return self._snapshots[-1] if self._snapshots else None

    def peak_memory_mb(self) -> float:
        if not self._snapshots:
            return 0.0
        return max(s.memory_used_mb for s in self._snapshots)

    def avg_utilization(self) -> float:
        if not self._snapshots:
            return 0.0
        return sum(s.utilization_gpu_pct for s in self._snapshots) / len(self._snapshots)

    def clear(self):
        self._snapshots.clear()

    def __del__(self):
        self.stop()
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
