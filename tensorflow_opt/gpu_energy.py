"""
gpu_energy.py — Telemetria de GPU NVIDIA (energia, potência e memória REAL) via NVML.

Projetado para ser importado por scripts PyTorch e TensorFlow/Keras, em qualquer
arquitetura (x86 ou ARM/Grace) e em qualquer ambiente (G5K singularity, GPPD docker).
Degrada graciosamente para None se o NVML não estiver disponível — NUNCA derruba o treino.

Métricas expostas:
- energia: contador de hardware `nvmlDeviceGetTotalEnergyConsumption` (mJ -> J). Exato,
  integrado em hardware (Volta+). É o número de energia primário.
- potência: `nvmlDeviceGetPowerUsage` (mW -> W).
- memória REAL: `nvmlDeviceGetMemoryInfo().used` (consumo real do device; em nó exclusivo
  == footprint do processo). CORRIGE o uso anterior de torch.cuda.memory_allocated/reserved
  e tf.config.experimental.get_memory_info, que medem só o caching allocator do framework.

Uso típico (PyTorch):
    from gpu_energy import GpuTelemetry, EnergyScope
    tele = GpuTelemetry()
    with EnergyScope(tele) as scope:
        ... treino de uma época / teste final ...
    scope.energy_j, scope.peak_mem_mb, scope.avg_mem_mb, scope.avg_power_w

Uso pontual (substituir a amostragem de memória por-batch):
    used = tele.memory_used_mb()   # MB reais; None se NVML indisponível

Uso TensorFlow/Keras:
    from gpu_energy import make_keras_energy_callback
    cb = make_keras_energy_callback()      # mede energia/mem da fase de fit()
    model.fit(..., callbacks=[cb])
    cb.train_energy_j, cb.train_peak_mem_mb, cb.train_avg_power_w
"""
from __future__ import annotations

import os
import threading
import time

try:
    import pynvml as _nvml
    _NVML_IMPORTED = True
except Exception:  # pynvml ausente
    _nvml = None
    _NVML_IMPORTED = False

_INIT_DONE = False
_INIT_OK = False


def _ensure_init() -> bool:
    """Inicializa o NVML uma única vez. Retorna True se disponível."""
    global _INIT_DONE, _INIT_OK
    if _INIT_DONE:
        return _INIT_OK
    _INIT_DONE = True
    if not _NVML_IMPORTED:
        return False
    try:
        _nvml.nvmlInit()
        _INIT_OK = True
    except Exception:
        _INIT_OK = False
    return _INIT_OK


def _resolve_handle(torch_device=None):
    """Resolve o handle NVML da GPU que ESTE processo usa. Framework-neutro.

    Ordem: (1) UUID do device torch ativo — robusto a CUDA_DEVICE_ORDER e à
    diferença entre índice CUDA e índice NVML; (2) primeiro de CUDA_VISIBLE_DEVICES
    (cobre TensorFlow); (3) fallback índice 0.
    """
    if not _ensure_init():
        return None

    # (1) torch: UUID do device ativo
    try:
        import torch  # import tardio: não força dependência em ambiente TF
        if torch.cuda.is_available():
            idx = torch_device if isinstance(torch_device, int) else torch.cuda.current_device()
            try:
                uuid = torch.cuda.get_device_properties(idx).uuid
                return _nvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
            except Exception:
                pass
    except Exception:
        pass

    # (2) CUDA_VISIBLE_DEVICES (primeiro device visível) — funciona p/ TF
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = cvd.split(",")[0].strip() if cvd else ""
    if first and first not in ("-1",):
        try:
            if first.startswith("GPU-"):
                return _nvml.nvmlDeviceGetHandleByUUID(first.encode())
            if "-" in first:  # parece UUID sem prefixo
                return _nvml.nvmlDeviceGetHandleByUUID(f"GPU-{first}".encode())
            return _nvml.nvmlDeviceGetHandleByIndex(int(first))
        except Exception:
            pass

    # (3) fallback: índice 0
    try:
        return _nvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        return None


class GpuTelemetry:
    """Leitor pontual de energia / potência / memória real da GPU ativa.

    Todos os métodos retornam None se o NVML estiver indisponível ou se a métrica
    não for suportada pela GPU/driver — o chamador deve tratar None (ex.: fallback).
    """

    def __init__(self, torch_device=None):
        self._h = _resolve_handle(torch_device)

    @property
    def available(self) -> bool:
        return self._h is not None

    def energy_j(self):
        """Energia acumulada desde o load do driver (Joules). Delta entre duas leituras = energia da região."""
        if self._h is None:
            return None
        try:
            return _nvml.nvmlDeviceGetTotalEnergyConsumption(self._h) / 1000.0
        except Exception:
            return None

    def power_w(self):
        """Potência instantânea (Watts)."""
        if self._h is None:
            return None
        try:
            return _nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
        except Exception:
            return None

    def memory_used_mb(self):
        """Uso REAL de memória do device (MB). Em nó exclusivo = footprint do processo."""
        if self._h is None:
            return None
        try:
            return _nvml.nvmlDeviceGetMemoryInfo(self._h).used / (1024.0 ** 2)
        except Exception:
            return None

    def process_memory_used_mb(self, pid=None):
        """Memória atribuída a ESTE processo (MB) — mais preciso se a GPU for compartilhada."""
        if self._h is None:
            return None
        pid = pid or os.getpid()
        try:
            for p in _nvml.nvmlDeviceGetComputeRunningProcesses(self._h):
                if p.pid == pid and getattr(p, "usedGpuMemory", None):
                    return p.usedGpuMemory / (1024.0 ** 2)
        except Exception:
            pass
        return None


class _Sampler(threading.Thread):
    """Amostra potência e memória real em background, guardando as séries."""

    def __init__(self, tele: GpuTelemetry, interval: float = 0.2):
        super().__init__(daemon=True)
        self._tele = tele
        self._interval = interval
        # NÃO usar self._stop: threading.Thread já tem um método interno _stop()
        # (chamado por join()/_after_fork); sobrescrevê-lo quebra o join.
        self._stop_event = threading.Event()
        self.mem_samples = []
        self.power_samples = []

    def run(self):
        while not self._stop_event.is_set():
            m = self._tele.memory_used_mb()
            if m is not None:
                self.mem_samples.append(m)
            p = self._tele.power_w()
            if p is not None:
                self.power_samples.append(p)
            self._stop_event.wait(self._interval)

    def stop(self):
        self._stop_event.set()


class EnergyScope:
    """Context manager que mede, numa região (ex.: uma época ou o teste final):

    - energy_j      : energia (J) pelo contador NVML (delta enter/exit) — exato
    - peak_mem_mb   : pico de memória REAL usada (MB)
    - avg_mem_mb    : média de memória REAL usada (MB)
    - avg_power_w   : potência média (W)
    - peak_power_w  : potência de pico (W)
    - elapsed_s     : tempo de parede (s)

    Tudo None se o NVML não estiver disponível. Se o contador de energia não existir
    mas houver potência, energy_j cai para o fallback potência_média × tempo.
    """

    def __init__(self, telemetry: GpuTelemetry, sample_interval: float = 0.2,
                 background_sampling: bool = True):
        self.tele = telemetry
        self._interval = sample_interval
        self._bg = background_sampling
        self.energy_j = None
        self.peak_mem_mb = None
        self.avg_mem_mb = None
        self.avg_power_w = None
        self.peak_power_w = None
        self.elapsed_s = None
        self._e0 = None
        self._t0 = None
        self._sampler = None

    def start(self):
        """Atalho para uso sem `with` (ex.: em volta de uma chamada multi-linha)."""
        return self.__enter__()

    def stop(self):
        """Encerra a medição iniciada por start(). Lê energy_j/peak_mem_mb/avg_power_w."""
        return self.__exit__(None, None, None)

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._e0 = self.tele.energy_j()
        if self._bg and self.tele.available:
            self._sampler = _Sampler(self.tele, self._interval)
            self._sampler.start()
        return self

    def __exit__(self, *exc):
        self.elapsed_s = time.perf_counter() - self._t0
        e1 = self.tele.energy_j()
        if self._e0 is not None and e1 is not None:
            self.energy_j = max(0.0, e1 - self._e0)
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=2.0)
            if self._sampler.mem_samples:
                self.peak_mem_mb = max(self._sampler.mem_samples)
                self.avg_mem_mb = sum(self._sampler.mem_samples) / len(self._sampler.mem_samples)
            if self._sampler.power_samples:
                self.peak_power_w = max(self._sampler.power_samples)
                self.avg_power_w = sum(self._sampler.power_samples) / len(self._sampler.power_samples)
        # Fallback de energia: integrar potência média × tempo, se o contador faltar.
        if self.energy_j is None and self.avg_power_w is not None and self.elapsed_s:
            self.energy_j = self.avg_power_w * self.elapsed_s
        return False


def make_keras_energy_callback(telemetry: GpuTelemetry | None = None,
                               sample_interval: float = 0.2):
    """Cria um tf.keras.callbacks.Callback que mede energia/memória/potência da fase
    de treino (entre on_train_begin e on_train_end). Importa o TF tardiamente para
    manter este módulo framework-neutro.

    Após o fit(), leia: cb.train_energy_j, cb.train_peak_mem_mb, cb.train_avg_power_w.
    """
    import tensorflow as tf  # import tardio

    tele = telemetry or GpuTelemetry()

    class _EnergyCallback(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.train_energy_j = None
            self.train_peak_mem_mb = None
            self.train_avg_mem_mb = None
            self.train_avg_power_w = None
            self.train_peak_power_w = None
            self._scope = None

        def on_train_begin(self, logs=None):
            self._scope = EnergyScope(tele, sample_interval)
            self._scope.__enter__()

        def on_train_end(self, logs=None):
            if self._scope is not None:
                self._scope.__exit__(None, None, None)
                self.train_energy_j = self._scope.energy_j
                self.train_peak_mem_mb = self._scope.peak_mem_mb
                self.train_avg_mem_mb = self._scope.avg_mem_mb
                self.train_avg_power_w = self._scope.avg_power_w
                self.train_peak_power_w = self._scope.peak_power_w

    return _EnergyCallback()
