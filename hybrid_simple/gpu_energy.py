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


def _resolve_all_local_handles():
    """Handles NVML de TODAS as GPUs que ESTE processo dirige — para o caso de UM
    processo controlar várias GPUs (TF MirroredStrategy / MultiWorker num node).

    Respeita CUDA_VISIBLE_DEVICES (índices ou UUIDs). Se não estiver setado, usa
    todas as GPUs do node. Retorna [] se o NVML não iniciar.

    NÃO usar para DDP: lá cada rank tem 1 GPU (current_device) e a soma entre GPUs
    é feita por all_reduce entre os processos — enumerar 'todas as visíveis' aqui
    contaria a mesma GPU em vários ranks (dupla contagem).
    """
    if not _ensure_init():
        return []
    handles = []
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    try:
        if cvd and cvd not in ("-1",):
            for tok in (t.strip() for t in cvd.split(",") if t.strip()):
                try:
                    if tok.startswith("GPU-"):
                        handles.append(_nvml.nvmlDeviceGetHandleByUUID(tok.encode()))
                    elif "-" in tok:
                        handles.append(_nvml.nvmlDeviceGetHandleByUUID(f"GPU-{tok}".encode()))
                    else:
                        handles.append(_nvml.nvmlDeviceGetHandleByIndex(int(tok)))
                except Exception:
                    pass
        else:
            for i in range(_nvml.nvmlDeviceGetCount()):
                handles.append(_nvml.nvmlDeviceGetHandleByIndex(i))
    except Exception:
        pass
    return handles


def _sum_local_gpus_enabled() -> bool:
    """Ligado quando UM processo dirige N GPUs (TF Mirrored/MultiWorker). NÃO ligar
    em DDP (1 GPU/processo + all_reduce). Env HCPA_ENERGY_ALL_VISIBLE_GPUS=1."""
    return os.environ.get("HCPA_ENERGY_ALL_VISIBLE_GPUS", "0").strip().lower() in ("1", "true", "yes")


class GpuTelemetry:
    """Leitor pontual de energia / potência / memória real da GPU ativa.

    Todos os métodos retornam None se o NVML estiver indisponível ou se a métrica
    não for suportada pela GPU/driver — o chamador deve tratar None (ex.: fallback).
    """

    def __init__(self, torch_device=None):
        # Caminho single-GPU (auditado): 1 handle = a GPU deste processo.
        # Caminho multi-GPU num processo (TF Mirrored/MultiWorker): soma TODAS as
        # GPUs locais em energia e potência (aditivas). mem/util ficam no primário.
        if _sum_local_gpus_enabled():
            self._handles = _resolve_all_local_handles() or [_resolve_handle(torch_device)]
        else:
            self._handles = [_resolve_handle(torch_device)]
        self._handles = [h for h in self._handles if h is not None]
        self._h = self._handles[0] if self._handles else None

    @property
    def available(self) -> bool:
        return self._h is not None

    @property
    def n_gpus(self) -> int:
        return len(self._handles)

    def energy_j(self):
        """Energia acumulada desde o load do driver (Joules), SOMADA sobre as GPUs
        locais deste processo. Delta entre duas leituras = energia da região."""
        if not self._handles:
            return None
        try:
            total = 0.0
            for h in self._handles:
                total += _nvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0
            return total
        except Exception:
            return None

    def power_w(self):
        """Potência instantânea (Watts), SOMADA sobre as GPUs locais do processo."""
        if not self._handles:
            return None
        try:
            total = 0.0
            for h in self._handles:
                total += _nvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            return total
        except Exception:
            return None

    def util_pct(self):
        """Utilização do núcleo GPU em % (0-100), via nvmlDeviceGetUtilizationRates.
        É a fração do intervalo em que houve >=1 kernel executando. Serve para
        separar tempo GPU-ativo de tempo host/CPU (GPU ociosa). None se indisponível."""
        if self._h is None:
            return None
        try:
            return float(_nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
        except Exception:
            return None

    def mem_util_pct(self):
        """Utilização de BANDA de memória em % (0-100) — nvmlDeviceGetUtilizationRates().memory.
        É o % do tempo em que o controlador de memória esteve lendo/escrevendo. NÃO é
        ocupação (MB) — para MB use memory_used_mb(). Serve para diagnosticar
        memory-bound (mem_util alto) vs compute-bound vs faminta/host-bound (ambos baixos)."""
        if self._h is None:
            return None
        try:
            return float(_nvml.nvmlDeviceGetUtilizationRates(self._h).memory)
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


SAMPLING_DISABLED = -1.0

_SYNC_WARNED = False   # avisa uma unica vez se o TF nao puder sincronizar

# Intervalo de amostragem padrão, em segundos. ÚNICO para todas as abordagens e
# frameworks — é o que torna util%/potência comparáveis entre elas. Sobrepor com
# HCPA_GPU_SAMPLE_MS (0 = desliga a amostragem).
DEFAULT_SAMPLE_INTERVAL_S = 0.2


def _effective_interval(passed: float) -> float:
    """Intervalo efetivo de amostragem (s). A env var HCPA_GPU_SAMPLE_MS, se definida,
    SOBREPÕE qualquer valor passado — uma única manopla para toda a telemetria.

    `HCPA_GPU_SAMPLE_MS=0` DESLIGA a amostragem em background: energia e tempo
    continuam sendo medidos (contador NVML + perf_counter nas bordas), mas
    potência/util/memória ficam None. É o braço de controle do teste de efeito
    observador — e o modo de menor perturbação possível.

    Regra de projeto: a amostragem deve ser <= ~10% do tempo de um step de kernel
    para resolver potência/util no nível do step. Para o step estável mais rápido
    (~25 ms), isso é ~2.5 ms. Ex.: `export HCPA_GPU_SAMPLE_MS=2`.

    IMPORTANTE: (1) NÃO afeta a energia — o contador NVML é integrado em HW e exato
    independentemente do intervalo; só melhora a média/pico de potência/util.
    (2) nvmlDeviceGetUtilizationRates tem janela interna própria de agregação
    (dependente do device); intervalos muito finos re-amostram o mesmo valor —
    resolução por-kernel real exige CUPTI, não NVML (ver gpu_kernel_profile.py).
    (3) Efeito observador: intervalos muito finos adicionam carga de CPU que pode
    perturbar o próprio step — por isso a amostragem é SEMPRE em thread de fundo
    (nunca por-batch no laço de treino), e o intervalo é o mesmo em todos os
    frameworks. Amostrar por-batch tornaria o overhead proporcional ao número de
    batches, que difere entre abordagens — enviesando justamente o util%."""
    ov = os.environ.get("HCPA_GPU_SAMPLE_MS")
    if ov:
        try:
            v = float(ov)
            if v == 0:
                return SAMPLING_DISABLED
            if v > 0:
                return v / 1000.0
        except Exception:
            pass
    return passed


class _Sampler(threading.Thread):
    """Amostra potência, memória real e utilização em background, guardando as séries."""

    def __init__(self, tele: GpuTelemetry, interval: float = 0.2):
        super().__init__(daemon=True)
        self._tele = tele
        self._interval = _effective_interval(interval)
        # NÃO usar self._stop: threading.Thread já tem um método interno _stop()
        # (chamado por join()/_after_fork); sobrescrevê-lo quebra o join.
        self._stop_event = threading.Event()
        self.mem_samples = []
        self.power_samples = []
        self.util_samples = []
        self.mem_util_samples = []
        self.nvml_calls = 0   # auditoria do efeito observador

    def run(self):
        while not self._stop_event.is_set():
            m = self._tele.memory_used_mb()
            if m is not None:
                self.mem_samples.append(m)
            p = self._tele.power_w()
            if p is not None:
                self.power_samples.append(p)
            u = self._tele.util_pct()
            if u is not None:
                self.util_samples.append(u)
            mu = self._tele.mem_util_pct()
            if mu is not None:
                self.mem_util_samples.append(mu)
            self.nvml_calls += 4
            self._stop_event.wait(self._interval)

    def stop(self):
        self._stop_event.set()


def _cuda_sync():
    """Sincroniza a GPU antes de ler energia/tempo, para a fronteira da medição
    refletir o trabalho de fato CONCLUÍDO (CUDA é assíncrono).

    Cobre os DOIS frameworks. Antes, isto só agia em processos PyTorch, e no TF
    confiava-se em que "o Keras sincroniza sozinho em on_test_begin" — uma
    suposição não verificada. Se ela fosse falsa, a energia e o tempo da fase de
    treino do TF estariam sistematicamente deslocados em relação aos do PyTorch,
    e a comparação entre frameworks (o objeto do estudo) ficaria contaminada.
    Sincronizar explicitamente custa ~nada nas bordas de época e remove a dúvida.
    """
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            return
    except Exception:
        pass

    import sys
    tf = sys.modules.get("tensorflow")   # não IMPORTA o TF; só usa se já estiver carregado
    if tf is None:
        return                           # nem torch-CUDA nem TF ativos: nada a sincronizar
    try:
        tf.test.experimental.sync_devices()
    except Exception as exc:
        # NÃO falhar em silêncio: sem sync, a borda da janela mede trabalho que a
        # GPU ainda não terminou, e o tempo/energia do TF fica deslocado em
        # relação ao do PyTorch — exatamente o viés que este estudo compara.
        global _SYNC_WARNED
        if not _SYNC_WARNED:
            _SYNC_WARNED = True
            print("[gpu_energy][ATENCAO] tf.test.experimental.sync_devices() indisponivel "
                  f"({exc}). As bordas de medicao do TensorFlow NAO estao sincronizadas: "
                  "tempo e energia por epoca podem estar deslocados. Verifique a versao do TF.")


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
        self.avg_util_pct = None       # utilização média de COMPUTE da GPU no intervalo (0-100)
        self.peak_util_pct = None
        self.avg_mem_util_pct = None   # utilização média de BANDA de memória (0-100)
        self.peak_mem_util_pct = None
        self.busy_time_s = None        # tempo GPU-ATIVA = elapsed_s * avg_util_pct/100 (tira ociosidade/CPU)
        self.nvml_calls = 0            # quantas leituras NVML a amostragem fez nesta região
        self._e0 = None
        self._t0 = None
        self._sampler = None
        self._stopped = False

    def start(self):
        """Atalho para uso sem `with` (ex.: em volta de uma chamada multi-linha)."""
        return self.__enter__()

    def stop(self):
        """Encerra a medição iniciada por start(). Lê energy_j/peak_mem_mb/avg_power_w."""
        return self.__exit__(None, None, None)

    def __enter__(self):
        _cuda_sync()   # borda limpa: garante que o trabalho anterior terminou
        self._t0 = time.perf_counter()
        self._e0 = self.tele.energy_j()
        if self._bg and self.tele.available \
                and _effective_interval(self._interval) != SAMPLING_DISABLED:
            self._sampler = _Sampler(self.tele, self._interval)
            self._sampler.start()
        return self

    def __exit__(self, *exc):
        # Idempotente: fechar duas vezes (ex.: on_test_begin e depois on_epoch_end,
        # ou vários caminhos de saída) não deve re-medir nem inflar a energia.
        if self._stopped:
            return False
        self._stopped = True
        _cuda_sync()   # borda limpa: garante que os kernels da região terminaram
        self.elapsed_s = time.perf_counter() - self._t0
        e1 = self.tele.energy_j()
        if self._e0 is not None and e1 is not None:
            self.energy_j = max(0.0, e1 - self._e0)
        if self._sampler is not None:
            self._sampler.stop()
            self._sampler.join(timeout=2.0)
            self.nvml_calls = self._sampler.nvml_calls
            if self._sampler.mem_samples:
                self.peak_mem_mb = max(self._sampler.mem_samples)
                self.avg_mem_mb = sum(self._sampler.mem_samples) / len(self._sampler.mem_samples)
            if self._sampler.power_samples:
                self.peak_power_w = max(self._sampler.power_samples)
                self.avg_power_w = sum(self._sampler.power_samples) / len(self._sampler.power_samples)
            if self._sampler.util_samples:
                self.peak_util_pct = max(self._sampler.util_samples)
                self.avg_util_pct = sum(self._sampler.util_samples) / len(self._sampler.util_samples)
            if self._sampler.mem_util_samples:
                self.peak_mem_util_pct = max(self._sampler.mem_util_samples)
                self.avg_mem_util_pct = sum(self._sampler.mem_util_samples) / len(self._sampler.mem_util_samples)
        # Fallback de energia: integrar potência média × tempo, se o contador faltar.
        if self.energy_j is None and self.avg_power_w is not None and self.elapsed_s:
            self.energy_j = self.avg_power_w * self.elapsed_s
        # Tempo GPU-ATIVA (kernel-only): remove a ociosidade/espera-CPU do tempo de parede.
        if self.avg_util_pct is not None and self.elapsed_s is not None:
            self.busy_time_s = self.elapsed_s * (self.avg_util_pct / 100.0)
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
            self.train_avg_util_pct = None
            self.train_peak_util_pct = None
            self.train_busy_time_s = None
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
                self.train_avg_util_pct = self._scope.avg_util_pct
                self.train_peak_util_pct = self._scope.peak_util_pct
                self.train_busy_time_s = self._scope.busy_time_s

    return _EnergyCallback()
