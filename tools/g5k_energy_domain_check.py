#!/usr/bin/env python3
"""Check do DOMÍNIO de energia no GH200 (rodar no nó GPU, durante o smoke-test).

Objetivo: confirmar se nvmlDeviceGetTotalEnergyConsumption / PowerUsage medem só o
módulo HOPPER (a GPU) ou incluem a CPU Grace. Se incluir a Grace, a "energia da GPU"
estaria contaminada com CPU — o que invalidaria a métrica.

Como interpretar:
  - Nome do device deve ser um GH200/H100 (Hopper).
  - Power limit (TDP): ~700W => Hopper-only; ~900-1000W => módulo Grace+Hopper.
  - Potência OCIOSA (sem carga): Hopper idle ~70-120W. Se marcar 200W+ ociosa,
    provavelmente inclui a Grace / é module-level.
  - Potência derivada do contador (ΔE/Δt) deve bater com a instantânea.
  - Cruza com `nvidia-smi` (mesma leitura de power do NVML).
"""
import time
import subprocess

try:
    import pynvml as nvml
except Exception as e:
    raise SystemExit(f"pynvml ausente: {e}")

nvml.nvmlInit()
h = nvml.nvmlDeviceGetHandleByIndex(0)

def _try(fn, *a, div=1.0):
    try:
        return fn(*a) / div
    except Exception:
        return None

name = nvml.nvmlDeviceGetName(h)
name = name.decode() if isinstance(name, bytes) else name
tdp = _try(nvml.nvmlDeviceGetPowerManagementLimit, h, div=1000.0)
p0 = _try(nvml.nvmlDeviceGetPowerUsage, h, div=1000.0)
e0 = _try(nvml.nvmlDeviceGetTotalEnergyConsumption, h, div=1000.0)

print("=" * 64)
print(f"Device            : {name}")
print(f"Power limit (TDP) : {tdp:.0f} W" if tdp else "Power limit: N/A")
print(f"Power instant     : {p0:.1f} W" if p0 else "Power instant: N/A")

# janela ociosa de ~5s para potencia derivada do contador
SECS = 5.0
if e0 is not None:
    t0 = time.perf_counter()
    time.sleep(SECS)
    e1 = _try(nvml.nvmlDeviceGetTotalEnergyConsumption, h, div=1000.0)
    dt = time.perf_counter() - t0
    p_derived = (e1 - e0) / dt if (e1 is not None and dt > 0) else None
    print(f"Power derivada    : {p_derived:.1f} W  (ΔE={e1-e0:.1f} J em {dt:.1f}s)" if p_derived else "Power derivada: N/A")

print("-" * 64)
print("nvidia-smi (cross-check):")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,power.draw,power.limit,utilization.gpu,memory.used",
         "--format=csv,noheader"], encoding="utf-8")
    print("  " + out.strip())
except Exception as ex:
    print(f"  nvidia-smi falhou: {ex}")

print("-" * 64)
if tdp:
    if tdp <= 750:
        print("VEREDITO provável: TDP ~700W => energia NVML = HOPPER (GPU). OK, GPU-only.")
    else:
        print(f"ATENÇÃO: TDP {tdp:.0f}W > 750W => pode ser MÓDULO (Grace+Hopper). "
              "Verificar potência ociosa: se >200W ociosa, energia inclui a Grace.")
print("=" * 64)
nvml.nvmlShutdown()
