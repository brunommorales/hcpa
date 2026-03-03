"""
gpu_discovery.py — ETAPA 0: Detecção de GPU NVIDIA via nvidia-smi (obrigatório).

Coleta modelo, driver, memória total/livre/usada, utilização (%), compute
capability, arquitetura e suporte a funcionalidades (TF32, BF16, Flash Attn).
Respeita CUDA_VISIBLE_DEVICES.  Degrada para CPU se NVIDIA ausente.
"""
from __future__ import annotations

import os
import subprocess
import dataclasses
from typing import Dict, List, Optional, Tuple


# ── Mapeamento compute_capability → (arquitetura, nome, bw_gb_s estimado) ──────
_CC_PROFILES: Dict[str, Tuple[str, str, float]] = {
    # (architecture, marketing_name, memory_bandwidth_gb_s)
    "9.0":  ("hopper",      "Hopper (H100/H200)",   3350.0),
    "8.9":  ("ada",         "Ada Lovelace (L40S/RTX4090/4080/4070)", 717.0),
    "8.6":  ("ampere",      "Ampere (RTX3090/A40/A30)", 936.0),
    "8.0":  ("ampere",      "Ampere (A100/A30)",     2000.0),
    "7.5":  ("turing",      "Turing (T4/RTX2080)",   320.0),
    "7.0":  ("volta",       "Volta (V100)",           900.0),
    "6.1":  ("pascal",      "Pascal (P4/GTX1080)",    320.0),
    "6.0":  ("pascal",      "Pascal (P100)",          720.0),
}


def _infer_architecture(cc: Optional[str]) -> str:
    """Retorna nome da arquitetura dado o compute capability."""
    if not cc:
        return "unknown"
    profile = _CC_PROFILES.get(cc)
    if profile:
        return profile[0]
    major = cc.split(".")[0] if "." in cc else cc
    return {
        "9": "hopper", "8": "ampere_or_ada", "7": "volta_or_turing",
        "6": "pascal", "5": "maxwell",
    }.get(major, "unknown")


def _estimate_bandwidth(cc: Optional[str], gpu_name: str, mem_total_mb: float) -> float:
    """Estima largura de banda de memória em GB/s baseado no CC e nome da GPU."""
    if cc and cc in _CC_PROFILES:
        base_bw = _CC_PROFILES[cc][2]
        # RTX 4070 tem BW menor que o topo da geração Ada
        name_lower = gpu_name.lower()
        if "4070 ti" in name_lower:
            return 504.0
        if "4070 super" in name_lower:
            return 504.0
        if "4070" in name_lower:
            return 504.0
        if "4080" in name_lower:
            return 717.0
        if "4090" in name_lower:
            return 1008.0
        if "l40s" in name_lower or "l40" in name_lower:
            return 864.0
        return base_bw
    # Fallback por memória total
    if mem_total_mb >= 70000:   # H200 SXM
        return 4800.0
    if mem_total_mb >= 40000:   # A100 80 GB
        return 2000.0
    if mem_total_mb >= 20000:   # A100 40 GB
        return 1555.0
    if mem_total_mb >= 12000:   # L40S / RTX3090 / 4080
        return 600.0
    return 300.0


@dataclasses.dataclass
class GPUInfo:
    index: int
    name: str
    driver_version: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_gpu_pct: float
    utilization_mem_pct: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    # ── Hardware profile (adicionado na v2) ──────────────────────────────
    compute_capability: Optional[str] = None          # "8.9", "9.0", etc.
    architecture: Optional[str] = None                # "ada", "hopper", "ampere", …
    memory_bandwidth_gb_s: Optional[float] = None     # GB/s (estimado)
    # Suporte a funcionalidades de precisão
    has_tf32: bool = False          # Ampere+ (CC >= 8.0)
    has_bf16: bool = False          # Ampere+ (CC >= 8.0)
    has_flash_attention: bool = False  # Ampere+ (CC >= 8.0)
    supports_torch_compile: bool = False  # Ampere+ (CC >= 7.0)

    def enrich_from_cc(self):
        """Preenche campos derivados a partir do compute_capability."""
        cc = self.compute_capability
        if not cc:
            return
        self.architecture = _infer_architecture(cc)
        self.memory_bandwidth_gb_s = _estimate_bandwidth(cc, self.name, self.memory_total_mb)
        try:
            major = float(cc.split(".")[0])
            minor = float(cc.split(".")[1]) if "." in cc else 0
            cc_float = major + minor * 0.1
            self.has_tf32 = cc_float >= 8.0
            self.has_bf16 = cc_float >= 8.0
            self.has_flash_attention = cc_float >= 8.0
            self.supports_torch_compile = cc_float >= 7.0
        except (ValueError, IndexError):
            pass

    def hw_summary(self) -> str:
        """Resumo compacto do perfil de hardware."""
        cc_str = f"CC {self.compute_capability}" if self.compute_capability else "CC unknown"
        arch = self.architecture or "unknown"
        bw = f"{self.memory_bandwidth_gb_s:.0f} GB/s" if self.memory_bandwidth_gb_s else "bw unknown"
        feats = []
        if self.has_tf32:
            feats.append("TF32")
        if self.has_bf16:
            feats.append("BF16")
        if self.has_flash_attention:
            feats.append("FlashAttn")
        if self.supports_torch_compile:
            feats.append("torch.compile")
        feat_str = " ".join(feats) if feats else "legacy"
        return f"{cc_str} | arch={arch} | {bw} | {feat_str}"


@dataclasses.dataclass
class GPUDiscoveryResult:
    available: bool
    gpus: List[GPUInfo]
    visible_device_indices: Optional[List[int]]  # from CUDA_VISIBLE_DEVICES
    error: Optional[str] = None

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        if not self.gpus:
            return None
        if self.visible_device_indices:
            for g in self.gpus:
                if g.index in self.visible_device_indices:
                    return g
        return self.gpus[0]

    def summary(self) -> str:
        if not self.available:
            return f"[GPU Discovery] Nenhuma GPU NVIDIA detectada. Erro: {self.error or 'N/A'}. Modo CPU ativado."
        lines = [f"[GPU Discovery] {len(self.gpus)} GPU(s) NVIDIA detectada(s):"]
        for g in self.gpus:
            lines.append(
                f"  GPU {g.index}: {g.name} | Driver {g.driver_version} | "
                f"Mem {g.memory_used_mb:.0f}/{g.memory_total_mb:.0f} MB "
                f"(livre {g.memory_free_mb:.0f} MB) | "
                f"Util GPU {g.utilization_gpu_pct:.0f}% Mem {g.utilization_mem_pct:.0f}%"
            )
            if g.compute_capability:
                lines.append(f"    Hardware: {g.hw_summary()}")
        if self.visible_device_indices is not None:
            lines.append(f"  CUDA_VISIBLE_DEVICES={self.visible_device_indices}")
        pg = self.primary_gpu
        if pg:
            lines.append(f"  -> GPU primária em uso: GPU {pg.index} ({pg.name})")
            if pg.memory_total_mb < 14000:
                lines.append(
                    f"  [AVISO] GPU com {pg.memory_total_mb:.0f} MB de VRAM — "
                    "batch_size será ajustado automaticamente para evitar OOM."
                )
            if pg.compute_capability:
                lines.append(
                    f"  Arquitetura: {pg.architecture or 'unknown'} | "
                    f"BW: {pg.memory_bandwidth_gb_s or 0:.0f} GB/s | "
                    f"TF32={pg.has_tf32} BF16={pg.has_bf16} FlashAttn={pg.has_flash_attention}"
                )
        return "\n".join(lines)


def _parse_cuda_visible_devices() -> Optional[List[int]]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return None
    try:
        return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
    except Exception:
        return None


def _query_compute_capability() -> Dict[int, str]:
    """Consulta o compute capability de todas as GPUs via nvidia-smi."""
    cc_map: Dict[int, str] = {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            timeout=10,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    idx = int(parts[0])
                    cc = parts[1].strip()
                    if cc and cc not in ("N/A", "[N/A]", ""):
                        cc_map[idx] = cc
                except (ValueError, IndexError):
                    pass
    except Exception:
        pass
    return cc_map


def discover_gpus() -> GPUDiscoveryResult:
    """Roda nvidia-smi e coleta telemetria completa de todas as GPUs visíveis.

    Coleta adicionalmente: compute capability, arquitetura, largura de banda estimada
    e suporte a funcionalidades como TF32, BF16 e Flash Attention.
    """
    visible = _parse_cuda_visible_devices()
    # Compute capability (falha silenciosamente em drivers antigos)
    cc_map = _query_compute_capability()

    query_fields = (
        "index,name,driver_version,"
        "memory.total,memory.used,memory.free,"
        "utilization.gpu,utilization.memory,"
        "temperature.gpu,power.draw,power.limit"
    )
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            timeout=15,
        )
    except FileNotFoundError:
        return GPUDiscoveryResult(available=False, gpus=[], visible_device_indices=visible,
                                  error="nvidia-smi não encontrado no PATH")
    except subprocess.TimeoutExpired:
        return GPUDiscoveryResult(available=False, gpus=[], visible_device_indices=visible,
                                  error="nvidia-smi timeout (>15s)")
    except subprocess.CalledProcessError as exc:
        return GPUDiscoveryResult(available=False, gpus=[], visible_device_indices=visible,
                                  error=f"nvidia-smi retornou código {exc.returncode}")

    gpus: List[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            def _float(s: str) -> float:
                s = s.strip().replace("[N/A]", "").replace("N/A", "").strip()
                return float(s) if s else 0.0

            idx = int(parts[0])
            # Skip if not in CUDA_VISIBLE_DEVICES (when set)
            if visible is not None and idx not in visible:
                continue
            gpu = GPUInfo(
                index=idx,
                name=parts[1],
                driver_version=parts[2],
                memory_total_mb=_float(parts[3]),
                memory_used_mb=_float(parts[4]),
                memory_free_mb=_float(parts[5]),
                utilization_gpu_pct=_float(parts[6]),
                utilization_mem_pct=_float(parts[7]),
                temperature_c=_float(parts[8]) if len(parts) > 8 else None,
                power_draw_w=_float(parts[9]) if len(parts) > 9 else None,
                power_limit_w=_float(parts[10]) if len(parts) > 10 else None,
                compute_capability=cc_map.get(idx),
            )
            gpu.enrich_from_cc()
            gpus.append(gpu)
        except (ValueError, IndexError):
            continue

    if not gpus:
        return GPUDiscoveryResult(available=False, gpus=[], visible_device_indices=visible,
                                  error="nvidia-smi retornou saída mas nenhuma GPU parseada")
    return GPUDiscoveryResult(available=True, gpus=gpus, visible_device_indices=visible)
