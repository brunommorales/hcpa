"""
main.py — Ponto de entrada principal do HCPA Autotuner.

CLI:
    python -m src.main --stack pytorch|tensorflow|monai --mode base|opt --path <raiz>

Fluxo:
1. GPU Discovery (obrigatório)
2. Auditoria do espaço derivado
3. Carrega variante como ponto de partida
4. Executa treino com autoajuste online
5. Logging em CSV + checkpoint do controller
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .gpu_discovery import GPUDiscoveryResult, discover_gpus
from .gpu_monitor import GPUMonitor
from .derived_space import DerivedConfigSpace, build_derived_space, get_initial_config
from .controller import AutoTuneController, TuningAction
from .safety import SafetyManager, HealthSignal
from .logging_csv import CSVLogger
from .audit import generate_audit_report, diff_base_opt
from .backends import PyTorchBackend, TensorFlowBackend, MonaiBackend
from .backends.base import BackendBase
from .offline_knowledge import get_knowledge_base
from .multi_objective import ObjectiveWeights


def _recommend_batch_size(
    gpu_result: GPUDiscoveryResult,
    current: int,
    stack: str = "",
    mode: str = "",
) -> int:
    """Sugere batch_size por GPU baseado em memória, stack, modo e compute capability.

    Considera o uso de memória empírico por sample de cada variante:
      - pytorch_opt:    ~195 MB/sample (DALI + AMP + timm + EMA)
      - pytorch_base:   ~200 MB/sample (no DALI, menos otimizado)
      - tensorflow_*:   ~110 MB/sample
      - monai_*:        ~70  MB/sample (menor: sem backbone pesado, DALI eficiente)

    Mantém headroom de 20% e alinha em múltiplos de 8.
    """
    if not gpu_result.available or not gpu_result.primary_gpu:
        return current
    mem = gpu_result.primary_gpu.memory_total_mb or 0
    num_gpus = len(gpu_result.gpus) if gpu_result.gpus else 1
    if mem <= 0:
        return current

    # MB/sample empírico por stack+mode (baseado nos resultados históricos)
    # tensorflow_opt: medido empiricamente ~19 MB/sample em RTX4070 (batch=80 → peak 1553 MB).
    # Usando 25 MB/sample como margem de segurança (≈30% acima do observado).
    mb_per_sample_map = {
        ("pytorch", "opt"):   195,
        ("pytorch", "base"):  200,
        ("tensorflow", "opt"):   25,  # empírico RTX4070: ~19 MB/sample real
        ("tensorflow", "base"):  28,
        ("monai", "opt"):   72,
        ("monai", "base"):   75,
    }
    mb_per_sample = mb_per_sample_map.get((stack, mode), 130)

    # Headroom maior para GPUs com pouca VRAM (≤ 16 GB)
    if mem < 12000:
        safe_pct = 0.65
    elif mem < 20000:
        safe_pct = 0.75
    else:
        safe_pct = 0.80

    safe_vram_mb = mem * safe_pct
    target_per_gpu = int(safe_vram_mb / mb_per_sample)
    # Arredondar para múltiplo de 8
    target_per_gpu = max(8, (target_per_gpu // 8) * 8)
    # Limites globais
    target_per_gpu = max(8, min(256, target_per_gpu))

    print(
        f"[BatchRec] GPU {gpu_result.primary_gpu.name} | "
        f"VRAM {mem:.0f} MB | safe_vram {safe_vram_mb:.0f} MB | "
        f"mb/sample={mb_per_sample} | batch_rec={target_per_gpu} (era {current})"
    )
    return target_per_gpu


def parse_main_args():
    parser = argparse.ArgumentParser(
        description="HCPA Autotuner — wrapper de autoajuste online para PyTorch, TensorFlow e MONAI"
    )
    parser.add_argument(
        "--stack",
        type=str,
        required=True,
        choices=["pytorch", "tensorflow", "monai"],
        help="Stack de framework a usar",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["base", "opt"],
        help="Modo: base (sem otimizações) ou opt (otimizado)",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Diretório raiz da variante (ex: /path/to/pytorch_base)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./autotuner_output",
        help="Diretório de saída do autotuner (logs, checkpoints, CSV)",
    )
    parser.add_argument(
        "--enable-tuning",
        dest="enable_tuning",
        action="store_true",
        default=True,
        help="Habilita autoajuste online (padrão: habilitado)",
    )
    parser.add_argument(
        "--disable-tuning",
        dest="enable_tuning",
        action="store_false",
        help="Desabilita autoajuste (executa variante como está)",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=5.0,
        help="Intervalo de monitoramento GPU em segundos",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        default=False,
        help="Apenas executa a auditoria e mostra o espaço derivado, sem treinar",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Retoma de checkpoint do controller (se existir)",
    )
    # Permitir override de parâmetros do espaço derivado via CLI
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Overrides no formato key=value (ex: --override batch_size=64 lrate=1e-3)",
    )
    return parser.parse_args()


def _create_backend(stack: str, mode: str, path: Path) -> BackendBase:
    """Cria o backend correto para o stack."""
    backends = {
        "pytorch": PyTorchBackend,
        "tensorflow": TensorFlowBackend,
        "monai": MonaiBackend,
    }
    cls = backends[stack]
    return cls(variant_path=path, mode=mode)


def _parse_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parseia overrides key=value da CLI."""
    result: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            print(f"[WARN] Override ignorado (sem '='): {item}")
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        # Tentar parsear tipos
        if val.lower() in ("true", "false"):
            result[key] = val.lower() == "true"
        else:
            try:
                result[key] = int(val)
            except ValueError:
                try:
                    result[key] = float(val)
                except ValueError:
                    result[key] = val
    return result


def _parse_metrics_from_output(output: str, backend: BackendBase) -> Dict[str, Any]:
    """Extrai métricas da última época na saída do treino."""
    lines = output.strip().split("\n")
    # Parsear de trás para frente para pegar a última época
    for line in reversed(lines):
        metrics = backend.parse_epoch_metrics(line)
        if metrics:
            return metrics
    return {}


def _run_subprocess_epoch_by_epoch(
    cmd: List[str],
    backend: BackendBase,
    controller: AutoTuneController,
    safety: SafetyManager,
    gpu_monitor: Optional[GPUMonitor],
    csv_logger: CSVLogger,
    config: Dict[str, Any],
    output_dir: Path,
    early_stop_cfg: Optional[Dict[str, Any]] = None,
    gpu_mem_total_mb: float = 0.0,
) -> Dict[str, Any]:
    """
    Executa o treino como subprocess e monitora saída para autoajuste.

    Para backends que treinam em loop interno (um único processo), monitoramos
    a saída e coletamos métricas. Ajustes que requerem restart não são
    aplicados mid-run; apenas ajustes online (LR, label_smoothing, etc.)
    são comunicados via sinais ou salvos para próxima execução.
    """
    print(f"\n[Runner] Comando: {' '.join(cmd)}")
    print(f"[Runner] Config inicial: {json.dumps({k: str(v) for k, v in config.items()}, indent=2)}")

    start_time = time.time()
    epoch_count = 0
    _pending_epoch = 0  # Para Keras: "Epoch N/M" vem numa linha, métricas na próxima
    final_metrics: Dict[str, Any] = {}
    had_oom = False
    had_nan = False
    requested_stop = False  # se true, sair cedo sem reentrar em wait infinito
    rb_reason = None

    best_metric = -float("inf")
    plateau = 0
    actions_applied = False
    _batch_recalib_done = False  # flag: recalibração de batch já feita neste run
    tuning_file = output_dir / "tuning_actions.json"
    if tuning_file.exists():
        tuning_file.unlink(missing_ok=True)
    es_enabled = bool(early_stop_cfg)
    es_patience = int(early_stop_cfg.get("patience", 8)) if es_enabled else 0
    es_min_delta = float(early_stop_cfg.get("min_delta", 1e-4)) if es_enabled else 0.0
    es_min_epochs = int(early_stop_cfg.get("min_epochs", 40)) if es_enabled else 0

    try:
        env_vars = {**os.environ, "PYTHONUNBUFFERED": "1", "HCPA_TUNING_FILE": str(tuning_file)}
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(backend.variant_path),
            env=env_vars,
            start_new_session=True,
        )

        output_log_path = output_dir / "train_output.log"
        log_file = open(output_log_path, "w", encoding="utf-8")

        for line in process.stdout:
            line = line.rstrip("\n")
            log_file.write(line + "\n")
            log_file.flush()
            print(line)  # echo para stdout

            # Tentar parsear métricas da linha
            metrics = backend.parse_epoch_metrics(line)

            # Detectar OOM na saída (independente de métricas)
            if "CUDA out of memory" in line or "OOM" in line.upper():
                signal_oom = HealthSignal(
                    epoch=epoch_count, loss=float("nan"), is_oom=True
                )
                safety.record_signal(signal_oom)
                print(f"  [SAFETY] OOM detectado na época {epoch_count}!")
                had_oom = True

            # NaN/Inf detectado no texto (incluindo linhas de progresso step-a-step,
            # ex: "1/138 ... - loss: inf") — NÃO matar o processo.
            # Registrar sinal, escrever payload de ROLLBACK no tuning_file e
            # deixar o callback in-process TuningFileCallback abortar a época e
            # restaurar o checkpoint sem precisar de restart.
            # O Slurm gerencia o ciclo de vida do job.
            _ll = line.lower()
            _has_nan_text = " nan" in _ll or "nan " in _ll
            # Captura "loss: inf" nas linhas de progresso step (contêm ETA ou ProgressBar)
            _has_inf_text = (
                "loss: inf" in _ll
                or "loss:inf" in _ll
                or "- inf -" in _ll
                or (" inf" in _ll and "eta:" in _ll)  # step progress line com inf
            )
            if (_has_nan_text or _has_inf_text) and not had_nan:
                had_nan = True
                signal_nan = HealthSignal(
                    epoch=epoch_count, loss=float("nan"), is_nan=True
                )
                safety.record_signal(signal_nan)
                safety.request_rollback("nan_detected_stream")
                _trigger = "Inf" if _has_inf_text else "NaN"
                print(
                    f"  [SAFETY] {_trigger} detectado na saída (época {epoch_count}). "
                    "Escrevendo payload de ROLLBACK no tuning_file — TuningFileCallback "
                    "abortará a época atual e restaurará o último checkpoint na seguinte. "
                    "O Slurm controla o ciclo de vida do job."
                )
                # Escrever config de recuperação para callbacks in-process lerem
                recovery_config: Dict[str, Any] = {}
                lr_key_nan = None
                for lk in ("lrate", "learning_rate", "fine_tune_lr"):
                    if lk in config:
                        lr_key_nan = lk
                        break
                if lr_key_nan:
                    current_lr_nan = float(config.get(lr_key_nan) or 1e-5)
                    recovery_config[lr_key_nan] = max(1e-6, current_lr_nan * 0.1)
                recovery_config["mixup_alpha"] = 0.0
                recovery_config["cutmix_alpha"] = 0.0
                recovery_config["label_smoothing"] = 0.0
                recovery_config["clip_grad_norm"] = 0.1
                recovery_config["grad_clip_norm"] = 0.1
                # Construir payload de recuperação incluindo rollback para best/last
                ckpt_root = Path(config.get("results_dir", config.get("results", output_dir))) / "checkpoints"
                best_ckpt = ckpt_root / "best.ckpt"
                last_ckpt = ckpt_root / "last.ckpt"
                rollback_entry = {
                    "mode": "best",
                    "reason": "nan_detected",
                    "path": str(best_ckpt if best_ckpt.exists() else last_ckpt),
                }
                nan_payload = {
                    "epoch": epoch_count,
                    "timestamp": time.time(),
                    "payload_id": time.time(),
                    "rollback": rollback_entry,
                    "config": recovery_config,
                    "actions": [
                        "NAN_RECOVERY: LR*0.1, mixup=0, cutmix=0, ls=0, clip=0.1",
                        f"ROLLBACK->{rollback_entry['path']}",
                    ],
                }
                try:
                    with open(tuning_file, "w", encoding="utf-8") as tf_n:
                        json.dump(nan_payload, tf_n, indent=2)
                except Exception as exc_w:
                    print(f"[WARN] Falha ao gravar tuning file de recuperação NaN: {exc_w}")

            if not metrics:
                continue

            # Keras pattern: "Epoch N/M" numa linha, métricas na próxima
            if "epoch" in metrics and "_is_metrics_line" not in metrics:
                _pending_epoch = metrics["epoch"]
                if "train_loss" not in metrics:
                    continue  # linha apenas com epoch header, esperar métricas

            # Métricas sem epoch (Keras): injetar o epoch pendente
            if "epoch" not in metrics and metrics.get("_is_metrics_line"):
                metrics["epoch"] = _pending_epoch

            metrics.pop("_is_metrics_line", None)

            if "epoch" in metrics and "train_loss" in metrics:
                epoch_count = metrics["epoch"]
                train_loss = metrics.get("train_loss")
                val_loss = metrics.get("val_loss")
                val_auc = metrics.get("val_auc")
                throughput = metrics.get("throughput", 0.0)
                lr = metrics.get("lr", 0.0)
                epoch_time = metrics.get("epoch_time_sec")
                val_throughput = metrics.get("val_throughput")

                # Só processar quando já temos validação; linhas parciais (sem val_loss)
                # aparecem no início do epoch e geravam falso positivo de NaN.
                if train_loss is None:
                    continue
                if val_loss is None:
                    continue

                def _is_nan(x: Any) -> bool:
                    try:
                        return math.isnan(float(x))
                    except Exception:
                        return False

                def _is_inf(x: Any) -> bool:
                    try:
                        return math.isinf(float(x))
                    except Exception:
                        return False

                # Métricas válidas (não NaN/Inf)? Resetar had_nan para re-armar
                # detecção de futuros episódios de NaN/Inf após recuperação.
                if had_nan and not _is_nan(train_loss) and not _is_nan(val_loss) and \
                        not _is_inf(train_loss) and not _is_inf(val_loss):
                    had_nan = False
                    print(f"  [SAFETY] Época {epoch_count} com métricas válidas após NaN/Inf — "
                          "detector re-armado para futuras épocas.")

                # Se métricas já vierem com NaN ou Inf ao nível de epoch-summary,
                # registrar e escrever rollback. NÃO terminar o processo.
                if _is_nan(train_loss) or _is_nan(val_loss) or _is_inf(train_loss) or _is_inf(val_loss):
                    had_nan = True
                    signal_nan = HealthSignal(
                        epoch=epoch_count, loss=train_loss, val_loss=val_loss, is_nan=True
                    )
                    safety.record_signal(signal_nan)
                    safety.request_rollback("nan_in_metrics")
                    # Gravar tuning_file com rollback best/last e mitigação conservadora
                    ckpt_root = Path(config.get("results_dir", config.get("results", output_dir))) / "checkpoints"
                    best_ckpt = ckpt_root / "best.ckpt"
                    last_ckpt = ckpt_root / "last.ckpt"

                    def _ckpt_prefix_available(prefix: Path) -> bool:
                        return (
                            prefix.with_suffix(".index").exists()
                            or any(prefix.parent.glob(prefix.name + ".data-*"))
                        )

                    _best_exists = _ckpt_prefix_available(best_ckpt)
                    rollback_entry = {
                        "mode": "best",
                        "reason": "nan_in_metrics",
                        "path": str(best_ckpt if _best_exists else last_ckpt),
                    }

                    # Reduzir LR após NaN para evitar novo colapso
                    _lr_key_nan = next((k for k in ("lrate", "learning_rate", "fine_tune_lr") if k in config), None)
                    _lr_now_nan = float(metrics.get("lr") or config.get(_lr_key_nan) or 0.0)
                    _recover_lr_nan = None
                    if _lr_key_nan:
                        if _lr_now_nan > 0:
                            _recover_lr_nan = max(1e-4, _lr_now_nan * 0.2)
                        else:
                            _recover_lr_nan = max(1e-4, float(config.get(_lr_key_nan, 1e-3)) * 0.5)

                    nan_payload = {
                        "epoch": epoch_count,
                        "timestamp": time.time(),
                        "payload_id": time.time(),
                        "rollback": rollback_entry,
                        "config": {
                            "mixup_alpha": 0.0,
                            "cutmix_alpha": 0.0,
                            "label_smoothing": 0.0,
                            "clip_grad_norm": 0.1,
                            "grad_clip_norm": 0.1,
                            **({_lr_key_nan: _recover_lr_nan} if _recover_lr_nan and _lr_key_nan else {}),
                        },
                        "actions": [
                            "NAN_METRICS: rollback best/last, zero mixup/cutmix/ls, clip=0.1",
                            f"ROLLBACK->{rollback_entry['path']}",
                        ],
                    }
                    try:
                        with open(tuning_file, "w", encoding="utf-8") as tf_n:
                            json.dump(nan_payload, tf_n, indent=2)
                    except Exception as exc_w:
                        print(f"[WARN] Falha ao gravar tuning file de recuperação NaN: {exc_w}")
                    print(
                        f"  [SAFETY] NaN nas métricas (época {epoch_count}): "
                        f"train_loss={train_loss} val_loss={val_loss}. "
                        f"Rollback -> {rollback_entry['path']}"
                    )
                    # Continua para permitir rollback/config ajustados nesta época

                # ─── Detector de colapso por LR excessivamente baixo ───────────────────
                # ReduceLROnPlateau (interno ao script de treino) pode reduzir o LR até
                # min_lr=1e-6 sem intervenção do autotuner, causando colapso silencioso.
                # Quando lr < 5μ E val_AUC ≤ 0.5010 (classificador aleatório), forçar rollback.
                _lr_now = float(metrics.get("lr") or 0.0)
                _auc_now = float(val_auc) if val_auc is not None else 1.0
                if (
                    not had_nan
                    and epoch_count > 20
                    and _lr_now > 0
                    and _lr_now < 5e-6           # LR efetivamente zerado
                    and not math.isnan(_auc_now)
                    and not math.isinf(_auc_now)
                    and _auc_now <= 0.5010       # AUC colapsado
                ):
                    had_nan = True  # prevenir re-disparo
                    safety.request_rollback("lr_crash_auc_flat")
                    print(
                        f"  [SAFETY] LR crash detectado (época {epoch_count}): "
                        f"lr={_lr_now:.2e}, val_auc={_auc_now:.4f} ≤ 0.5010. "
                        "Escrevendo rollback para best.ckpt."
                    )
                    ckpt_root_lc = Path(config.get("results_dir", config.get("results", output_dir))) / "checkpoints"
                    best_ckpt_lc = ckpt_root_lc / "best.ckpt"
                    last_ckpt_lc = ckpt_root_lc / "last.ckpt"
                    _best_exists_lc = _ckpt_prefix_available(best_ckpt_lc)
                    _lr_key_crash = next((k for k in ("lrate", "learning_rate", "fine_tune_lr") if k in config), None)
                    _orig_lr_crash = float(config.get(_lr_key_crash) or 1e-3) if _lr_key_crash else 1e-3
                    _recovery_lr_crash = max(1e-4, _orig_lr_crash * 0.1)
                    lr_crash_payload = {
                        "epoch": epoch_count,
                        "timestamp": time.time(),
                        "payload_id": time.time(),
                            "rollback": {
                                "mode": "best",
                                "reason": "lr_crash_auc_flat",
                                "path": str(best_ckpt_lc if _best_exists_lc else last_ckpt_lc),
                            },
                        "config": {
                            (_lr_key_crash or "lrate"): _recovery_lr_crash,
                            "mixup_alpha": 0.0,
                            "cutmix_alpha": 0.0,
                        },
                        "actions": [f"LR_CRASH: lr={_lr_now:.2e}→recovery {_recovery_lr_crash:.2e}, rollback best.ckpt"],
                    }
                    try:
                        with open(tuning_file, "w", encoding="utf-8") as tf_lc:
                            json.dump(lr_crash_payload, tf_lc, indent=2)
                    except Exception as exc_lc:
                        print(f"[WARN] Falha ao gravar tuning file LR crash: {exc_lc}")

                # Coletar snapshot GPU
                gpu_snap = None
                if gpu_monitor:
                    gpu_snap_obj = gpu_monitor.collect_once()
                    if gpu_snap_obj:
                        gpu_snap = {
                            "memory_used_mb": gpu_snap_obj.memory_used_mb,
                            "memory_total_mb": gpu_snap_obj.memory_total_mb,
                            "utilization_gpu_pct": gpu_snap_obj.utilization_gpu_pct,
                            "temperature_c": gpu_snap_obj.temperature_c,
                            "power_draw_w": gpu_snap_obj.power_draw_w,
                        }

                # Preferir gpu_mem_peak_mb reportado pelo Keras sobre nvidia-smi:
                # é medido dentro do processo TF e captura o pico exato da época.
                # O nvidia-smi snapshot é assíncrono e pode ser coletado fora do pico.
                _keras_peak_mb = metrics.get("gpu_mem_peak_mb")
                if _keras_peak_mb is not None:
                    try:
                        _kp = float(_keras_peak_mb)
                        if not math.isnan(_kp) and _kp > 0:
                            _total_mb_ref = (
                                float(gpu_snap["memory_total_mb"]) if gpu_snap and gpu_snap.get("memory_total_mb")
                                else (gpu_mem_total_mb if gpu_mem_total_mb > 0 else _kp / 0.85)
                            )
                            if gpu_snap:
                                gpu_snap["memory_used_mb"] = _kp
                            else:
                                gpu_snap = {
                                    "memory_used_mb": _kp,
                                    "memory_total_mb": _total_mb_ref,
                                    "utilization_gpu_pct": 0.0,
                                    "temperature_c": 0.0,
                                    "power_draw_w": 0.0,
                                }
                    except (ValueError, TypeError):
                        pass

                # Recalibração de batch: após épocas iniciais estáveis, compara
                # uso real de memória com estimativa inicial e salva batch ótimo
                # para o próximo restart (não muda batch mid-training).
                if (
                    not _batch_recalib_done
                    and epoch_count >= 10
                    and not had_nan
                    and not had_oom
                    and config.get("batch_size")
                    and gpu_snap
                    and gpu_snap.get("memory_total_mb", 0) > 0
                ):
                    _peak_rc = float(gpu_snap.get("memory_used_mb") or 0)
                    _total_rc = float(gpu_snap.get("memory_total_mb") or 0)
                    if _peak_rc > 0 and _total_rc > 0:
                        _actual_bs_rc = int(config["batch_size"])
                        _mb_per_s_rc = _peak_rc / _actual_bs_rc
                        _safe_vram_rc = _total_rc * 0.75
                        _optimal_bs_rc = int(_safe_vram_rc / _mb_per_s_rc / 8) * 8
                        _optimal_bs_rc = max(8, min(256, _optimal_bs_rc))
                        _batch_recalib_done = True
                        if _optimal_bs_rc > _actual_bs_rc * 1.5:  # >50% maior
                            config["_recalibrated_batch"] = _optimal_bs_rc
                            config["_actual_mb_per_sample"] = round(_mb_per_s_rc, 2)
                            controller.set_config(config)
                            print(
                                f"[BatchRecalib] E{epoch_count}: peak={_peak_rc:.0f}MB, "
                                f"{_mb_per_s_rc:.1f}MB/sample, "
                                f"optimal_batch={_optimal_bs_rc} (atual={_actual_bs_rc}). "
                                f"→ Aplicado via _recalibrated_batch no próximo restart."
                            )

                # Chamar controller para decidir ajustes
                actions = controller.on_epoch_end(
                    epoch=epoch_count,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_auc=val_auc,
                    throughput=throughput,
                    gpu_mem_used_mb=float(gpu_snap.get("memory_used_mb") or 0) if gpu_snap else 0.0,
                    gpu_mem_total_mb=float(gpu_snap.get("memory_total_mb") or 0) if gpu_snap else 0.0,
                )

                # Checar rollback solicitado pelo safety (não reinicia; injeta rollback via tuning_file)
                rb_flag, rb_reason = safety.consume_rollback_flag()
                if rb_flag:
                    rollback_cfg = safety.get_rollback_config() or controller.config
                    controller.set_config(rollback_cfg)
                    ckpt_root = Path(rollback_cfg.get("results_dir", rollback_cfg.get("results", output_dir))) / "checkpoints"
                    best_ckpt = ckpt_root / "best.ckpt"
                    rollback_payload = {
                        "epoch": epoch_count,
                        "timestamp": time.time(),
                        "rollback": {"mode": "best", "reason": rb_reason, "path": str(best_ckpt)},
                        "config": rollback_cfg,
                        "actions": [f"ROLLBACK:{rb_reason}"],
                    }
                    try:
                        with open(tuning_file, "w", encoding="utf-8") as tf_f:
                            json.dump(rollback_payload, tf_f, indent=2)
                    except Exception as exc:
                        print(f"[WARN] Falha ao gravar tuning file de rollback: {exc}")
                    safety.reset()
                    # Não registrar mais nada desta época; segue para próxima
                    continue

                if actions:
                    actions_applied = True
                    # persist tuning intents for in-process callbacks
                    tuning_payload = {
                        "epoch": epoch_count,
                        "timestamp": time.time(),
                        "config": {k: controller.config.get(k) for k in controller.config.keys()
                                   if k in ("lrate", "learning_rate", "mixup_alpha", "label_smoothing",
                                            "clip_grad_norm", "grad_clip_norm")},
                        "actions": [a.reason for a in actions],
                    }
                    try:
                        with open(tuning_file, "w", encoding="utf-8") as tf_f:
                            json.dump(tuning_payload, tf_f, indent=2)
                    except Exception as exc:
                        print(f"[WARN] Falha ao gravar tuning file: {exc}")

                # Logar ações
                action_strs = [str(a) for a in actions]
                for a_str in action_strs:
                    print(f"  >>> {a_str}")

                # Logar no CSV
                # ── extrair info multi-objetivo e convergência do controller ──────
                _mo_score = controller.latest_composite_score
                _composite_info: Optional[Dict[str, Any]] = None
                if _mo_score is not None:
                    _composite_info = {
                        "composite": getattr(_mo_score, "composite", None),
                        "auc_score": getattr(_mo_score, "auc_score", None),
                        "throughput_score": getattr(_mo_score, "throughput_score", None),
                        "memory_score": getattr(_mo_score, "memory_score", None),
                    }

                _conv_state = controller.latest_convergence_state
                _convergence_info: Optional[Dict[str, Any]] = None
                if _conv_state is not None:
                    _convergence_info = {
                        "phase": getattr(_conv_state, "phase", None),
                        "predicted_final_auc": getattr(_conv_state, "predicted_final_auc", None),
                        "tau": getattr(_conv_state, "tau", None),
                    }

                csv_logger.log_epoch(
                    epoch=epoch_count,
                    stage=metrics.get("stage", "train"),
                    train_metrics={
                        "loss": train_loss,
                        "auc": metrics.get("train_auc"),
                        "throughput": throughput,
                        "elapsed_s": epoch_time,
                    },
                    val_metrics={
                        "loss": val_loss,
                        "auc": val_auc,
                        "throughput": val_throughput,
                        "elapsed_s": None,
                    },
                    lr=lr,
                    gpu_snapshot=gpu_snap,
                    tuning_actions=action_strs,
                    config_snapshot=controller.config,
                    total_train_time_s=time.time() - start_time,
                    composite_score_info=_composite_info,
                    convergence_info=_convergence_info,
                )

                # Salvar estado do controller periodicamente
                safety.save_controller_state(controller.get_state())

                final_metrics = metrics

                # Early stopping (plateau em val_auc)
                if es_enabled and not math.isnan(val_auc):
                    if val_auc > best_metric + es_min_delta:
                        best_metric = val_auc
                        plateau = 0
                    else:
                        plateau += 1
                    if epoch_count >= es_min_epochs and plateau >= es_patience:
                        print(f"[EARLY-STOP] val_auc plateau por {plateau} épocas (best={best_metric:.4f}). Encerrando treino.")
                        process.terminate()
                        break

        # Garantir encerramento simples: espere o processo sair (SLURM mata o job se ele travar)
        if requested_stop:
            try:
                process.terminate()
            except Exception:
                pass
        process.wait()
        log_file.close()

        if process.returncode != 0 and not requested_stop:
            print(f"[Runner] Processo terminou com código {process.returncode}")
        else:
            print(f"[Runner] Treino concluído com sucesso.")

    except KeyboardInterrupt:
        print("\n[Runner] Interrompido pelo usuário. Salvando estado...")
        safety.save_controller_state(controller.get_state())
        process.kill()
    except Exception as exc:
        print(f"[Runner] Erro: {exc}")
        safety.save_controller_state(controller.get_state())

    elapsed = time.time() - start_time
    final_metrics["total_time_s"] = elapsed
    return {
        "metrics": final_metrics,
        "returncode": process.returncode if 'process' in locals() else -1,
        "had_oom": had_oom,
        "had_nan": had_nan,
        "log_path": output_log_path,
        "actions_applied": actions_applied,
        "rollback_triggered": requested_stop,
        "rollback_reason": rb_reason if 'rb_reason' in locals() else None,
    }


def main():
    args = parse_main_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ═══ ETAPA 0: GPU Discovery (OBRIGATÓRIO) ═══
    print("=" * 70)
    print("ETAPA 0: GPU DISCOVERY")
    print("=" * 70)
    gpu_result = discover_gpus()
    print(gpu_result.summary())

    if not gpu_result.available:
        print("\n[AVISO] Nenhuma GPU NVIDIA detectada. Autoajuste agressivo DESATIVADO.")
        print("[AVISO] Execução continuará em modo CPU (se o script suportar).\n")
        enable_tuning = False
        gpu_monitor = None
    else:
        enable_tuning = args.enable_tuning
        gpu_idx = gpu_result.primary_gpu.index if gpu_result.primary_gpu else 0
        gpu_monitor = GPUMonitor(gpu_index=gpu_idx, interval_s=args.monitor_interval)

    # ═══ ETAPA 1: Auditoria ═══
    print("\n" + "=" * 70)
    print("ETAPA 1: AUDITORIA DAS VARIANTES")
    print("=" * 70)
    report = generate_audit_report()
    print(report)

    # Salvar relatório
    report_path = output_dir / "audit_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Auditoria salva em {report_path}]")

    # ── Sub-etapa 1b: Relatório do conhecimento offline ──────────────────
    print("\n" + "-" * 70)
    print("BASE DE CONHECIMENTO OFFLINE")
    print("-" * 70)
    try:
        kb = get_knowledge_base()
        gpu_name_str = ""
        if gpu_result.available and gpu_result.primary_gpu:
            gpu_name_str = gpu_result.primary_gpu.name or ""
        kb_report = kb.summary_report(gpu_name_str)
        print(kb_report)
        kb_path = output_dir / "offline_knowledge_report.txt"
        with open(kb_path, "w", encoding="utf-8") as f:
            f.write(kb_report)
        print(f"[Conhecimento offline salvo em {kb_path}]")
    except Exception as exc:
        print(f"[OfflineKB] Aviso: não foi possível gerar relatório: {exc}")

    # ═══ Espaço de Configuração Derivado ═══
    print("\n" + "=" * 70)
    print("ESPAÇO DE CONFIGURAÇÃO DERIVADO")
    print("=" * 70)
    space = build_derived_space(args.stack)
    print(space.summary())

    # Salvar schema
    schema_path = output_dir / "derived_space_schema.json"
    schema = {}
    for name, spec in space.params.items():
        schema[name] = {
            "type": spec.param_type,
            "base_value": spec.base_value,
            "opt_value": spec.opt_value,
            "range": [spec.range_min, spec.range_max] if spec.range else None,
            "choices": spec.choices,
            "tunable_online": spec.tunable_online,
            "requires_restart": spec.requires_restart,
            "source_variants": spec.source_variants,
            "description": spec.description,
        }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, default=str)
    print(f"\n[Schema salvo em {schema_path}]")

    # Se audit-only, parar aqui
    if args.audit_only:
        print("\n[audit-only] Auditoria concluída. Saindo sem treinar.")
        return

    # ═══ ETAPA 2–3: Inicializar backend, controller e executar ═══
    print("\n" + "=" * 70)
    print(f"ETAPA 2-3: EXECUÇÃO — stack={args.stack} mode={args.mode}")
    print("=" * 70)

    variant_path = Path(args.path)
    backend = _create_backend(args.stack, args.mode, variant_path)

    if not backend.validate():
        print(f"[ERRO] Script de treino não encontrado: {backend.get_entry_point()}")
        print("[ERRO] Encerrando autotuner — o Slurm registrará a falha do job.")
        raise FileNotFoundError(
            f"Script de treino não encontrado: {backend.get_entry_point()}"
        )

    # Inicializar safety + controller
    checkpoint_dir = output_dir / "checkpoints"
    safety = SafetyManager(checkpoint_dir=checkpoint_dir)

    # Determinar GPU name e capacidades para knowledge base
    gpu_name_for_kb = ""
    gpu_cc_for_ctrl: Optional[str] = None
    gpu_vram_for_ctrl: Optional[float] = None
    if gpu_result.available and gpu_result.primary_gpu:
        gpu_name_for_kb = gpu_result.primary_gpu.name or ""
        gpu_cc_for_ctrl = getattr(gpu_result.primary_gpu, "compute_capability", None)
        gpu_vram_for_ctrl = getattr(gpu_result.primary_gpu, "memory_total_mb", None)

    # Extrair total_epochs do espaço derivado (sempre 200 nas variantes atuais)
    total_epochs = 200
    ep_spec = space.get("epochs")
    if ep_spec:
        total_epochs = int(ep_spec.base_value or 200)

    controller = AutoTuneController(
        space=space,
        safety=safety,
        gpu_monitor=gpu_monitor,
        enable_tuning=enable_tuning,
        # Parâmetros v2: contexto para knowledge base e multi-objetivo
        stack=args.stack,
        mode=args.mode,
        gpu_name=gpu_name_for_kb,
        total_epochs=total_epochs,
        # Capacidades de hardware para adaptar thresholds automaticamente
        gpu_compute_capability=gpu_cc_for_ctrl,
        gpu_vram_mb=gpu_vram_for_ctrl,
    )

    # Inicializar config
    if args.resume:
        saved_state = safety.load_controller_state()
        if saved_state:
            controller.load_state(saved_state)
            print("[Resume] Estado do controller carregado de checkpoint.")
        else:
            controller.initialize(args.mode)
            print("[Resume] Nenhum checkpoint encontrado; inicializando do zero.")
    else:
        controller.initialize(args.mode)

    # Aplicar overrides
    overrides = _parse_overrides(args.override)
    if overrides:
        current = controller.config
        for key, val in overrides.items():
            if key in space.params:
                current[key] = val
                print(f"[Override] {key} = {val}")
                if key == "batch_size":
                    current["_batch_user_override"] = True
            else:
                print(f"[Override WARN] '{key}' não está no espaço derivado — ignorado")
        controller.set_config(current)

    # Injetar paths e batch inicial adaptado à GPU
    config = controller.config

    # Recomenda batch inicial baseado na GPU (apenas se não veio override explícito)
    if "batch_size" in config and config.get("_batch_user_override") is None:
        _recalib_bs = config.pop("_recalibrated_batch", None)
        if _recalib_bs:
            _recalib_mb_s = config.pop("_actual_mb_per_sample", "?")
            print(f"[BatchRec] Usando batch recalibrado da execução anterior: {_recalib_bs} "
                  f"({_recalib_mb_s} MB/sample observado)")
            config["batch_size"] = int(_recalib_bs)
        else:
            config["batch_size"] = _recommend_batch_size(
                gpu_result, int(config["batch_size"]), args.stack, args.mode
            )

    # Descobrir automaticamente o caminho dos TFRecords. Suporta dois layouts:
    #   1) <variant>/data/all-tfrec (layout original do repo)
    #   2) <variant>/data           (quando montamos diretamente a pasta all-tfrec)
    if "tfrec_dir" not in config or not Path(config.get("tfrec_dir", "")).is_dir():
        tfrec_candidates = []
        if config.get("tfrec_dir"):
            tfrec_candidates.append(Path(config["tfrec_dir"]))
        tfrec_candidates.extend([
            variant_path / "data" / "all-tfrec",
            variant_path / "data",
        ])

        for cand in tfrec_candidates:
            if cand.is_dir():
                config["tfrec_dir"] = str(cand)
                break
        else:
            print("[WARN] Nenhum diretório de TFRecords encontrado; mantenho valor padrão.")
    if "results" not in config and "results_dir" not in config:
        config["results"] = str(output_dir / "results")
        config["results_dir"] = str(output_dir / "results")
    Path(config.get("results", config.get("results_dir", output_dir / "results"))).mkdir(
        parents=True, exist_ok=True
    )
    controller.set_config(config)

    # Construir comando
    config = controller.config
    cmd = backend.build_command(config)

    # Iniciar monitor GPU
    if gpu_monitor:
        gpu_monitor.start()

    # Iniciar CSV logger
    csv_logger = CSVLogger(output_dir / "autotuner_log.csv")
    csv_logger.open()

    # Executar treino (semelhante aos scripts SLURM originais).
    # Permitimos ajustar via env AUTOTUNER_MAX_RESTARTS; default=1 para habilitar rollback/restart automático.
    max_restarts = int(os.environ.get("AUTOTUNER_MAX_RESTARTS", "1"))
    restart_count = 0
    run_idx = 0
    final_result = None
    gpu_mem_total = gpu_result.primary_gpu.memory_total_mb if gpu_result.available and gpu_result.primary_gpu else 0

    while True:
        run_idx += 1
        if gpu_monitor:
            gpu_monitor.clear()  # métricas só deste run
        print(f"\n[RUN {run_idx}] iniciando com batch_size={config.get('batch_size')} tfrec_dir={config.get('tfrec_dir')}")
        result = _run_subprocess_epoch_by_epoch(
            cmd=backend.build_command(config),
            backend=backend,
            controller=controller,
            safety=safety,
            gpu_monitor=gpu_monitor,
            csv_logger=csv_logger,
            config=config,
            output_dir=output_dir,
            # Nunca injetar early-stop pelo autotuner: deixar o script de treino
            # ou o Slurm controlarem o encerramento.
            early_stop_cfg=None,
            gpu_mem_total_mb=float(gpu_mem_total),
        )

        final_result = result

        # Sincronizar config corrente do controller (pode ter mudado por ações)
        config = controller.config

        peak_mem = gpu_monitor.peak_memory_mb() if gpu_monitor else 0.0

        # Checar sinais para restart seguro
        needs_restart = False
        restart_reason = None
        if result.get("rollback_triggered"):
            rollback_cfg = safety.get_rollback_config()
            if rollback_cfg:
                print(f"[Restart] Rollback solicitado ({result.get('rollback_reason')}); restaurando última config estável.")
                controller.set_config(rollback_cfg)
                config = controller.config
            needs_restart = True
            restart_reason = result.get("rollback_reason") or "rollback"
            safety.reset()
        if result.get("had_oom"):
            # reduzir batch e tentar de novo
            old_bs = int(config.get("batch_size", 0) or 0)
            if old_bs > 32:
                new_bs = max(32, int(old_bs * 0.75) // 8 * 8)
                if new_bs < old_bs:
                    print(f"[Restart] OOM detectado: reduzindo batch {old_bs} -> {new_bs}")
                    config["batch_size"] = new_bs
                    needs_restart = True
                    restart_reason = "oom"
        if result.get("had_nan"):
            # rollback simples: reduzir lr
            lr = float(config.get("lrate", 0) or 0)
            if lr > 0:
                new_lr = lr * 0.5
                print(f"[Restart] NaN detectado: reduzindo lrate {lr} -> {new_lr}")
                config["lrate"] = new_lr
            # Mitigar instabilidade: zerar mixup/cutmix e reduzir label_smoothing
            if "mixup_alpha" in config:
                config["mixup_alpha"] = 0.0
            if "cutmix_alpha" in config:
                config["cutmix_alpha"] = 0.0
            if "label_smoothing" in config:
                try:
                    current_ls = float(config["label_smoothing"])
                    config["label_smoothing"] = max(0.0, min(current_ls, 0.02))
                except Exception:
                    config["label_smoothing"] = 0.0
            # Desativar mixed precision se existir na config (só TF usa)
            if "mixed_precision" in config:
                config["mixed_precision"] = False
            needs_restart = True
            restart_reason = restart_reason or "nan"

        # Nota: ações de tuning online (LR, label_smoothing, etc.) são comunicadas
        # via tuning_actions.json e aplicadas via callbacks sem reiniciar o treino.
        # Não forçamos restart por causa de ações aplicadas — o Slurm controla o job.

        # Informativo: registrar uso de VRAM para referência futura.
        # Não reiniciar para escalar batch — isso violaria o princípio de não-reinício.
        if (result.get("returncode", 1) == 0
                and not result.get("had_oom")
                and not result.get("had_nan")
                and gpu_monitor
                and gpu_mem_total > 0
                and config.get("batch_size")):
            mem_ratio = peak_mem / gpu_mem_total if gpu_mem_total else 0
            old_bs = int(config["batch_size"])
            if mem_ratio < 0.60 and old_bs < 256:
                new_bs = min(256, max(old_bs + 8, int(old_bs * 1.25) // 8 * 8))
                print(
                    f"[Info] VRAM subutilizado ({mem_ratio:.1%}); "
                    f"batch_size poderia ser aumentado de {old_bs} → {new_bs} em próximas execuções."
                )
                # Salvar a sugestão no estado do controller para uso em retomadas
                config["_suggested_next_batch"] = new_bs
                controller.set_config(config)

        if needs_restart and restart_count < max_restarts:
            restart_count += 1
            controller.set_config(config)
            # Aplicar batch recalibrado (medido no run anterior) se disponível
            _rb_batch = config.pop("_recalibrated_batch", None)
            if _rb_batch:
                _rb_mb_s = config.pop("_actual_mb_per_sample", "?")
                old_bs_rb = config.get("batch_size", "?")
                config["batch_size"] = int(_rb_batch)
                controller.set_config(config)
                print(f"[Restart] Batch recalibrado aplicado: {old_bs_rb} → {_rb_batch} ({_rb_mb_s} MB/sample)")
            print(f"[Restart] Relaunching (motivo={restart_reason or 'unknown'}) | restart {restart_count}/{max_restarts}")
            safety.save_controller_state(controller.get_state())
            continue  # relança com nova config

        break  # sai do loop se não precisa ou esgotou restarts

    # Parar monitor e fechar logger
    if gpu_monitor:
        gpu_monitor.stop()
    csv_logger.close()

    # ═══ ETAPA 4: Relatório final ═══
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL")
    print("=" * 70)
    print(controller.summary())

    # Salvar estado final
    safety.save_controller_state(controller.get_state())

    # Sumário de GPU
    if gpu_monitor:
        print(f"\n[GPU Monitor] Peak memory: {gpu_monitor.peak_memory_mb():.0f} MB")
        print(f"[GPU Monitor] Avg utilization: {gpu_monitor.avg_utilization():.1f}%")
        print(f"[GPU Monitor] Total snapshots: {len(gpu_monitor.snapshots)}")

    # ── Relatório multi-objetivo ──────────────────────────────────────────
    mo_summary = controller.get_multi_objective_summary()
    if "[MultiObjective]" in mo_summary and "Não inicializado" not in mo_summary:
        print("\n" + "-" * 60)
        print(mo_summary)
        mo_path = output_dir / "multi_objective_report.txt"
        with open(mo_path, "w", encoding="utf-8") as f:
            f.write(mo_summary)

    # ── Relatório de convergência ────────────────────────────────────────
    conv_summary = controller.get_convergence_summary()
    if "[ConvergenceTracker]" in conv_summary and "Não inicializado" not in conv_summary:
        print("\n" + "-" * 60)
        print(conv_summary)
        conv_path = output_dir / "convergence_report.txt"
        with open(conv_path, "w", encoding="utf-8") as f:
            f.write(conv_summary)

    print(f"\n[Saída] Resultados em: {output_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
