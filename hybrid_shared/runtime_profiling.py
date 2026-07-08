"""
Runtime profiling helpers for NVTX and torch.profiler.

Environment variables:
- HCPA_ENABLE_TORCH_PROFILER=1
- HCPA_TORCH_PROFILER_DIR=/path/to/output
- HCPA_TORCH_PROFILER_WAIT=1
- HCPA_TORCH_PROFILER_WARMUP=1
- HCPA_TORCH_PROFILER_ACTIVE=3
- HCPA_TORCH_PROFILER_REPEAT=1
- HCPA_TORCH_PROFILER_MAX_STEPS=0
- HCPA_TORCH_PROFILER_RECORD_SHAPES=1
- HCPA_TORCH_PROFILER_PROFILE_MEMORY=1
- HCPA_TORCH_PROFILER_WITH_STACK=0
- HCPA_EMIT_NVTX=1
- HCPA_PROFILE_RANK0_ONLY=1
- HCPA_PROFILE_STAGES=train,val,test,eval

When `PROFILE_TOOL` is set to an external CUDA profiler such as `nsys` or
`ncu`, torch.profiler is disabled automatically to avoid CUPTI subscriber
conflicts. NVTX ranges remain enabled.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class RuntimeProfilingConfig:
    torch_profiler_requested: bool
    enable_torch_profiler: bool
    emit_nvtx: bool
    rank0_only: bool
    wait_steps: int
    warmup_steps: int
    active_steps: int
    repeat: int
    max_steps: int
    record_shapes: bool
    profile_memory: bool
    with_stack: bool
    stages: tuple[str, ...]
    output_dir: str
    external_profiler: str

    @classmethod
    def from_env(cls, output_dir: Path) -> "RuntimeProfilingConfig":
        stages_raw = os.environ.get("HCPA_PROFILE_STAGES", "train,val,test,eval")
        stages = tuple(stage.strip() for stage in stages_raw.split(",") if stage.strip())
        torch_profiler_requested = _env_flag("HCPA_ENABLE_TORCH_PROFILER", False)
        external_profiler = os.environ.get("PROFILE_TOOL", "none").strip().lower() or "none"
        external_profiler_active = external_profiler not in {"none", "0", "false", "no", "off"}
        return cls(
            torch_profiler_requested=torch_profiler_requested,
            enable_torch_profiler=torch_profiler_requested and not external_profiler_active,
            emit_nvtx=_env_flag("HCPA_EMIT_NVTX", False),
            rank0_only=_env_flag("HCPA_PROFILE_RANK0_ONLY", True),
            wait_steps=max(0, _env_int("HCPA_TORCH_PROFILER_WAIT", 1)),
            warmup_steps=max(0, _env_int("HCPA_TORCH_PROFILER_WARMUP", 1)),
            active_steps=max(1, _env_int("HCPA_TORCH_PROFILER_ACTIVE", 3)),
            repeat=max(1, _env_int("HCPA_TORCH_PROFILER_REPEAT", 1)),
            max_steps=max(0, _env_int("HCPA_TORCH_PROFILER_MAX_STEPS", 0)),
            record_shapes=_env_flag("HCPA_TORCH_PROFILER_RECORD_SHAPES", True),
            profile_memory=_env_flag("HCPA_TORCH_PROFILER_PROFILE_MEMORY", True),
            with_stack=_env_flag("HCPA_TORCH_PROFILER_WITH_STACK", False),
            stages=stages,
            output_dir=str(output_dir),
            external_profiler=external_profiler,
        )


class RuntimeProfiler:
    """Combines NVTX ranges and torch.profiler traces."""

    def __init__(
        self,
        output_root: str | os.PathLike[str],
        rank: int = 0,
        project_name: str = "hcpa",
    ) -> None:
        self.rank = int(rank)
        self.project_name = project_name
        requested_dir = os.environ.get("HCPA_TORCH_PROFILER_DIR")
        self.output_dir = Path(requested_dir) if requested_dir else Path(output_root) / "profiling"
        self.config = RuntimeProfilingConfig.from_env(self.output_dir)
        self.active = (not self.config.rank0_only) or self.rank == 0
        self.enabled = self.active and (self.config.enable_torch_profiler or self.config.emit_nvtx)
        self.step_count = 0
        self.trace_index = 0
        self._trace_files: list[str] = []
        self._profiler_cm: Optional[contextlib.AbstractContextManager] = None
        self._profiler = None

        if (
            self.rank == 0
            and self.config.torch_profiler_requested
            and not self.config.enable_torch_profiler
            and self.config.external_profiler != "none"
        ):
            print(
                "[RuntimeProfiler] Disabling torch.profiler because "
                f"PROFILE_TOOL={self.config.external_profiler} already uses CUPTI; keeping NVTX ranges."
            )

        if not self.enabled:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.enable_torch_profiler:
            self._start_torch_profiler()

    def _start_torch_profiler(self) -> None:
        try:
            from torch.profiler import ProfilerActivity, profile, schedule
        except Exception:
            return

        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

        self._profiler_cm = profile(
            activities=activities,
            schedule=schedule(
                wait=self.config.wait_steps,
                warmup=self.config.warmup_steps,
                active=self.config.active_steps,
                repeat=self.config.repeat,
            ),
            on_trace_ready=self._on_trace_ready,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.profile_memory,
            with_stack=self.config.with_stack,
        )
        self._profiler = self._profiler_cm.__enter__()

    def _on_trace_ready(self, prof) -> None:
        trace_name = f"{self.project_name}_rank{self.rank}_trace_{self.trace_index:03d}"
        trace_path = self.output_dir / f"{trace_name}.json"
        table_path = self.output_dir / f"{trace_name}_kernels.txt"

        prof.export_chrome_trace(str(trace_path))
        sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
        with table_path.open("w", encoding="utf-8") as handle:
            handle.write(prof.key_averages().table(sort_by=sort_key, row_limit=200))

        self._trace_files.append(trace_path.name)
        self.trace_index += 1

    def stage_enabled(self, stage: str) -> bool:
        return self.enabled and stage in self.config.stages

    @contextlib.contextmanager
    def range(self, name: str):
        if not self.enabled:
            yield
            return

        pushed_nvtx = False
        record_cm = torch.autograd.profiler.record_function(name)
        try:
            if self.config.emit_nvtx and torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
                pushed_nvtx = True
            with record_cm:
                yield
        finally:
            if pushed_nvtx:
                torch.cuda.nvtx.range_pop()

    def fetch_next(self, iterator: Iterator, stage: str):
        if self.stage_enabled(stage):
            with self.range(f"{stage}/data_fetch"):
                return next(iterator)
        return next(iterator)

    def step(self, stage: str) -> None:
        if not self.stage_enabled(stage):
            return
        if self._profiler is None:
            return
        if self.config.max_steps > 0 and self.step_count >= self.config.max_steps:
            return
        self._profiler.step()
        self.step_count += 1

    def close(self) -> None:
        if not self.enabled:
            return
        if self._profiler_cm is not None:
            self._profiler_cm.__exit__(None, None, None)
            self._profiler_cm = None
            self._profiler = None

        manifest_path = self.output_dir / f"{self.project_name}_rank{self.rank}_profiling_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "project_name": self.project_name,
                    "rank": self.rank,
                    "step_count": self.step_count,
                    "trace_files": self._trace_files,
                    "config": asdict(self.config),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
