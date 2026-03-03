#!/usr/bin/env python3
"""
Remove temporary logger artefacts left by training runs in the hcpa project.

The script targets common logging outputs from TensorFlow/PyTorch/MONAI runs
inside the `results` folders while leaving the main `logs/LOGS` job outputs
untouched.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

# Directories we never descend into while scanning (speed + safety).
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "env",
    "env_arm",
    "env_x86",
    "env_graph",
    "logs",
    "LOGS",
}

# File/dir names that are safe to delete when found under a results run folder.
TARGET_FILE_NAMES = {
    "train.log",  # generic training logger output
    "metrics.jsonl",  # step-by-step metrics logger
}

# Files we also delete even outside results (e.g., crash dumps).
ALWAYS_FILE_NAMES = {
    "core",  # crash dump files
}

# Patterns that catch TensorBoard event files produced by TF/PT runs.
TARGET_FILE_PATTERNS = (
    re.compile(r"events\.out\.tfevents\..+"),
    re.compile(r"checkpoint.*\.(msgpack|ckpt|pkl|pt|pth|npz)$"),
)

# Logger directories that are safe to drop.
TARGET_DIR_NAMES = {
    "torchelastic_logs",  # PyTorch elastic/torchrun logs
    "lightning_logs",  # PyTorch Lightning/TensorBoard logs
    "wandb",  # Weights & Biases run folders
    "triton_cache",  # PyTorch Triton compilation cache
    "tfdata_cache",  # TensorFlow tf.data cache
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def find_targets(root: Path, scope: str) -> tuple[list[Path], list[Path]]:
    file_targets: list[Path] = []
    dir_targets: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        if should_skip(current):
            dirnames[:] = []  # prevent descending further
            continue

        # Remove skipped dirs from traversal to keep the walk fast.
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]

        # Even when limiting to results, still allow certain global files (e.g., crash core dumps).
        limiting_to_results = scope == "results-only" and "results" not in current.parts

        matched_dirnames = [d for d in dirnames if d in TARGET_DIR_NAMES]
        for d in matched_dirnames:
            dir_targets.append(current / d)

        for fname in filenames:
            if fname in ALWAYS_FILE_NAMES:
                file_targets.append(current / fname)
                continue
            if limiting_to_results:
                continue
            if fname in TARGET_FILE_NAMES:
                file_targets.append(current / fname)
                continue
            if any(pat.match(fname) for pat in TARGET_FILE_PATTERNS):
                file_targets.append(current / fname)

    return file_targets, dir_targets


def remove_paths(paths: Iterable[Path], dry_run: bool, max_print: int) -> int:
    paths_list = list(paths) if not isinstance(paths, list) else paths
    total = len(paths_list)
    removed = 0

    for idx, path in enumerate(paths_list):
        suppress_print = max_print >= 0 and idx >= max_print
        if suppress_print and idx == max_print:
            remaining = total - max_print
            if remaining > 0:
                print(f"[info] skipping print for {remaining} more items (use --max-print -1 to show all)")

        if dry_run:
            if not suppress_print:
                print(f"[dry-run] would remove {path}")
            removed += 1
            continue

        try:
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=False)
            else:
                path.unlink(missing_ok=True)
            if not suppress_print:
                print(f"removed {path}")
            removed += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to remove {path}: {exc}")

    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete temporary logger files produced by hcpa training runs.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder to scan (default: folder containing this script).",
    )
    parser.add_argument(
        "--scope",
        choices=["results-only", "all"],
        default="results-only",
        help="Limit cleanup to paths that contain 'results' (safer default) or scan all folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be removed.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=200,
        help="Maximum number of matched paths to print (-1 shows all).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"Root folder does not exist: {root}")

    file_targets, dir_targets = find_targets(root, args.scope)

    print(f"Scanning root: {root}")
    print(f"Matched files: {len(file_targets)}, directories: {len(dir_targets)}")

    removed_files = remove_paths(file_targets, args.dry_run, args.max_print)
    removed_dirs = remove_paths(dir_targets, args.dry_run, args.max_print)

    print(
        f"Cleanup {'preview' if args.dry_run else 'done'}: "
        f"{removed_files} files, {removed_dirs} directories",
    )


if __name__ == "__main__":
    main()
