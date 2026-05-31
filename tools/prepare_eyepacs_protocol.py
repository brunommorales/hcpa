#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


BUCKET_ORDER = ("low", "mid", "high", "very_high")
SPLIT_ORDER = ("train", "val", "test")
CRC32C_POLY = 0x82F63B78


@dataclass(frozen=True)
class ManifestRecord:
    image_path: str
    patient_id: str
    eye: str
    label: int
    width: int
    height: int
    short_side: int
    bucket: str
    label_source: str
CRC32C_TABLE = []
for table_index in range(256):
    crc = table_index
    for _ in range(8):
        if crc & 1:
            crc = (crc >> 1) ^ CRC32C_POLY
        else:
            crc >>= 1
    CRC32C_TABLE.append(crc & 0xFFFFFFFF)


class SimpleTFRecordWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = self.path.open("wb")

    def __enter__(self) -> "SimpleTFRecordWriter":
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def write(self, payload: bytes) -> None:
        length_bytes = struct.pack("<Q", len(payload))
        self.handle.write(length_bytes)
        self.handle.write(struct.pack("<I", masked_crc32c(length_bytes)))
        self.handle.write(payload)
        self.handle.write(struct.pack("<I", masked_crc32c(payload)))


def crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def encode_varint(value: int) -> bytes:
    value = int(value)
    if value < 0:
        raise ValueError(f"Negative varint not supported here: {value}")
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def encode_len_field(field_number: int, payload: bytes) -> bytes:
    tag = (int(field_number) << 3) | 2
    return encode_varint(tag) + encode_varint(len(payload)) + payload


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-free EyePACS manifest and optionally export TFRecord "
            "shards from the same split protocol."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=data_root / "EYEPACS_original_final",
        help="Directory with the original-resolution EyePACS images.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=data_root / "trainLabels.csv",
        help="Optional Kaggle/TFDS labels CSV with columns image,level.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "eyepacs_original_protocol",
        help="Directory that will receive the manifest(s) and summary JSON.",
    )
    parser.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Filename for the base manifest with the requested columns only.",
    )
    parser.add_argument(
        "--split-manifest-name",
        default="manifest_with_split.csv",
        help="Filename for the manifest augmented with split and binary_label.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(16, os.cpu_count() or 4),
        help="Worker threads used while reading image metadata.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Patient-level train ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Patient-level validation ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Patient-level test ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the patient-level split.",
    )
    parser.add_argument(
        "--split-stratify",
        choices=("raw_max", "binary_max", "none"),
        default="raw_max",
        help="Patient-level target used to stratify the split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick dry runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output directories/files.",
    )
    parser.add_argument(
        "--tfrec-dir",
        type=Path,
        default=None,
        help="Optional directory for TFRecord export derived from the split manifest.",
    )
    parser.add_argument(
        "--split-manifest-csv",
        type=Path,
        default=None,
        help="Optional existing manifest_with_split.csv to reuse for TFRecord export without rescanning images.",
    )
    parser.add_argument(
        "--linked-dir",
        type=Path,
        default=None,
        help="Optional directory for a bucket-specific linked dataset view with split/label subdirectories.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        choices=BUCKET_ORDER,
        default=None,
        help="Optional bucket filter for TFRecord export. Repeat the flag to keep multiple buckets.",
    )
    parser.add_argument(
        "--label-mode",
        choices=("binary", "raw"),
        default="binary",
        help="Label written inside TFRecords.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=512,
        help="Number of examples per TFRecord shard.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    ratios = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(value < 0.0 for value in ratios):
        raise SystemExit("Split ratios must be non-negative.")
    total = sum(ratios)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(
            f"Split ratios must sum to 1.0; received train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )
    return ratios


def map_binary_label(raw_label: int) -> int:
    return 0 if int(raw_label) < 2 else 1


def bucket_for_short_side(short_side: int) -> str:
    short_side = int(short_side)
    if short_side < 1024:
        return "low"
    if short_side < 1536:
        return "mid"
    if short_side < 2048:
        return "high"
    return "very_high"


def load_labels_map(labels_csv: Path) -> dict[str, int]:
    if not labels_csv.exists():
        return {}

    df = pd.read_csv(labels_csv)
    required_columns = {"image", "level"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"Missing columns in {labels_csv}: {sorted(missing_columns)}")
    return {
        str(row.image): int(row.level)
        for row in df.loc[:, ["image", "level"]].itertuples(index=False)
    }


def collect_image_paths(dataset_root: Path, limit: int | None) -> list[Path]:
    image_paths = sorted(dataset_root.rglob("*.jpeg"))
    if limit is not None:
        image_paths = image_paths[: int(limit)]
    if not image_paths:
        raise SystemExit(f"No .jpeg files found under {dataset_root}")
    return image_paths


def infer_eye(stem: str) -> tuple[str, str]:
    patient_id, sep, eye = stem.partition("_")
    if not sep or eye not in {"left", "right"}:
        raise ValueError(f"Unexpected EyePACS filename pattern: {stem}")
    return patient_id, eye


def build_manifest_record(image_path: Path, labels_map: dict[str, int]) -> ManifestRecord:
    patient_id, eye = infer_eye(image_path.stem)
    directory_label: int | None = None
    if image_path.parent.name.isdigit():
        directory_label = int(image_path.parent.name)

    csv_label = labels_map.get(image_path.stem)
    if csv_label is not None and directory_label is not None and int(csv_label) != int(directory_label):
        raise ValueError(
            f"Label mismatch for {image_path.name}: csv={csv_label} directory={directory_label}"
        )

    if csv_label is not None:
        label = int(csv_label)
        label_source = "csv"
    elif directory_label is not None:
        label = int(directory_label)
        label_source = "directory"
    else:
        raise ValueError(f"Could not infer label for {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    short_side = min(int(width), int(height))
    return ManifestRecord(
        image_path=str(image_path.resolve()),
        patient_id=str(patient_id),
        eye=str(eye),
        label=int(label),
        width=int(width),
        height=int(height),
        short_side=int(short_side),
        bucket=bucket_for_short_side(short_side),
        label_source=label_source,
    )


def build_manifest_dataframe(
    image_paths: list[Path],
    labels_map: dict[str, int],
    workers: int,
) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        records = list(executor.map(build_manifest_record, image_paths, [labels_map] * len(image_paths)))

    df = pd.DataFrame.from_records(record.__dict__ for record in records)
    df = df.sort_values(["patient_id", "eye", "image_path"], kind="stable").reset_index(drop=True)
    return df


def allocate_group_counts(group_size: int, ratios: tuple[float, float, float]) -> np.ndarray:
    if group_size <= 0:
        return np.zeros(3, dtype=np.int64)

    raw = np.asarray(ratios, dtype=np.float64) * int(group_size)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(group_size) - int(counts.sum())
    if remainder > 0:
        frac = raw - counts
        order = np.argsort(-frac, kind="stable")
        for idx in order[:remainder]:
            counts[idx] += 1
    return counts


def build_patient_split(
    manifest_df: pd.DataFrame,
    *,
    ratios: tuple[float, float, float],
    seed: int,
    stratify_mode: str,
) -> pd.DataFrame:
    patient_df = (
        manifest_df.assign(binary_label=manifest_df["label"].map(map_binary_label))
        .groupby("patient_id", sort=True)
        .agg(
            raw_max=("label", "max"),
            binary_max=("binary_label", "max"),
            image_count=("image_path", "size"),
        )
        .reset_index()
    )

    if stratify_mode == "raw_max":
        stratify_column = "raw_max"
    elif stratify_mode == "binary_max":
        stratify_column = "binary_max"
    else:
        stratify_column = None

    rng = np.random.default_rng(int(seed))
    split_assignments: list[pd.DataFrame] = []

    if stratify_column is None:
        groups = [(None, patient_df)]
    else:
        groups = list(patient_df.groupby(stratify_column, sort=True))

    for _, group_df in groups:
        group_df = group_df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        counts = allocate_group_counts(len(group_df), ratios)
        boundaries = np.cumsum(counts).tolist()
        split_values = np.empty(len(group_df), dtype=object)
        start = 0
        for split_name, stop in zip(SPLIT_ORDER, boundaries):
            split_values[start:stop] = split_name
            start = stop
        assigned = group_df.copy()
        assigned["split"] = split_values
        split_assignments.append(assigned)

    patient_split_df = pd.concat(split_assignments, axis=0, ignore_index=True)
    patient_split_df = patient_split_df.sort_values("patient_id", kind="stable").reset_index(drop=True)
    return patient_split_df


def apply_patient_split(manifest_df: pd.DataFrame, patient_split_df: pd.DataFrame) -> pd.DataFrame:
    split_df = manifest_df.merge(
        patient_split_df.loc[:, ["patient_id", "split", "raw_max", "binary_max"]],
        how="left",
        on="patient_id",
        validate="many_to_one",
    )
    if split_df["split"].isna().any():
        raise SystemExit("Some rows were not assigned to a patient split.")

    split_df["binary_label"] = split_df["label"].map(map_binary_label).astype(int)
    split_df = split_df.sort_values(["split", "patient_id", "eye", "image_path"], kind="stable").reset_index(drop=True)
    return split_df


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if any(path.iterdir()) and not overwrite:
            raise SystemExit(f"Output directory already exists and is not empty: {path} (use --overwrite)")
        if overwrite:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def summarize_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    counts = df[column].value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def load_existing_split_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Split manifest not found: {path}")

    df = pd.read_csv(path)
    required_columns = {
        "split",
        "image_path",
        "patient_id",
        "eye",
        "label",
        "binary_label",
        "width",
        "height",
        "short_side",
        "bucket",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing_columns)}")

    df = df.copy()
    df["image_path"] = df["image_path"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["eye"] = df["eye"].astype(str)
    df["split"] = df["split"].astype(str)
    df["bucket"] = df["bucket"].astype(str)
    for column in ("label", "binary_label", "width", "height", "short_side"):
        df[column] = df[column].astype(int)
    return df


def write_manifest_outputs(
    *,
    manifest_df: pd.DataFrame,
    split_df: pd.DataFrame,
    patient_split_df: pd.DataFrame,
    output_dir: Path,
    manifest_name: str,
    split_manifest_name: str,
    dataset_root: Path,
    labels_csv: Path,
    stratify_mode: str,
    ratios: tuple[float, float, float],
) -> tuple[Path, Path]:
    manifest_path = output_dir / manifest_name
    split_manifest_path = output_dir / split_manifest_name
    patient_split_path = output_dir / "patient_split.csv"
    summary_path = output_dir / "summary.json"

    manifest_columns = ["image_path", "patient_id", "eye", "label", "width", "height", "short_side", "bucket"]
    split_columns = ["split", "image_path", "patient_id", "eye", "label", "binary_label", "width", "height", "short_side", "bucket"]

    manifest_df.loc[:, manifest_columns].to_csv(manifest_path, index=False)
    split_df.loc[:, split_columns].to_csv(split_manifest_path, index=False)
    patient_split_df.to_csv(patient_split_path, index=False)

    split_summary: dict[str, dict[str, object]] = {}
    for split_name in SPLIT_ORDER:
        rows = split_df.loc[split_df["split"] == split_name]
        split_summary[split_name] = {
            "images": int(len(rows)),
            "patients": int(rows["patient_id"].nunique()),
            "raw_label_counts": summarize_counts(rows, "label"),
            "binary_label_counts": summarize_counts(rows, "binary_label"),
            "bucket_counts": summarize_counts(rows, "bucket"),
        }

    duplicate_patient_splits = (
        split_df.loc[:, ["patient_id", "split"]]
        .drop_duplicates()
        .groupby("patient_id", sort=False)["split"]
        .nunique()
    )
    leakage_patients = int((duplicate_patient_splits > 1).sum())

    label_source_counts = summarize_counts(manifest_df, "label_source")
    summary_payload = {
        "dataset_root": str(dataset_root),
        "labels_csv": str(labels_csv) if labels_csv.exists() else None,
        "manifest_path": str(manifest_path),
        "split_manifest_path": str(split_manifest_path),
        "patient_split_path": str(patient_split_path),
        "total_images": int(len(manifest_df)),
        "total_patients": int(manifest_df["patient_id"].nunique()),
        "raw_label_counts": summarize_counts(manifest_df, "label"),
        "binary_label_counts": summarize_counts(split_df, "binary_label"),
        "bucket_counts": summarize_counts(manifest_df, "bucket"),
        "eye_counts": summarize_counts(manifest_df, "eye"),
        "label_source_counts": label_source_counts,
        "split_stratify": stratify_mode,
        "split_ratios": {
            "train": float(ratios[0]),
            "val": float(ratios[1]),
            "test": float(ratios[2]),
        },
        "post_split_leakage_patients": leakage_patients,
        "splits": split_summary,
    }
    ensure_parent_dir(summary_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, split_manifest_path


def serialize_example(image_bytes: bytes, image_name: str, label: int) -> bytes:
    image_feature = encode_len_field(1, encode_len_field(1, image_bytes))
    image_name_feature = encode_len_field(1, encode_len_field(1, image_name.encode("utf-8")))
    label_feature = encode_len_field(3, encode_len_field(1, encode_varint(int(label))))

    imagem_entry = encode_len_field(1, b"imagem") + encode_len_field(2, image_feature)
    image_name_entry = encode_len_field(1, b"image_name") + encode_len_field(2, image_name_feature)
    label_entry = encode_len_field(1, b"retinopatia") + encode_len_field(2, label_feature)

    features_payload = (
        encode_len_field(1, imagem_entry)
        + encode_len_field(1, image_name_entry)
        + encode_len_field(1, label_entry)
    )
    return encode_len_field(1, features_payload)


def write_single_tfrec_shard(
    *,
    split_name: str,
    shard_idx: int,
    total_shards: int,
    shard_rows: list[tuple[str, str, int]],
    shard_path: Path,
) -> tuple[Path, int, float]:
    offsets: list[int] = []
    offset = 0
    started = time.time()

    with shard_path.open("wb") as handle:
        for image_path_str, image_name, label_value in shard_rows:
            image_bytes = Path(image_path_str).read_bytes()
            payload = serialize_example(
                image_bytes=image_bytes,
                image_name=image_name,
                label=label_value,
            )
            length_bytes = struct.pack("<Q", len(payload))
            length_crc = struct.pack("<I", masked_crc32c(length_bytes))
            payload_crc = struct.pack("<I", masked_crc32c(payload))
            offsets.append(offset)
            handle.write(length_bytes)
            handle.write(length_crc)
            handle.write(payload)
            handle.write(payload_crc)
            offset += len(length_bytes) + len(length_crc) + len(payload) + len(payload_crc)

    idx_path = Path(str(shard_path) + ".idx")
    with idx_path.open("w", encoding="utf-8") as writer:
        for value in offsets:
            writer.write(f"{value}\n")

    elapsed = round(time.time() - started, 1)
    return shard_path, len(shard_rows), elapsed


def write_tfrec_split(
    *,
    split_name: str,
    rows: pd.DataFrame,
    output_dir: Path,
    shard_size: int,
    label_mode: str,
    write_workers: int,
) -> int:
    if rows.empty:
        print(f"{split_name}: no rows selected")
        return 0

    shard_count = math.ceil(len(rows) / shard_size)
    tasks = []
    for shard_idx in range(shard_count):
        start = shard_idx * shard_size
        stop = min(len(rows), start + shard_size)
        shard_rows = rows.iloc[start:stop]
        shard_path = output_dir / f"{split_name}{shard_idx:02d}-{len(shard_rows)}.tfrec"
        shard_payload = [
            (
                str(row.image_path),
                Path(row.image_path).name,
                int(row.binary_label if label_mode == "binary" else row.label),
            )
            for row in shard_rows.itertuples(index=False)
        ]
        tasks.append((shard_idx, shard_payload, shard_path))

    max_workers = max(1, min(int(write_workers), len(tasks)))
    print(f"{split_name}: writing {len(rows)} records into {shard_count} shard(s) with workers={max_workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                write_single_tfrec_shard,
                split_name=split_name,
                shard_idx=shard_idx,
                total_shards=shard_count,
                shard_rows=shard_payload,
                shard_path=shard_path,
            )
            for shard_idx, shard_payload, shard_path in tasks
        ]
        for future in as_completed(futures):
            shard_path, record_count, elapsed = future.result()
            print(f"  finished {shard_path.name} ({record_count} records) in {elapsed} sec", flush=True)
    return shard_count


def export_tfrecords(
    *,
    split_df: pd.DataFrame,
    tfrec_dir: Path,
    bucket_filter: list[str] | None,
    shard_size: int,
    label_mode: str,
    write_workers: int,
    overwrite: bool,
) -> None:
    ensure_output_dir(tfrec_dir, overwrite=overwrite)

    filtered_df = split_df.copy()
    if bucket_filter:
        keep = set(bucket_filter)
        filtered_df = filtered_df.loc[filtered_df["bucket"].isin(keep)].copy()
        if filtered_df.empty:
            raise SystemExit(f"No rows left after bucket filter: {bucket_filter}")

    export_manifest_path = tfrec_dir / "manifest_used.csv"
    export_summary_path = tfrec_dir / "summary.json"
    filtered_df.loc[
        :,
        ["split", "image_path", "patient_id", "eye", "label", "binary_label", "width", "height", "short_side", "bucket"],
    ].to_csv(export_manifest_path, index=False)

    split_counts: dict[str, dict[str, object]] = {}
    total_shards = 0
    for split_name in SPLIT_ORDER:
        rows = filtered_df.loc[filtered_df["split"] == split_name].reset_index(drop=True)
        total_shards += write_tfrec_split(
            split_name=split_name,
            rows=rows,
            output_dir=tfrec_dir,
            shard_size=shard_size,
            label_mode=label_mode,
            write_workers=write_workers,
        )
        split_counts[split_name] = {
            "images": int(len(rows)),
            "patients": int(rows["patient_id"].nunique()),
            "raw_label_counts": summarize_counts(rows, "label"),
            "binary_label_counts": summarize_counts(rows, "binary_label"),
            "bucket_counts": summarize_counts(rows, "bucket"),
        }

    export_summary = {
        "manifest_used": str(export_manifest_path),
        "total_images": int(len(filtered_df)),
        "total_patients": int(filtered_df["patient_id"].nunique()),
        "bucket_filter": list(bucket_filter) if bucket_filter else None,
        "label_mode": str(label_mode),
        "shard_size": int(shard_size),
        "total_shards": int(total_shards),
        "splits": split_counts,
    }
    export_summary_path.write_text(json.dumps(export_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        os.symlink(src, dst)


def export_linked_dataset(
    *,
    split_df: pd.DataFrame,
    linked_dir: Path,
    bucket_filter: list[str] | None,
    overwrite: bool,
) -> None:
    ensure_output_dir(linked_dir, overwrite=overwrite)

    filtered_df = split_df.copy()
    if bucket_filter:
        keep = set(bucket_filter)
        filtered_df = filtered_df.loc[filtered_df["bucket"].isin(keep)].copy()
        if filtered_df.empty:
            raise SystemExit(f"No rows left after bucket filter: {bucket_filter}")

    manifest_path = linked_dir / "manifest_used.csv"
    summary_path = linked_dir / "summary.json"
    filtered_df.loc[
        :,
        ["split", "image_path", "patient_id", "eye", "label", "binary_label", "width", "height", "short_side", "bucket"],
    ].to_csv(manifest_path, index=False)

    split_counts: dict[str, dict[str, object]] = {}
    for split_name in SPLIT_ORDER:
        rows = filtered_df.loc[filtered_df["split"] == split_name].reset_index(drop=True)
        split_csv_path = linked_dir / f"{split_name}.csv"
        rows.loc[
            :,
            ["split", "image_path", "patient_id", "eye", "label", "binary_label", "width", "height", "short_side", "bucket"],
        ].to_csv(split_csv_path, index=False)

        for row in rows.itertuples(index=False):
            src = Path(row.image_path)
            dst = linked_dir / split_name / str(row.label) / src.name
            link_file(src, dst)

        split_counts[split_name] = {
            "images": int(len(rows)),
            "patients": int(rows["patient_id"].nunique()),
            "raw_label_counts": summarize_counts(rows, "label"),
            "binary_label_counts": summarize_counts(rows, "binary_label"),
            "bucket_counts": summarize_counts(rows, "bucket"),
        }

    export_summary = {
        "manifest_used": str(manifest_path),
        "total_images": int(len(filtered_df)),
        "total_patients": int(filtered_df["patient_id"].nunique()),
        "bucket_filter": list(bucket_filter) if bucket_filter else None,
        "splits": split_counts,
    }
    summary_path.write_text(json.dumps(export_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    split_df: pd.DataFrame

    if args.split_manifest_csv is not None:
        split_manifest_path = args.split_manifest_csv.resolve()
        split_df = load_existing_split_manifest(split_manifest_path)
        print(f"Reusing split manifest from {split_manifest_path}")
        print(f"Rows: {len(split_df)} | Patients: {split_df['patient_id'].nunique()}")
    else:
        ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

        dataset_root = args.dataset_root.resolve()
        labels_csv = args.labels_csv.resolve()
        output_dir = args.output_dir.resolve()

        if not dataset_root.is_dir():
            raise SystemExit(f"Dataset directory not found: {dataset_root}")

        if output_dir.exists():
            existing_paths = [
                output_dir / args.manifest_name,
                output_dir / args.split_manifest_name,
                output_dir / "patient_split.csv",
                output_dir / "summary.json",
            ]
            if any(path.exists() for path in existing_paths) and not args.overwrite:
                raise SystemExit(f"Manifest outputs already exist under {output_dir} (use --overwrite)")
            if args.overwrite:
                shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = collect_image_paths(dataset_root=dataset_root, limit=args.limit)
        labels_map = load_labels_map(labels_csv)

        started = time.time()
        manifest_df = build_manifest_dataframe(
            image_paths=image_paths,
            labels_map=labels_map,
            workers=args.workers,
        )
        patient_split_df = build_patient_split(
            manifest_df=manifest_df,
            ratios=ratios,
            seed=args.seed,
            stratify_mode=args.split_stratify,
        )
        split_df = apply_patient_split(manifest_df=manifest_df, patient_split_df=patient_split_df)
        manifest_path, split_manifest_path = write_manifest_outputs(
            manifest_df=manifest_df,
            split_df=split_df,
            patient_split_df=patient_split_df,
            output_dir=output_dir,
            manifest_name=args.manifest_name,
            split_manifest_name=args.split_manifest_name,
            dataset_root=dataset_root,
            labels_csv=labels_csv,
            stratify_mode=args.split_stratify,
            ratios=ratios,
        )

        elapsed = round(time.time() - started, 1)
        print(f"Manifest written to {manifest_path}")
        print(f"Split manifest written to {split_manifest_path}")
        print(f"Rows: {len(manifest_df)} | Patients: {manifest_df['patient_id'].nunique()} | Elapsed: {elapsed} sec")

    if args.tfrec_dir is not None:
        tfrec_dir = args.tfrec_dir.resolve()
        output_dir = args.output_dir.resolve()
        if args.split_manifest_csv is None and tfrec_dir == output_dir:
            raise SystemExit("--tfrec-dir must be different from --output-dir to avoid deleting the manifest outputs.")
        export_tfrecords(
            split_df=split_df,
            tfrec_dir=tfrec_dir,
            bucket_filter=args.bucket,
            shard_size=args.shard_size,
            label_mode=args.label_mode,
            write_workers=args.workers,
            overwrite=args.overwrite,
        )
        print(f"TFRecords written to {tfrec_dir}")

    if args.linked_dir is not None:
        linked_dir = args.linked_dir.resolve()
        export_linked_dataset(
            split_df=split_df,
            linked_dir=linked_dir,
            bucket_filter=args.bucket,
            overwrite=args.overwrite,
        )
        print(f"Linked dataset written to {linked_dir}")


if __name__ == "__main__":
    main()
