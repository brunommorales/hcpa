#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image


@dataclass(frozen=True)
class Record:
    src_path: Path
    output_name: str
    label: int
    raw_label: int


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"

    parser = argparse.ArgumentParser(
        description=(
            "Prepare the resized_train dataset with binary labels and export it "
            "as TFRecord shards compatible with the HCPA pipeline."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_root / "resized_train",
        help="Directory with source images.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=data_root / "resized_train_labels.csv",
        help="CSV with columns imagem,retinopatia.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_root / "resized_train-tfrec",
        help="Output directory for TFRecord shards and metadata.",
    )
    parser.add_argument(
        "--diameter",
        type=int,
        default=299,
        help="Final square image size.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100,
        help="Number of records per TFRecord shard.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=10,
        help="Background threshold used to estimate the fundus bounding box.",
    )
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.03,
        help="Extra padding around the detected fundus box before resizing.",
    )
    parser.add_argument(
        "--image-format",
        choices=("png", "jpeg"),
        default="png",
        help="Encoded image format stored inside the TFRecord.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality when --image-format=jpeg.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Number of worker threads used while preparing records.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.30,
        help="Fraction reserved for the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the stratified split.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for quick validation runs.",
    )
    return parser.parse_args()


def map_label(raw_label: int) -> int:
    return 0 if raw_label < 2 else 1


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def square_crop_with_padding(image: Image.Image, left: int, top: int, size: int) -> Image.Image:
    src_left = max(left, 0)
    src_top = max(top, 0)
    src_right = min(left + size, image.width)
    src_bottom = min(top + size, image.height)

    canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    cropped = image.crop((src_left, src_top, src_right, src_bottom))
    dst_x = src_left - left
    dst_y = src_top - top
    canvas.paste(cropped, (dst_x, dst_y))
    return canvas


def preprocess_image(
    image_path: Path,
    diameter: int,
    mask_threshold: int,
    pad_ratio: float,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    gray = arr.mean(axis=2)
    mask = gray > mask_threshold

    if mask.any():
        ys, xs = np.nonzero(mask)
        x_min = int(xs.min())
        x_max = int(xs.max())
        y_min = int(ys.min())
        y_max = int(ys.max())
    else:
        x_min = 0
        y_min = 0
        x_max = image.width - 1
        y_max = image.height - 1

    box_w = x_max - x_min + 1
    box_h = y_max - y_min + 1
    size = max(box_w, box_h)
    size = max(1, int(math.ceil(size * (1.0 + pad_ratio))))

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0

    left = int(round(center_x - size / 2.0))
    top = int(round(center_y - size / 2.0))

    cropped = square_crop_with_padding(image, left=left, top=top, size=size)
    return cropped.resize((diameter, diameter), resample=Image.Resampling.LANCZOS)


def encode_image(image: Image.Image, image_format: str, jpeg_quality: int) -> bytes:
    buf = BytesIO()
    if image_format == "png":
        image.save(buf, format="PNG")
    else:
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
    return buf.getvalue()


def serialize_example(img_bytes: bytes, image_name: str, label: int) -> bytes:
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                "imagem": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
                "image_name": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_name.encode("utf-8")])
                ),
                "retinopatia": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(label)])
                ),
            }
        )
    )
    return example.SerializeToString()


def build_record_payload(
    record: Record,
    diameter: int,
    mask_threshold: int,
    pad_ratio: float,
    image_format: str,
    jpeg_quality: int,
) -> bytes:
    image = preprocess_image(
        image_path=record.src_path,
        diameter=diameter,
        mask_threshold=mask_threshold,
        pad_ratio=pad_ratio,
    )
    img_bytes = encode_image(
        image=image,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )
    return serialize_example(img_bytes=img_bytes, image_name=record.output_name, label=record.label)


def summarize_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in df[column].value_counts().sort_index().items()
    }


def load_labels(labels_csv: Path, input_dir: Path, limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    required_columns = {"imagem", "retinopatia"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise SystemExit(f"Missing columns in {labels_csv}: {sorted(missing_columns)}")

    df = df.copy()
    if limit is not None:
        df = df.iloc[:limit].copy()

    df["retinopatia_raw"] = df["retinopatia"].astype(int)
    df["retinopatia"] = df["retinopatia_raw"].map(map_label).astype(int)
    df["imagem_source"] = df["imagem"].astype(str)
    df["imagem"] = df["imagem_source"].map(lambda value: f"{Path(value).stem}.jpg")

    missing_files = [
        name
        for name in df["imagem_source"]
        if not (input_dir / name).exists()
    ]
    if missing_files:
        preview = ", ".join(missing_files[:10])
        raise SystemExit(
            f"Missing {len(missing_files)} input images under {input_dir}. Sample: {preview}"
        )

    return df


def stratified_split(df: pd.DataFrame, test_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= test_ratio < 1.0:
        raise SystemExit(f"--test-ratio must be between 0.0 and 1.0 (received {test_ratio})")

    rng = np.random.default_rng(seed)
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for _, group in df.groupby("retinopatia", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)

        test_count = int(round(len(indices) * test_ratio))
        if 0 < test_count >= len(indices):
            test_count = len(indices) - 1

        test_idx = np.sort(indices[:test_count])
        train_idx = np.sort(indices[test_count:])

        train_frames.append(df.loc[train_idx])
        if len(test_idx) > 0:
            test_frames.append(df.loc[test_idx])

    train_df = pd.concat(train_frames, axis=0).reset_index(drop=True)
    test_df = (
        pd.concat(test_frames, axis=0).reset_index(drop=True)
        if test_frames
        else df.iloc[0:0].copy()
    )
    return train_df, test_df


def build_records(df: pd.DataFrame, input_dir: Path) -> list[Record]:
    return [
        Record(
            src_path=input_dir / row.imagem_source,
            output_name=row.imagem,
            label=int(row.retinopatia),
            raw_label=int(row.retinopatia_raw),
        )
        for row in df.itertuples(index=False)
    ]


def write_metadata(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
    train_shards: int,
    test_shards: int,
) -> None:
    labels_train = output_dir / "labels_train.csv"
    labels_test = output_dir / "labels_test.csv"
    manifest = output_dir / "manifest.csv"
    summary = output_dir / "summary.json"

    train_df.loc[:, ["imagem", "retinopatia"]].to_csv(labels_train, index=False)
    test_df.loc[:, ["imagem", "retinopatia"]].to_csv(labels_test, index=False)

    manifest_df = pd.concat(
        [
            train_df.assign(split="train"),
            test_df.assign(split="test"),
        ],
        axis=0,
        ignore_index=True,
    )
    manifest_df.loc[:, ["split", "imagem_source", "imagem", "retinopatia_raw", "retinopatia"]].to_csv(
        manifest,
        index=False,
    )

    payload = {
        "input_dir": str(args.input_dir),
        "labels_csv": str(args.labels_csv),
        "output_dir": str(args.output_dir),
        "total_records": int(len(train_df) + len(test_df)),
        "train_records": int(len(train_df)),
        "test_records": int(len(test_df)),
        "raw_label_counts_total": summarize_counts(manifest_df, "retinopatia_raw"),
        "raw_label_counts_train": summarize_counts(train_df, "retinopatia_raw"),
        "raw_label_counts_test": summarize_counts(test_df, "retinopatia_raw"),
        "binary_label_counts_total": summarize_counts(manifest_df, "retinopatia"),
        "binary_label_counts_train": summarize_counts(train_df, "retinopatia"),
        "binary_label_counts_test": summarize_counts(test_df, "retinopatia"),
        "diameter": int(args.diameter),
        "shard_size": int(args.shard_size),
        "train_shards": int(train_shards),
        "test_shards": int(test_shards),
        "mask_threshold": int(args.mask_threshold),
        "pad_ratio": float(args.pad_ratio),
        "image_format": args.image_format,
        "jpeg_quality": int(args.jpeg_quality),
        "test_ratio": float(args.test_ratio),
        "seed": int(args.seed),
    }
    summary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_split_records(
    output_dir: Path,
    split_name: str,
    records: list[Record],
    args: argparse.Namespace,
    executor: ThreadPoolExecutor,
) -> int:
    if not records:
        print(f"{split_name}: no records to write")
        return 0

    shard_count = math.ceil(len(records) / args.shard_size)

    for shard_idx in range(shard_count):
        shard_start = shard_idx * args.shard_size
        shard_end = min(len(records), shard_start + args.shard_size)
        shard_records = records[shard_start:shard_end]
        shard_path = output_dir / f"{split_name}{shard_idx:02d}-{len(shard_records)}.tfrec"

        print()
        print(
            f"Writing {split_name} TFRecord {shard_idx + 1} of {shard_count}: "
            f"{shard_path.name}"
        )
        shard_started = time.time()

        with tf.io.TFRecordWriter(str(shard_path)) as writer:
            payloads = executor.map(
                build_record_payload,
                shard_records,
                [args.diameter] * len(shard_records),
                [args.mask_threshold] * len(shard_records),
                [args.pad_ratio] * len(shard_records),
                [args.image_format] * len(shard_records),
                [args.jpeg_quality] * len(shard_records),
            )
            for idx, payload in enumerate(payloads, start=1):
                writer.write(payload)
                if idx % 10 == 0 or idx == len(shard_records):
                    pct = 100.0 * idx / len(shard_records)
                    print(f"  Writing {pct:6.2f}% complete", end="\r", flush=True)

        elapsed = round(time.time() - shard_started, 1)
        print()
        print(f"Elapsed: {elapsed} sec")

    return shard_count


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    labels_csv = args.labels_csv.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not labels_csv.is_file():
        raise SystemExit(f"Labels CSV not found: {labels_csv}")

    df = load_labels(labels_csv=labels_csv, input_dir=input_dir, limit=args.limit)
    train_df, test_df = stratified_split(df=df, test_ratio=args.test_ratio, seed=args.seed)
    train_records = build_records(df=train_df, input_dir=input_dir)
    test_records = build_records(df=test_df, input_dir=input_dir)

    ensure_clean_dir(output_dir)

    print("Raw label counts total:")
    print(df["retinopatia_raw"].value_counts().sort_index())
    print("Binary label counts total:")
    print(df["retinopatia"].value_counts().sort_index())
    print("Binary label counts train:")
    print(train_df["retinopatia"].value_counts().sort_index())
    print("Binary label counts test:")
    print(test_df["retinopatia"].value_counts().sort_index())

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        train_shards = write_split_records(
            output_dir=output_dir,
            split_name="train",
            records=train_records,
            args=args,
            executor=executor,
        )
        test_shards = write_split_records(
            output_dir=output_dir,
            split_name="test",
            records=test_records,
            args=args,
            executor=executor,
        )

    total_elapsed = round(time.time() - start_time, 1)
    write_metadata(
        output_dir=output_dir,
        train_df=train_df,
        test_df=test_df,
        args=args,
        train_shards=train_shards,
        test_shards=test_shards,
    )

    print()
    print(
        f"Finished writing {train_shards + test_shards} TFRecord shard(s) to {output_dir}"
    )
    print(f"Total elapsed: {total_elapsed} sec")


if __name__ == "__main__":
    main()
