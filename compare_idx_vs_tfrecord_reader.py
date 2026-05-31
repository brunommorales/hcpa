#!/usr/bin/env python3
"""
Compara a leitura manual por offset do `.idx` com a leitura padrão de TFRecord.

Uso típico:
  /home/users/bmmorales/projects/hcpa/env/bin/python compare_idx_vs_tfrecord_reader.py \
      --tfrec data/resized_train-tfrec/train00-100.tfrec \
      --n 8

Isso testa o ponto crítico do `pytorch_base` vs `pytorch_opt`:
- `pytorch_base` abre o `.tfrec`, usa offsets do `.idx` e faz seek manual.
- `pytorch_opt` com DALI também usa o par `.tfrec + .idx` via `fn.readers.tfrecord(...)`.

O script valida:
1. Se o registro protobuf é byte-a-byte igual.
2. Se o `imagem` embutido no Example é byte-a-byte igual.
3. Se o RGB decodificado bate.
4. Quanto o tensor "base" (resize + preprocess Inception) difere de um
   tensor "opt-like" com crop central antes do resize.
"""

from __future__ import annotations

import argparse
import io
import json
import struct
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare idx seek vs TFRecord reader.")
    parser.add_argument(
        "--tfrec",
        type=Path,
        default=Path("data/resized_train-tfrec/train00-100.tfrec"),
        help="Arquivo .tfrec a ser inspecionado.",
    )
    parser.add_argument(
        "--idx",
        type=Path,
        default=None,
        help="Arquivo .idx correspondente. Se omitido, usa <tfrec>.idx ou <tfrec>.tfrec.idx.",
    )
    parser.add_argument("--n", type=int, default=8, help="Quantidade de registros para conferir.")
    parser.add_argument("--img-size", type=int, default=299, help="Resize final usado na comparação.")
    parser.add_argument(
        "--fundus-crop-ratio",
        type=float,
        default=0.9,
        help="Crop central usado na simulação do pipeline otimizado.",
    )
    return parser.parse_args()


def infer_idx_path(tfrec_path: Path) -> Path:
    candidates = [tfrec_path.with_suffix(".tfrec.idx"), Path(str(tfrec_path) + ".idx"), tfrec_path.with_suffix(".idx")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Nenhum .idx encontrado para {tfrec_path}")


def read_record_at(tf_path: Path, offset: int) -> bytes:
    with tf_path.open("rb") as fh:
        fh.seek(offset)
        length = struct.unpack("<Q", fh.read(8))[0]
        fh.read(4)  # crc do tamanho
        data = fh.read(length)
        fh.read(4)  # crc dos dados
        return data


def parse_example(record_bytes: bytes) -> tuple[bytes, int, str | None]:
    ex = tf.train.Example()
    ex.ParseFromString(record_bytes)
    feats = ex.features.feature
    img_bytes = feats["imagem"].bytes_list.value[0]
    label = int(feats["retinopatia"].int64_list.value[0])
    image_name = feats["image_name"].bytes_list.value[0].decode() if "image_name" in feats else None
    return img_bytes, label, image_name


def pil_rgb(img_bytes: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"), dtype=np.uint8)


def tf_rgb(img_bytes: bytes) -> np.ndarray:
    tensor = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    return tensor.numpy().astype(np.uint8)


def center_crop_pil(img: Image.Image, ratio: float) -> Image.Image:
    if ratio >= 0.999:
        return img
    width, height = img.size
    crop_w = max(1, int(round(width * ratio)))
    crop_h = max(1, int(round(height * ratio)))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def resize_pil(arr: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(arr)
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)


def preprocess_inception(arr: np.ndarray) -> np.ndarray:
    return arr / 127.5 - 1.0


def summarize_diff(a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "equal": bool(np.array_equal(a, b)),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "sum_abs": float(diff.sum()),
    }


def main() -> None:
    args = parse_args()
    tfrec_path = args.tfrec.resolve()
    idx_path = args.idx.resolve() if args.idx is not None else infer_idx_path(tfrec_path)

    offsets = np.loadtxt(str(idx_path), dtype=np.int64, usecols=0, max_rows=args.n, ndmin=1)
    offsets = np.atleast_1d(offsets).astype(np.int64).tolist()
    seq_records = list(tf.data.TFRecordDataset([str(tfrec_path)]).take(len(offsets)).as_numpy_iterator())

    rows = []
    all_equal = True

    for i, (offset, record_seq) in enumerate(zip(offsets, seq_records)):
        record_idx = read_record_at(tfrec_path, int(offset))
        img_idx, label_idx, name_idx = parse_example(record_idx)
        img_seq, label_seq, name_seq = parse_example(record_seq)
        rgb_idx = pil_rgb(img_idx)
        rgb_seq = pil_rgb(img_seq)

        row = {
            "i": i,
            "offset": int(offset),
            "name_idx": name_idx,
            "name_seq": name_seq,
            "label_idx": label_idx,
            "label_seq": label_seq,
            "record_equal": bool(record_idx == record_seq),
            "image_bytes_equal": bool(img_idx == img_seq),
            "rgb_equal": bool(np.array_equal(rgb_idx, rgb_seq)),
            "rgb_sum_abs": int(np.abs(rgb_idx.astype(np.int16) - rgb_seq.astype(np.int16)).sum()),
        }
        rows.append(row)
        all_equal = all_equal and row["record_equal"] and row["image_bytes_equal"] and row["rgb_equal"]

    first_offset = offsets[0]
    first_record = read_record_at(tfrec_path, int(first_offset))
    first_image_bytes, first_label, first_name = parse_example(first_record)

    pil_arr = pil_rgb(first_image_bytes)
    tf_arr = tf_rgb(first_image_bytes)

    base_tensor = preprocess_inception(resize_pil(pil_arr, args.img_size))
    opt_like_arr = np.asarray(center_crop_pil(Image.fromarray(pil_arr), args.fundus_crop_ratio), dtype=np.uint8)
    opt_like_tensor = preprocess_inception(resize_pil(opt_like_arr, args.img_size))

    result = {
        "tfrec": str(tfrec_path),
        "idx": str(idx_path),
        "n_checked": len(rows),
        "all_equal": all_equal,
        "first_sample": {
            "offset": int(first_offset),
            "image_name": first_name,
            "label": first_label,
            "pil_vs_tf_decode_rgb": summarize_diff(pil_arr, tf_arr),
            "base_vs_opt_like": summarize_diff(base_tensor, opt_like_tensor),
            "base_stats": {
                "min": float(base_tensor.min()),
                "max": float(base_tensor.max()),
                "mean": float(base_tensor.mean()),
            },
            "opt_like_stats": {
                "min": float(opt_like_tensor.min()),
                "max": float(opt_like_tensor.max()),
                "mean": float(opt_like_tensor.mean()),
            },
        },
        "rows": rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
