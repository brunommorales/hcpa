# -*- coding: utf-8 -*-
"""
Cross-dataset inference: best tensorflow_opt InceptionV3 checkpoint (trained on
EyePACS / all-tfrec) evaluated on the DDR (China) test set.

Replicates the exact tensorflow_opt test pipeline:
  decode_jpeg -> central_crop(0.9)+resize 299 -> inception_v3.preprocess_input
  model: InceptionV3(include_top=False) -> GAP -> Dropout(0.2) -> Dense(1,sigmoid)
"""
import os, sys, glob, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # CPU (broken CUDA driver on this node)
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # checkpoint is a Keras 2 / tf.train.Checkpoint

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import applications
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

IMAGE_SIZE = (299, 299)
PREPROCESS = applications.inception_v3.preprocess_input

def parse(example):
    feat = {"imagem": tf.io.FixedLenFeature([], tf.string),
            "retinopatia": tf.io.FixedLenFeature([], tf.int64)}
    ex = tf.io.parse_single_example(example, feat)
    img = tf.image.decode_jpeg(ex['imagem'], channels=3)
    img = tf.reshape(img, [*IMAGE_SIZE, 3])
    img = tf.cast(img, tf.float32)
    label = tf.cast(ex['retinopatia'], tf.int32)
    return img, label

def crop_norm(img, label, ratio=0.9):
    cropped = tf.image.central_crop(img, ratio)
    resized = tf.image.resize(cropped, IMAGE_SIZE, method='bilinear')
    return PREPROCESS(resized), label

def build_model():
    inp = keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    base = applications.InceptionV3(weights=None, include_top=False,
                                    input_shape=(*IMAGE_SIZE, 3))
    x = base(inp)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return keras.Model(inputs=inp, outputs=x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tfrec_dir', default=os.path.expanduser(
        '~/projects/hcpa/data/china_dataset/DDR_dataset/DDR-tfrec'))
    ap.add_argument('--split', default='test')
    ap.add_argument('--ckpt', default=os.path.expanduser(
        '~/projects/hcpa/tensorflow_opt/results/'
        'result770699_tupi_1xnvidia-geforce-rtx-4090_bs96/run_0/checkpoints/best.ckpt'))
    ap.add_argument('--batch_size', type=int, default=64)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.tfrec_dir, f'{args.split}*.tfrec')))
    print(f"[cross-eval] {len(files)} tfrec files ({args.split}) in {args.tfrec_dir}")

    ds = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(crop_norm, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = build_model()
    model.load_weights(args.ckpt).expect_partial()
    print(f"[cross-eval] loaded weights: {args.ckpt}")

    probs, labels = [], []
    for i, (xb, yb) in enumerate(ds):
        p = model(xb, training=False).numpy().reshape(-1)
        probs.append(p); labels.append(yb.numpy().reshape(-1))
        print(f"\r  batch {i+1} ({sum(len(x) for x in labels)} imgs)", end='')
    print()
    probs = np.concatenate(probs); labels = np.concatenate(labels)

    auc = roc_auc_score(labels, probs)
    ap_score = average_precision_score(labels, probs)
    acc = accuracy_score(labels, (probs >= 0.5).astype(int))
    pos = int(labels.sum()); n = len(labels)
    print("\n========== CROSS-DATASET RESULT (DDR / China) ==========")
    print(f" model      : tensorflow_opt InceptionV3 (trained on EyePACS)")
    print(f" images     : {n}  (pos={pos} / neg={n-pos}, prevalence={pos/n:.3f})")
    print(f" ROC-AUC    : {auc:.4f}")
    print(f" PR-AUC(AP) : {ap_score:.4f}")
    print(f" Accuracy@0.5: {acc:.4f}")
    print(f" --- reference: EyePACS in-domain val AUC = 0.967 ---")

if __name__ == '__main__':
    main()
