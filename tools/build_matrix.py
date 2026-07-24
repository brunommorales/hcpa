# -*- coding: utf-8 -*-
"""
Constroi a matriz 2x2 de validacao externa reciproca (treino x teste) para
tensorflow_opt / InceptionV3:

           TESTE: HCPA            TESTE: DDR
  TREINO HCPA   in-domain          cross (HCPA->DDR)
  TREINO DDR    cross (DDR->HCPA)  in-domain

Avalia os MESMOS 2 checkpoints GH200 (HCPA-trained, DDR-trained) nos 2 test sets
locais -> matriz totalmente consistente. Replica o pipeline de teste do
tensorflow_opt (decode_jpeg -> central_crop 0.9 -> inception preprocess).
"""
import os, glob, json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import applications
from sklearn.metrics import roc_auc_score, average_precision_score

IMAGE_SIZE = (299, 299)
PREPROCESS = applications.inception_v3.preprocess_input
BASE = os.path.expanduser('~/projects/hcpa')

CKPTS = {
    'HCPA': f'{BASE}/new_results/checkpoints/hcpa/best.ckpt',
    'DDR':  f'{BASE}/new_results/checkpoints/ddr/best.ckpt',
}
TESTSETS = {
    'HCPA': f'{BASE}/data/all-tfrec',
    'DDR':  f'{BASE}/data/china_dataset/DDR_dataset/DDR-tfrec',
}

def parse(example):
    feat = {"imagem": tf.io.FixedLenFeature([], tf.string),
            "retinopatia": tf.io.FixedLenFeature([], tf.int64)}
    ex = tf.io.parse_single_example(example, feat)
    img = tf.cast(tf.reshape(tf.image.decode_jpeg(ex['imagem'], channels=3), [*IMAGE_SIZE, 3]), tf.float32)
    return img, tf.cast(ex['retinopatia'], tf.int32)

def crop_norm(img, label, ratio=0.9):
    img = tf.image.resize(tf.image.central_crop(img, ratio), IMAGE_SIZE, method='bilinear')
    return PREPROCESS(img), label

def build_model():
    inp = keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    base = applications.InceptionV3(weights=None, include_top=False, input_shape=(*IMAGE_SIZE, 3))
    x = keras.layers.GlobalAveragePooling2D()(base(inp))
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return keras.Model(inp, x)

def spec_at_sens(labels, probs, target=0.95):
    # varre thresholds; acha o ponto com sens>=target e maior spec
    order = np.argsort(-probs)
    P = labels.sum(); N = len(labels) - P
    best = 0.0
    for t in np.unique(probs):
        pred = probs >= t
        tp = np.sum((pred == 1) & (labels == 1)); fp = np.sum((pred == 1) & (labels == 0))
        sens = tp / P if P else 0; spec = 1 - (fp / N if N else 0)
        if sens >= target:
            best = max(best, spec)
    return best

def eval_one(model, tfrec_dir):
    files = sorted(glob.glob(f'{tfrec_dir}/test*.tfrec'))
    ds = (tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
          .map(parse, num_parallel_calls=tf.data.AUTOTUNE)
          .map(crop_norm, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(64).prefetch(tf.data.AUTOTUNE))
    P, L = [], []
    for xb, yb in ds:
        P.append(model(xb, training=False).numpy().reshape(-1)); L.append(yb.numpy().reshape(-1))
    probs = np.concatenate(P); labels = np.concatenate(L)
    return {
        'auc': float(roc_auc_score(labels, probs)),
        'ap': float(average_precision_score(labels, probs)),
        'spec_at_sens95': float(spec_at_sens(labels, probs)),
        'n': int(len(labels)), 'pos': int(labels.sum()),
    }

results = {}
model = build_model()
for train_name, ckpt in CKPTS.items():
    model.load_weights(ckpt).expect_partial()
    print(f"[matrix] loaded {train_name}-trained checkpoint")
    for test_name, tdir in TESTSETS.items():
        r = eval_one(model, tdir)
        kind = 'in-domain' if train_name == test_name else 'cross'
        results[f'{train_name}->{test_name}'] = {**r, 'train': train_name, 'test': test_name, 'kind': kind}
        print(f"  {train_name:4s} -> {test_name:4s} [{kind:9s}] AUC={r['auc']:.4f} "
              f"AP={r['ap']:.4f} Spec@95={r['spec_at_sens95']:.4f} (n={r['n']}, pos={r['pos']})")

out = f'{BASE}/new_results/comparison/matrix_results.json'
json.dump(results, open(out, 'w'), indent=2)
print(f"\nsalvo: {out}")
