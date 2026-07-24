# -*- coding: utf-8 -*-
"""
Cross-inference: avalia uma LISTA de checkpoints tensorflow_opt (InceptionV3)
num test set tfrec arbitrario. Replica o pipeline de teste do tensorflow_opt
(decode_jpeg -> central_crop 0.9 + resize 299 -> inception preprocess_input).

Uso:
  cross_infer.py --ckpt-glob '.../run_*/checkpoints/best.ckpt' \
                 --tfrec-dir .../DDR-tfrec --split test --tag HCPAmodel_on_DDRdata

Salva: new_results/comparison/infer_<tag>.csv (uma linha por checkpoint + metricas)
"""
import os, sys, glob, argparse
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
    return keras.Model(inputs=inp, outputs=x)

def spec_at_sens(y, p, target=0.95):
    # varre thresholds, acha maior spec com sens >= target
    thr = np.unique(p)[::-1]
    best = 0.0
    for t in thr:
        pred = (p >= t).astype(int)
        tp = np.sum((pred == 1) & (y == 1)); fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0)); fp = np.sum((pred == 1) & (y == 0))
        sens = tp / (tp + fn + 1e-9); spec = tn / (tn + fp + 1e-9)
        if sens >= target and spec > best:
            best = spec
    return best

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    tp = np.sum((yhat==1)&(y==1)); fn = np.sum((yhat==0)&(y==1))
    tn = np.sum((yhat==0)&(y==0)); fp = np.sum((yhat==1)&(y==0))
    sens = tp/(tp+fn+1e-9); spec = tn/(tn+fp+1e-9)
    prec = tp/(tp+fp+1e-9); f1 = 2*prec*sens/(prec+sens+1e-9)
    return dict(auc=roc_auc_score(y,p), ap=average_precision_score(y,p),
                sens=sens, spec=spec, precision=prec, f1=f1,
                spec_at_sens95=spec_at_sens(y,p,0.95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-glob', required=True)
    ap.add_argument('--tfrec-dir', required=True)
    ap.add_argument('--split', default='test')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.tfrec_dir, f'{args.split}*.tfrec')))
    ds = (tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
          .map(parse, num_parallel_calls=tf.data.AUTOTUNE)
          .map(crop_norm, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(args.batch_size).prefetch(tf.data.AUTOTUNE))
    # materializa uma vez (reusa entre checkpoints)
    Xs, ys = [], []
    for xb, yb in ds:
        Xs.append(xb.numpy()); ys.append(yb.numpy())
    y = np.concatenate(ys)
    print(f"[cross-infer] {args.tag}: {len(files)} tfrec, {len(y)} imgs (pos={int(y.sum())})", flush=True)

    ckpts = sorted({c.replace('.index','').replace('.data-00000-of-00001','')
                    for c in glob.glob(args.ckpt_glob + '*') if '.ckpt' in c})
    model = build_model()
    out_rows = []
    for i, ck in enumerate(ckpts):
        model.load_weights(ck).expect_partial()
        probs = np.concatenate([model(xb, training=False).numpy().reshape(-1) for xb in Xs])
        m = metrics(y, probs)
        out_rows.append(m)
        print(f"  ckpt {i} ({os.path.basename(os.path.dirname(os.path.dirname(ck)))}): "
              f"AUC={m['auc']:.4f} sens={m['sens']:.3f} spec={m['spec']:.3f} spec@95={m['spec_at_sens95']:.3f}", flush=True)

    import csv
    out = os.path.expanduser(f'~/projects/hcpa/new_results/comparison/infer_{args.tag}.csv')
    keys = ['auc','ap','sens','spec','precision','f1','spec_at_sens95']
    with open(out,'w',newline='') as f:
        w = csv.writer(f); w.writerow(['ckpt']+keys)
        for i,m in enumerate(out_rows): w.writerow([i]+[f"{m[k]:.6f}" for k in keys])
    print(f"\n=== {args.tag}: media +- desvio (n={len(out_rows)}) ===")
    for k in keys:
        v=[m[k] for m in out_rows]; print(f"  {k:15s}= {np.mean(v):.4f} +- {np.std(v):.4f}")
    print(f"salvo: {out}")

if __name__ == '__main__':
    main()
