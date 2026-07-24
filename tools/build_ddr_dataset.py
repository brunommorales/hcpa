# -*- coding: utf-8 -*-
"""
Prepare the DDR DR_grading dataset to be used exactly like data/all-tfrec.

Pipeline (same crop/preprocess as tensorflow_opt/preprocess_data.py):
  1. resize_and_center_fundus (diameter=299) on every image -> DDR_processed/
  2. binary referable-DR mapping: grades {0,1}->0, {2,3,4}->1, grade 5 dropped
     (matches the convention used to build data/all-tfrec)
  3. write labels_train.csv / labels_test.csv (cols: imagem,retinopatia)
     train = DDR train + valid, test = DDR test
"""
import os, sys, time, concurrent.futures
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

HCPA = os.path.expanduser('~/projects/hcpa')
sys.path.insert(0, os.path.join(HCPA, 'tensorflow_opt'))
from lib.preprocess import resize_and_center_fundus

DDR = os.path.join(HCPA, 'data/china_dataset/DDR_dataset/DDR-dataset/DR_grading')
OUT = os.path.join(HCPA, 'data/china_dataset/DDR_dataset/DDR_processed')
DIAMETER = 299

# grade -> binary referable label; None == drop (ungradable)
BIN = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: None}

def read_split(name):
    rows = []
    with open(os.path.join(DDR, name + '.txt')) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fn, grade = line.split()
            rows.append((name, fn, int(grade)))
    return rows

def work(t):
    path, fn = t
    try:
        res = resize_and_center_fundus(save_path=OUT, image_paths=[path],
                                       diameter=DIAMETER, verbosity=0)
        if res != 1:
            return fn
    except Exception as e:
        sys.stderr.write(f"\n{fn}: {e}\n")
        return fn
    return None

def main():
    splits = {'train': read_split('train') + read_split('valid'),
              'test': read_split('test')}

    os.makedirs(OUT, exist_ok=True)
    # clean output
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))

    # collect all (src_dir, filename) to preprocess, skipping dropped grades
    tasks = []
    for split, rows in splits.items():
        for src, fn, grade in rows:
            if BIN[grade] is None:
                continue
            tasks.append((os.path.join(DDR, src, fn), fn))

    failed = set()
    t0 = time.time()
    n = len(tasks)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        for i, r in enumerate(ex.map(work, tasks, chunksize=16), 1):
            if r is not None:
                failed.add(r)
            if i % 200 == 0 or i == n:
                sys.stdout.write(f"\r- preprocessing {i}/{n}  failed={len(failed)}")
                sys.stdout.flush()
    print(f"\n preprocess done in {round(time.time()-t0,1)}s, failed={len(failed)}")

    # write CSVs (only images that produced an output file)
    for split, rows in splits.items():
        csv_path = os.path.join(OUT, f'labels_{split}.csv')
        kept = 0
        with open(csv_path, 'w') as w:
            w.write('imagem,retinopatia\n')
            for src, fn, grade in rows:
                lab = BIN[grade]
                if lab is None or fn in failed:
                    continue
                base = os.path.splitext(fn)[0] + '.jpg'
                if not os.path.exists(os.path.join(OUT, base)):
                    continue
                w.write(f'{base},{lab}\n')
                kept += 1
        print(f" {split}: wrote {kept} rows -> {csv_path}")

if __name__ == '__main__':
    main()
