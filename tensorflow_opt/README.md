# HCPA scrips

This repository contains the source-codes, and scripts from our work: 

## Requirements
```shell
python3 -m pip install -r requirements.txt
```
## Run

### To preprocess data before creating the tfrecord files
This script pre-processes retinal fundus images, centering the fundus and resizing it to a defined pixel size (-d) to standardize them and later create tfrecord files. See more: python3 preprocess_data.py -h.
```shell
python3 preprocess_data.py -i data/hcpa21-images/ -o data/test1/ -d 299
```

### To create the tfrecord files
This script creates tfrecord files based on already processed and standardized images. See more: python3 create-tfrecord.py -h
```shell
python3 create-tfrecord.py --base_dir data --labels_train hcpa-all-train.csv --labels_test hcpa-all-test.csv --data_dir hcpa-all-images --tfrec_dir hcpa-all --size 100
```
Please adjust the size based on the quantity of images. Ex.: 1000 images, size 100 | 3000 images, size 300.

#### OBS.- the csv columns are: imagem,retinopatia

*imagem* is the name of each image that links to a file in --data_dir

*retinopatia* is the label (0 or 1); if the data have categories 0,1,2,3,4, needs to map to 0,1. This means you must change all 0s and 1s to 0 and 2s,3s, and 4s labels to 1. 

The application is to decide if to forward a patient to a specialist. Therefore, all patients with mild or no retinopathy move to 0, and patients with moderate, severe, or proliferative retinopathy move to 1. In addition, in some cases, it is necessary to add the image extension (.jpg) in the CSV files.

ex.
| imagem | retinopatia |
| :---:   | :---: | 
| e4dcca36ceb4.jpg | 0   | 
| e4e343eaae2a.jpg | 1   | 

In this cade, for second image the label is 2, so we change to 1.

### To verify that tfrecords were generated correctly
This script checks whether the tfrecord files were generated correctly.
```shell
python3 tfrecord-check.py -t data/test-hcpa21-tfrec/
```

### To model train use:
```shell
python3 dr_hcpa_v2_2024.py --tfrec_dir data/all --dataset all --results results/all

python3 dr_hcpa_v2_2024.py --help
```

## Authors

👤 **Cristiano Alex Künas**

> ResearchGate: [Cristiano Künas](https://www.researchgate.net/profile/Cristiano-Kunas)

👤 **Thiago da Silva Araújo**

> ResearchGate: [Thiago Araújo](https://www.researchgate.net/profile/Thiago-Araujo-36)

## 📝 License

Copyright © [GPPD](http://www.inf.ufrgs.br/gppd/site/) 2024.
