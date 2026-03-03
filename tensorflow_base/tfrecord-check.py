import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import argparse
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='This script checks whether the tfrecord files were generated correctly.')
parser.add_argument("-t", "--tfrec_dir", help="Directory where the tfrecord files reside.", default="./tfrec")
parser.add_argument("-b", "--batch", help="Batch size to load (default=256).", default = 256)
parser.add_argument("-i", "--img_pixel", help="Diameter of the images (in pixels) (default=299).", default= 299)

args = parser.parse_args()
TFREC_DIR = str(args.tfrec_dir)
BATCH_SIZE = args.batch
IMG_PIXEL = args.img_pixel

# verifica conteudo tfrecords
IMAGE_SIZE= [IMG_PIXEL,IMG_PIXEL]
AUTO = tf.data.experimental.AUTOTUNE

def has_tfrecord_files(directory):
    return any(file.endswith(".tfrec") for file in os.listdir(directory))


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "imagem": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        # 'patient_id' : tf.io.FixedLenFeature([], tf.int64), 
        # 'side' : tf.io.FixedLenFeature([], tf.int64),
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'retinopatia' : tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    
    image = decode_image(example['imagem'])
    label = tf.cast(example['retinopatia'], tf.int32)
    name = example['image_name']

    
    return image, label, name

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def get_training_dataset(filenames):
    dataset = load_dataset(filenames, labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

# def show_batch_v1(image_batch, label_batch, name, dataset):
#         plt.figure(figsize=(5, 5))
#         for n in range(25):
#             ax = plt.subplot(5, 5, n + 1)
#             plt.imshow(image_batch[n] / 255.0)
#             # if label_batch[n]:
#             plt.title(label_batch[n])
#             # else:
#                 # plt.title("Não encaminha")
#             plt.axis("off")
#             plt.savefig(dataset + ".png", format="png", bbox_inches="tight")
        
#         plt.close()

def show_batch_v2(image_batch, label_batch, name, dataset):
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for n, ax in enumerate(axes.flat):
        ax.imshow(image_batch[n] / 255.0)
        ax.set_title(label_batch[n])
        ax.axis("off")

    # Save the figure without displaying it
    plt.savefig(dataset + ".png", format="png", bbox_inches="tight")
    # Close the figure to free up resources
    plt.close()

if not os.path.exists(TFREC_DIR):
    print(f"The directory {TFREC_DIR} does not exist.")
    sys.exit()

if not has_tfrecord_files(TFREC_DIR):
    print("The directory does not contain TFRecord files.")
    sys.exit()

train_filenames = tf.io.gfile.glob(os.path.join(TFREC_DIR,'train*.tfrec'))
valid_filenames = tf.io.gfile.glob(os.path.join(TFREC_DIR,'test*.tfrec'))
print('Number of images in train tfrecords: ', count_data_items(train_filenames))
print('Number of images in test tfrecords: ', count_data_items(valid_filenames))

filenames = [train_filenames, valid_filenames]

# print a set of images from train and valid datasets
dataset = "train"
for files in filenames:

    data = get_training_dataset(files)

    image_batch, label_batch, name = next(iter(data))
    
    show_batch_v2(image_batch.numpy(), label_batch.numpy(), name.numpy(), dataset)
    dataset = "test"