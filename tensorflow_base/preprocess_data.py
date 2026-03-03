# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
"""
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
"""
import argparse
import sys
import time
from shutil import rmtree
from glob import glob
from os import makedirs, rename, listdir
from os.path import join, exists, isfile
from lib.preprocess import resize_and_center_fundus
import concurrent.futures
import psutil

physical_cores = psutil.cpu_count(logical=False)

parser = argparse.ArgumentParser(description='This script pre-processes retinal fundus images, centering the fundus and resizing it to a defined pixel size (-p) to standardize them and later create tfrecord files. See more: python3 preporcess.py -h.')
parser.add_argument("-i", "--input_dir", help="Directory where the input images resides.", default="input")
parser.add_argument("-o", "--output_dir", help="Directory where the processed images will be saved.", default="output")
parser.add_argument("-w", "--workers", type=int, help="Number of processes that will pre-process images concurrently (default are the machine's physical cores).", 
                    default=physical_cores)
parser.add_argument("-d", "--diameter", type=int, help="Pixel diameter for final image (default= 299).", default=299)

args = parser.parse_args()
input_dir = str(args.input_dir)
output_dir = str(args.output_dir)
workers = args.workers
diameter = args.diameter

'''
Create directories for output.
'''
if exists(output_dir):
    rmtree(output_dir)
makedirs(output_dir)

'''
Create a tmp directory for saving temporary preprocessing files.
'''
tmp_path = join(input_dir, 'tmp')
if exists(tmp_path):
    rmtree(tmp_path)
makedirs(tmp_path)

failed_images = []
files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]

def process_images(file):
    basename = file.split(".png")[0]                # I changed it to .png because the data is in .png
    im_paths = glob(join(input_dir, "{}*".format(basename)))

    '''
    Find contour of eye fundus in image, and scale
    diameter of fundus to 299 pixels and crop the edges.
    '''
    res = resize_and_center_fundus(save_path=tmp_path,
                                    image_paths=im_paths,
                                    diameter=diameter, verbosity=0)

    if res != 1:
        failed_images.append(basename)

    new_filename = "{0}.jpg".format(basename)

    rename(join(tmp_path, new_filename),
        join(output_dir, new_filename))
    
    # Status message.
    msg = "\r- Preprocessing image: {0:>7}".format(new_filename)
    sys.stdout.write(msg)
    sys.stdout.flush()

tStart = time.time()
with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
    print("\r- Preprocessing...")
    executor.map(process_images, files)

tElapsed = round(time.time() - tStart, 1)
print('\n Time (sec) to preprocess: ', tElapsed)
'''
Clean tmp folder.
'''
rmtree(tmp_path)

print("\n Could not preprocess {} images.".format(len(failed_images)))
print(", ".join(failed_images))
