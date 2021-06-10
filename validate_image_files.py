import os
import glob
import sys
import time
import logging
import subprocess
import argparse
from pathlib import Path
import concurrent.futures

from skimage import io
from tqdm import tqdm
import cv2


def validate_pic(path):
    try:
        img = cv2.imread(path)
        img  = io.imread(path)
        del img

        logging.info(f"Checking image at {path}")
        return None
    except:
        logging.warning(f"Corrupted file detected at: {path}")
        dirname, filename = os.path.split(path)
        corrupted_path = os.path.join(dirname+'_invalid', filename)
        os.rename(path, corrupted_path)
        return corrupted_path 
def rewrite_pic(path):
    try:
        img = io.imread(path)
        io.imsave(path, img)
        del img
        return None
    except:
        logging.warning(f"Corrupted file detected at: {path}")
        dirname, filename = os.path.split(path)
        corrupted_path = os.path.join(dirname+'_invalid', filename)
        try:
            os.rename(path, corrupted_path)
        except FileNotFoundError:
            logging.warning(f"FileNotFoundError: file probably already moved by validation step")
            pass
        return corrupted_path 

def delete_pics(paths: list):
    if len(paths) < 1:
        logging.info("Nothing to delete")
        return
    print("Please review these images to delete...")
    time.sleep(1)
    dirname, _ = os.path.split(paths[0])
    subprocess.Popen(["nautilus", dirname])
    DELETE_IMAGES = yes_or_no(f"Delete these files right now?")
    if DELETE_IMAGES:
        for path in paths:
            os.remove(path)
        os.rmdir(dirname)

def yes_or_no(question):
    reply = str(input(question+ '(y/n): ')).lower().strip()
    if len(reply) < 1:
        return yes_or_no("Please enter an answer")
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("Please enter an answer")
def main(args):
    path = os.path.abspath(args.path)

    # create directory to temporarily keep corrupt images
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)
    path_for_corrupted =  os.path.join(dirname, basename+'_invalid')
    if not os.path.exists(path_for_corrupted):
        os.mkdir(path_for_corrupted)

    images = []
    for filename in os.listdir(path):
        image_path = os.path.join(path, filename)
        images.append(image_path)

    if args.num_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            results = list(tqdm(executor.map(validate_pic, images), total=len(os.listdir(path)), desc='Validating images'))
    else:
        results = []
        for image in tqdm(images, desc='Validating images'):
           results.append(rewrite_pic(image))

    problem_paths = [x for x in results if x is not None] 

    delete_pics(problem_paths)



if __name__=="__main__":
    parser=argparse.ArgumentParser(
            description = "Validate images by reading with scikit-image. Option to delete any that aren't.")
    parser.add_argument('--path', type=str, help='Path to folder containing images')
    parser.add_argument('--num-workers', default=12, type=int, help='Number of concurrent workers. Match with number of CPU cores.')
    args = parser.parse_args()

    main(args)

