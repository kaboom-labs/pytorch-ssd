# stdlibs
import time
import logging
import os
import sys
import argparse
import functools
import pathlib
import shutil
import requests
import zipfile
import concurrent.futures

from tqdm.auto import tqdm



def download(url, root):
    print(f"Downloading from {url}")
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(root).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path   

def unzip(path_from, path_to):
    print(f"Unzipping {path_from} to {path_to}")
    with zipfile.ZipFile(path_from, 'r') as zip_ref:
        zip_ref.extractall(path_to)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description= "Download COCO 2017 training/validation/test datasets. Note that ground truth for test image is not public. Results must be uploaded to https://cocodataset.org/#upload for accuracy results")

    parser.add_argument("--root", "--data", type=str, default="coco-data", help="The root directory that you want to store the image and annotation data")
    parser.add_argument("--num-workers", "--workers", type=int, default=4, help="Number of simultaneous downloads")

    args = parser.parse_args()

    DL_URLS = {
    "train2017.zip":"http://images.cocodataset.org/zips/train2017.zip", # train images
    "test2017.zip":"http://images.cocodataset.org/zips/test2017.zip", # test images 
    "val2017.zip":"http://images.cocodataset.org/zips/val2017.zip", # validation images
    "annotations_trainval2017.zip":"http://images.cocodataset.org/annotations/annotations_trainval2017.zip", #annotation information
    }
    root = os.path.abspath(args.root)

    # download zip files from cocodataset.org
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for folder_name in DL_URLS:
            executor.submit(download, DL_URLS[folder_name], os.path.join(root, folder_name))

    # unzip files
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        for folder_name in DL_URLS:
            executor.submit(unzip, os.path.join(root, folder_name), root)

