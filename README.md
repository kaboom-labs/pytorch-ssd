# SSD-based Object Detection in PyTorch

The provenance for this code is: 
+ [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd): implementation of [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) in PyTorch, using MobileNet backbones. It has out-of-box support for Google Open Images dataset.
+ [dusty-nv/pytorch-ssd](https://github.com/dusty-nv/pytorch-ssd): Used for training backend for [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference). Integrates into NVIDIA Jetson Object Detection capability. See **[Hello AI World](https://github.com/dusty-nv/jetson-inference/tree/dev#training)** tutorial:
> [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/dev/docs/pytorch-ssd.md)
+ *This repository* [kaboom-labs/pytorch-ssd](https://github.com/kaboom-labs/pytorch-ssd): Train on COCO, Improved Training Checkpoint, 

# kaboom-labs/pytorch-ssd

**Get started: clone this repo with submodules**
```bash
git clone --recurse-submodules git://github.com/kaboom-labs/pytorch-ssd.git
```

## Use COCO dataset to train Object Detection

**Pro tip: just run `./setup_coco` and skip steps 1~3.**

### Step 1. Download the COCO 2017 Object Detection dataset
```bash
python3 coco_downloader.py --path coco-data
```
will download the 2017 image object detection zip files to 'coco-data' folder and unzip them.

### 2. (Optional) Check image files for corrupted data
```bash
python3 validate_image_files.py --path coco-data/train2017
python3 validate_image_files.py --path coco-data/test2017
python3 validate_image_fiels.py --path coco-data/val2017
```
This will read the files using scikit-image and filter out any that are corrupt or otherwise unreadable.
Checking for corrupt images is a good idea before feeding them into PyTorch.
There should be no problematic images. If this script finds more than a few corrupt images, there might be a problem with your computer.

### 3. Fix COCO annotations

[Download and make cocoapi python](https://github.com/cocodataset/cocoapi)

COCO 2017 consists of 80 categories of objects.
However, due to historical reasons, the 'id' number of each category does not line up with the number of categories.
This becomes a problem in PyTorch MultiBox Loss.
The script `fix_coco_annotations.py` addresses this problem by re-assigning zero-indexed sequential 'id' numbers to each category for every boudning box annotation.
```bash
python3 fix_coco_annotations.py --path coco-data
```

### 4. Install `pycocotools`

Navigate to cocoapi/PythonAPI and make
```bash
cd cocoapi/PythonAPI
make
```
If it fails root access is required, you might need to install numpy and Cython on the root account with `pip install numpy Cython`

### 5. All set! Train your model with COCO

Example arguments to train_ssd.py:
```bash
python3 train_ssd.py \
--dataset-type=coco \
--datasets coco-data \
--net mb2-ssd-lite \
--epochs 100 \
--workers 12 \ # match to CPU cores for faster performance
```
