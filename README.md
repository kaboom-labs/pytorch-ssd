# SSD-based Object Detection in PyTorch

**Get started: clone this repo with submodules**
```bash
git clone --recurse-submodules git://github.com/kaboom-labs/pytorch-ssd.git
```
**install required python packages**
```bash
cd pytorch-ssd
pip install -r requirements.txt
```

## Table of Contents
+ [A. Provenance for this code](#a.-provenance-for-this-code)
+ B. Use COCO dataset to train Object Detection
+ C. Comprehensive Checkpoints and Exact Resume
+ D. Multi-GPU Training
+ E. Albumentations; faster image augmentation

## A. Provenance for this code
+ [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd): initial implementation of [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) in PyTorch, using MobileNet backbones. It has out-of-box support for Google Open Images dataset.
+ [dusty-nv/pytorch-ssd](https://github.com/dusty-nv/pytorch-ssd): Used for training backend for [dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference). Integrates into NVIDIA Jetson Object Detection capability. See **[Hello AI World](https://github.com/dusty-nv/jetson-inference/tree/dev#training)** tutorial: [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/dev/docs/pytorch-ssd.md)
+ ***This repository*** [kaboom-labs/pytorch-ssd](https://github.com/kaboom-labs/pytorch-ssd): Train on COCO, 4.7x faster training than upstream repos.


## B. How to pretrain SSD object detector on COCO dataset

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
python3 validate_image_files.py --path coco-data/val2017
```
This will read the files using scikit-image and filter out any that are corrupt or otherwise unreadable.
Checking for corrupt images is a good idea before feeding them into PyTorch.
There should be no problematic images. If this script finds more than a few corrupt images, there might be a problem with your computer.

### 3. Fix COCO annotations

[Download and make cocoapi python](https://github.com/cocodataset/cocoapi)

COCO 2017 consists of 80 categories of objects.
However, due to historical reasons, the 'id' number of each category does not line up with the number of categories.
This becomes a problem in PyTorch MultiBox Loss.
The script `fix_coco_annotations.py` addresses this problem by re-assigning zero-indexed sequential 'id' numbers to each category for every bounding box annotation.
```bash
python3 fix_coco_annotations.py --path coco-data
```
### 4. Start training.

**To initialize training, define checkpoint folder path and other arguments**
```bash
python3 train_ssd.py \
--dataset-type=coco \
--datasets=coco-data \
--net=mb2-ssd-lite \
--epochs=100 \
--workers=12 \ # match to CPU cores for faster performance
--checkpoint-folder=models/my-experiment-1
```

## C. Comprehensive Checkpoints and Exact Resume

Upstream repos save model weights after each epoch as a `.pth` file.

However, saving only the net state dict is not enough to be able to resume training after stopping. In fact, the loss increases significantly and undoes many tens of epochs of training.
This is because the optimizer and learning rate scheduler states are not saved, and therefore reset.

The solution here is to save the model weights, optimizer state, learning rate scheduler state, and epoch number for each epoch, and also to have a JSON file in the checkpoint folder that contains the arguments.


**To resume training without any changes, pass in checkpoint folder path to --resume**

For example, resume from last saved epoch (77). The script will find the JSON file containing the arguments and use them.
```bash
python3 train_ssd.py --resume models/my-experiment-1/COMBO_CHECKPOINT...Epoch-77....pth
```
If you need to edit the arguments, directly edit the annotations JSON file.

## D. Tensorboard

While training, training (very frequent) and validation (epoch) losses are logged to `runs` directory. In the terminal, start `tensorboard --logdir runs` and go to http://localhost:6006 to see results.

## E. Multi-GPU Training

Automatic multi-GPU training is enabled.
Use the exact same script & weights to train on a multiple GPUs or a single GPU.

## F. Albumentations; faster image augmentation
Not only does the original image augmentations modify the image way too radically, it is slow.

I replaced it with [albumentations](https://github.com/albumentations-team/albumentations), an optimized image augmentation library.
Results:
+ 2.5x faster training, because CPU doesn't block GPU ops. Went from ~50% GPU usage to ~83%. 2000 secs/epoch to 800 secs/epoch on RTX 2070S
+ 50% less RAM usage

## G. Half precision training

40% faster training with NVIDIA Mixed Precision (AMP) & PyTorch integration. See: https://pytorch.org/blog/accelerating-training-on-nvidia-gpus-with-pytorch-automatic-mixed-precision/
