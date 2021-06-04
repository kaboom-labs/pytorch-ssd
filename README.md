# SSD-based Object Detection in PyTorch

This repo implements [SSD (Single Shot MultiBox Detector)](https://arxiv.org/abs/1512.02325) in PyTorch for object detection, using MobileNet backbones.  It also has out-of-box support for retraining on Google Open Images dataset.  

> For documentation, please refer to Object Detection portion of the **[Hello AI World](https://github.com/dusty-nv/jetson-inference/tree/dev#training)** tutorial:
> [Re-training SSD-Mobilenet](https://github.com/dusty-nv/jetson-inference/blob/dev/docs/pytorch-ssd.md)

Thanks to @qfgaohao for the upstream implementation from:  [https://github.com/qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)

# How to use COCO dataset to (pre)train

## 1. Download the 2017 Object Detection dataset using `coco_downloader.py`
```bash
python3 coco_downloader.py --path coco-data
```
will download necessary zip files to 'coco-data' folder and unzip them.

## 2. (Optional) Check image files for corrupted data using `validate_image_files.py`
```bash
python3 validate_image_files.py --path coco-data/train2017
python3 validate_image_files.py --path coco-data/test2017
python3 validate_image_fiels.py --path coco-data/val2017
```
This will read the files using scikit-image and filter out any that are corrupt or otherwise unreadable.
Checking for corrupt images is a good idea before feeding them into PyTorch.
There should be no problematic images. If this script finds many corrupt images, there might be a problem with your computer.

## 3. Fix COCO annotations
COCO 2017 consists of 80 categories of objects.
However, due to historical reasons, the 'id' number of each category does not line up with the number of categories.
This becomes a problem in PyTorch MultiBox Loss.
The script `fix_coco_annotations.py` addresses this problem by re-assigning zero-indexed sequential 'id' numbers to each category for every boudning box annotation.
```bash
python3 fix_coco_annotations.py --path coco-data
```

## 4. All set! Train your model
Remember to set argument `--dataset-type=coco` in `train_ssd.py` to use the new COCODataset PyTorch Dataset class I wrote. 
