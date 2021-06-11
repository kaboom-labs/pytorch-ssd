python3 coco_downloader.py --path coco-data
python3 validate_image_files.py --path coco-data/train2017
python3 validate_image_files.py --path coco-data/test2017
python3 validate_image_files.py --path coco-data/val2017
python3 fix_coco_annotations.py --path coco-data

