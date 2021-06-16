'''
Fix COCO Annotations

MIT License
Copyright 2021 Jason Sohn https://github.com/tensorturtle

Short version: COCO labels are not indexed sequentially, so this fixes it.

Long version: The COCO Object Detection (bounding box) dataset was labeled with 91 labels. After the labeling process was completed, some of the labels were deleted on purpose. (I don't know the reason). Now there are 80 labels, but the label indexes have not been changed from the 91 maximum. This creates friction when dealing with dictionaries/lists. Most importantly, PyTorch MultiBox Loss does not accept label indexes higher than the total number of labels. It raises 'RuntimeError: CUDA runtime error 'Assertion 't >= 0 && tt < n_classes' failed'.
'''
import sys
import os
import time
import json
import copy
import shutil
import logging
import argparse

logging.basicConfig(level=logging.DEBUG)

def fix_category_id(coco_dict: dict):
    coco_dict = copy.deepcopy(coco_dict)
    
    coco_categories_dict = coco_dict['categories']
    coco_annotations_dict = coco_dict['annotations']
    
    # build dictionary that correlates 91-class annotations to one-indexed 80-class annotations
    cat_convert = {}
    
    # exit if dictionary appears to have been modified already
    if coco_categories_dict[-1]['id'] != 90:
        logging.error("Idempotency Warning: It looks like you have already run this script. Do not run this script more than once.")
        sys.exit(1)
        
    i = 1
    for cat in coco_categories_dict:

        cat_convert.update({cat['id']: i}) # create simpler dictionary for later use
        cat.update({'id':i}) # rename the dictionary itself too
        i += 1
    
    # change the 'category_id' on every single bbox
    for annotation in coco_annotations_dict:
        annotation.update({'category_id' : cat_convert[annotation['category_id']]})
    
    return coco_dict

def move_empty_images(coco_dict: dict, coco_root: str, dataset_type: str):
    # THIS IS A DANGEROUS FUNCTION: it breaks pytorch dataloader because it only moves files without unlinking the annotation file
    coco_dict = copy.deepcopy(coco_dict)
    
    all_images = set()
    for image in coco_dict['images']:
        all_images.add(image['id'])
    logging.info(f"Total number of images: {len(all_images)}")
    
    annotated_images = set()
    for annotation in coco_dict['annotations']:
        annotated_images.add(annotation['image_id'])
    logging.info(f"Images with at least one annotation: {len(annotated_images)}")
            
    no_annotation_images = all_images - annotated_images
    logging.info(f"Images without any annotations: {len(no_annotation_images)}")
    
    # build image_id to filename dictionary
    id_to_filename = {}
    for image in coco_dict['images']:
        id_to_filename.update({image['id']:image['file_name']})
    
    # move files to new directory
    rejects_dir = os.path.join(coco_root, 'no_annotations_'+dataset_type+'2017')
    try: 
        os.mkdir(rejects_dir)
    except OSError:
        logging.error("Idempotency Warning: It appears you have already run this script. Do not run this script more than once.")
        sys.exit(1)
        
    counter = 0
    for image_id in no_annotation_images:
        image_filename = id_to_filename[image_id]
        from_image_path = os.path.join(coco_root, f"{dataset_type}2017", image_filename)
        shutil.move(from_image_path, rejects_dir)
        counter += 1
    logging.info(f"{counter} images have been moved out of {from_image_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= "Fix COCO label indexes")
    parser.add_argument('--path', default='coco-data', type=str, help='Path to root of unzipped COCO dataset')
    args = parser.parse_args()

    coco_root = args.path
    dataset_types = ['train', 'val']

    for dataset_type in dataset_types:
        path = os.path.join(coco_root, 'annotations', f'instances_{dataset_type}2017.json')
        with open(path, 'r') as j:
            json_file = json.loads(j.read())
            
        print("Fixing COCO label indexes...")
        coco_dict = fix_category_id(json_file)
        print("Moving images with no annotations to new folder...")
        # WARNING: moving images causes problem.
        move_empty_images(coco_dict, coco_root, dataset_type)
        
        #rename old json
        shutil.move(path, path+'_old')
        
        with open(path, 'w') as j:
            json.dump(coco_dict, j)
