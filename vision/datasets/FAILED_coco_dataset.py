import numpy as np
import cv2
import torch
import logging
import os
import copy

import json
import csv
import pandas as pd

from tqdm import tqdm
import requests

from icecream import ic
import time

class COCODataset():
    '''
    PyTorch Dataset class for COCO. 
    Must implement '__len__()' and '__getitem__()'. Other methods are auxiliary.
    '''

    def __init__(self, root, transform=None, target_transform=None, dataset_type="train"):
        self.root = os.path.abspath(root)
        self.class_stat = None
        self.image_id_filename_dict = {} 
        self.image_id_boxes_dict = {} # e.g. {391895 : [[199.84, 200.46, 77.71, 70.88],[234.22, 317.11, 149.39, 38.55]]}
        self.image_id_labels_dict = {} # e.g. {391895: [1,10,2]}
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.ids = [info['image_id'] for info in self.data]

    def _getitem(self, index):
        # called every time the dataloader needs to load a new batch
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])
        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels

    def get_annotation(self, index):
        """To conform the eval_ssd implementation that is based on the VOC dataset."""
        image_id, image, boxes, labels = self._getitem(index)
        is_difficult = np.zeros(boxes.shape[0], dtype=np.uint8)
        return image_id, (boxes, labels, is_difficult)

    def get_image(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        if self.transform:
            image, _ = self.transform(image)
        return image

    def _read_data(self):
        # runs at initialization to parse annotation json and return metadata
        annotation_json_path = os.path.join(self.root,"annotations", f"instances_{self.dataset_type}2017.json")
        logging.info(f"Loading annotations from {annotation_json_path}")

        with open(annotation_json_path, 'r') as j:
            annotation_json = json.loads(j.read())

        class_names =  []
        class_dict = {}
        annot_data = []

        for category in annotation_json['categories']:
            class_dict.update({category['id']:category['name']})
            class_names.append(str(category['name']))

        # image_id:filename dictionary for quicker lookup
        for info in annotation_json['images']:
            self.image_id_filename_dict.update({info['id']:info['file_name']})

        # image_id:boxes dictionary to group bbox's for each image
        # since json stores every instance (bbox) as separate dictionary entry
        for annotation in annotation_json['annotations']:
            image_id = int(annotation['image_id'])
            bbox = annotation['bbox']
            label = annotation['category_id']

            image_filename = self.image_id_filename_dict[image_id]
            img_path = os.path.join(self.root, self.dataset_type + '2017' , image_filename)
            
            # check if file exists
            if os.path.isfile(img_path) is False:
                logging.error(f"Missing ImageID {image_id} - dropping from annotations")
                continue

            if image_id not in self.image_id_labels_dict.keys():
                self.image_id_labels_dict.update({image_id:[annotation['category_id']]})
                self.image_id_boxes_dict.update({image_id:[annotation['bbox']]})
            else:
                labels = self.image_id_labels_dict[image_id]
                labels.append(label)
                self.image_id_labels_dict.update({image_id: labels}) 

                boxes = self.image_id_boxes_dict[image_id]
                boxes.append(bbox)
                self.image_id_boxes_dict.update({image_id: boxes})

            annot_data.append({
                'image_id' : int(image_id),
                'boxes': np.array(self.image_id_boxes_dict[image_id], dtype=np.float32),
                'labels': np.array(self.image_id_labels_dict[image_id], dtype='int64'),
                })

        print(f"Number of images: {len(annot_data)}")

        return annot_data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_dict[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)
    def _read_image(self, image_id):
        image_filename = self.image_id_filename_dict[image_id]
        image_file = os.path.join(self.root,self.dataset_type + '2017',image_filename)
        image = cv2.imread(image_file)
        #print(image)
        if image is None:
            print("Image is None")
            import IPython; IPython.embed();exit(1)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        logging.error("_balance_data method has not been implemented for COCO")
