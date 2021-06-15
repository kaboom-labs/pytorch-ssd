import numpy as np
import pathlib
import cv2
import pandas as pd
import copy
import os
import logging
import json

class COCOTest:
    '''
    Minimal pytorch dataset class for loading COCO test dataset, 
    which has only images and no annotations.
    '''

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.info_json_path = os.path.join(root, 'annotations', 'image_info_test-dev2017.json')
        self.image_ids, self.image_paths = self._parse_json()

    def _parse_json(self):
        with open(self.info_json_path, 'r') as j:
            info_json = json.load(j)

        image_ids = []
        image_paths = []
        for info in info_json['images']:
            image_ids.append(int(info['id']))
            image_paths += str(os.path.join(self.root, 'test2017', info['file_name']))

        return image_ids, image_paths

    def __getitem__(self, index):
        image = self.read_image(index)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return f"COCO Test Dataset. Contains {len(self.filepaths)} images."

    def read_image(self, index):
        image_path = self.image_paths[index]
        # loading with opencv
        image = cv2.imread(image_path)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
