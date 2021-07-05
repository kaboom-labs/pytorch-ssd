import numpy as np
import pathlib
import cv2
import pandas as pd
import copy
import os
import logging

from ..utils.misc import xywh_to_xyxy


# visualization
from vision.utils.visualization import plot_image_grid, make_square
"""
Expected dataset structure:
root
    |
    --- video_name (frames and annotations are extracted from this .mp4)
        |--- det_labels
        |   |---[video_name_frameNumber].txt
        |       where each row is: "category x_min y_min x_max, y_max confidence"),
        |           where xyxy is normalized (0,1)
        |              .
        |              .
        |--- det_stills
            |---[stills_frameNumber].jpg
                    .
                    .
                    .
"""

class HydoDataset:

    def __init__(self, root, plus_categories=[cyclist], transform=None, target_transform=None,
    dataset_type="train", balance_data=False, viz_inputs=False):

    self.root = pathlib.Path(root)
    self.dataset_type = dataset_type.lower()
    self.transform = transform
    self.target_transform = target_transform

    self.data, self.class_names, self.class_dict = self._read_data()
    # self.data is a list whose length is number of images
    #     a given list item is a dictionary, with three keys: 'image_id':int, 'boxes':np.array, 'labels':np.array
    #           'boxes' has shape (N,4), where N is number of bounding boxes per image, and bounding box format is [xmin,ymin,xmax,ymax] (pixels)
    #           'labels' has shape (N)
    # self.class_names is a list of strings, where each string is name of category.
    #       Importantly, the first item must be "BACKGROUND"
    # self.class_dict is a dictionary, where {"category_name":int(consecutive number)}. It is just an enumerated self.class_names

    self.ids
    self.balance_data = balance_data
    self.min_image_num = -1
    if self.balance_data:
        self.data = self.balance_data()

    self.class_stat = None
    self.viz_inputs = viz_inputs

    def _read_data(self):
        summary_json = json.loads(self.data / "summary.json")
        class_dict = {'BACKGROUND':0} #SSD needs this 0th empty label for training
        class_dict.update(summary_json["categories"])
        class_names = [key for key:value in class_dict]
        data = []
        class_dict = {}
        video_name_dirs = [x for x in self.root if x.is_dir()]
        for video_name_dir in video_name_dirs:
            det_labels_paths = (video_name_dir / "det_labels").glob('*.txt')
            det_stills_paths = (video_name_dir / "det_stills").glob('*.jpg')
            assert len(det_labels_paths) == len(det_stills_paths) # number of label file should match number of images

        for det_labels_path in det_labels_paths:
            image_path = video_name_dir / Path(det_labels_path.name) / ".jpg"

            boxes = np.empty((0,4))
            labels = np.empty((0))
            with open(det_labels_path, 'r') as labels_txt:
                labels_lines = labels_txt.readlines()
                for line in labels_lines:
                    boxes = np.vstack(boxes,np.array([[line[1],line[2],line[3],line[4]]]), dtype='float32'))
                    labels = np.hstack(labels, np.array([line[0]], dtype='int64'))
            data.append({
                'image_path':image_path,
                'boxes':boxes,
                'labels':labels
            })
        return data, class_names, class_dict

    def _read_image(self,path)
        image = cv2.imread(path)
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _getitem(self,index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_path'])
        boxes = copy.copy(image_info['boxes'])
        labels = copy.copy(image_info['labels'])

        # Compare images before and after transflower
        if self.viz_inputs: #before
            image_before = copy.copy(image)
            sq_image_before = make_square(image_before, (0,0,0), 300) # change size for

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.viz_inputs: # after
            image_after = np.moveaxis(image.numpy(), 0, -1)
            image_overlay = copy.copy(image_after)
            for i in range(boxes.shape[0]):
                box = boxes[i]
                cv2.rectangle(image_overlay,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255,255,0),
                    4)
                label = f"{labels[i]}"
                cv2.putText(image_overlay,
                        label,
                        (int(box[0]) + 20, int(box[1]) + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, #font scale
                        (255,0,255),
                        2) # line type
            plot_image_grid([sq_image_before, image_after, image_overlay])
        width = image.shape[1]
        height = image.shape[2]

        # boxes normalized from pixels to [0,1] range
        width = image.shape[1] # ACTUALLY, not sure which one is which
        height = image.shape[2]
        #boxes[:,0] /= width
        #boxes[:,1] /= height
        #boxes[:,2] /= width
        #boxes[:,3] /= height

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)


        return image_info['image_id'], image, boxes, labels

    def __getitem__(self, index):
        _, image, boxes, labels = self._getitem(index)
        return image, boxes, labels
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name:0 for name in self.class_names[:]}
            for example in self.data:
                for class_index in example['labels:
                    class_names = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                    f"Number of Images: {len(self.data)}",
                    f"Minimum Number of Images for a Class: {self.min_image_num}",
                    f"Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}:{num}")
        return "\n".join(content)
