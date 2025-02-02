import numpy as np
import pathlib
import cv2
import pandas as pd
import copy
import os
import logging

#debugging
from vision.utils.visualization import plot_image_grid, make_square

class OpenImagesDataset:

    def __init__(self, root,
                 transform=None, target_transform=None,
                 dataset_type="train", balance_data=False, viz_inputs=False):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_type = dataset_type.lower()

        self.data, self.class_names, self.class_dict = self._read_data()
        self.balance_data = balance_data
        self.min_image_num = -1
        if self.balance_data:
            self.data = self._balance_data()
        self.ids = [info['image_id'] for info in self.data]

        self.class_stat = None
        self.viz_inputs = viz_inputs

    def _getitem(self, index):
        image_info = self.data[index]
        image = self._read_image(image_info['image_id'])
        
        # duplicate boxes to prevent corruption of dataset
        boxes = copy.copy(image_info['boxes'])

        # convert coordinates in range (0.0, 1.0) to pixels
        boxes[:, 0] *= image.shape[1] 
        boxes[:, 1] *= image.shape[0]
        boxes[:, 2] *= image.shape[1]
        boxes[:, 3] *= image.shape[0]

        # duplicate labels to prevent corruption of dataset
        labels = copy.copy(image_info['labels'])

        if self.viz_inputs: #visualize raw images
            image_before = copy.copy(image)
            sq_image_before = make_square(image_before, (0,0,0), 300)
        
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        # visualize overlaid bounding boxes
        if self.viz_inputs:
            image_after = np.moveaxis(image.numpy(), 0 ,-1)
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

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels) # creates LOTs of bounding boxes; see paper on SSD for info on 'prior bounding boxes'



        
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
        annotation_file = f"{self.root}/sub-{self.dataset_type}-annotations-bbox.csv"
        logging.info(f'loading annotations from: {annotation_file}')
        annotations = pd.read_csv(annotation_file)
        logging.info(f'annotations loaded from:  {annotation_file}')
        class_names = ['BACKGROUND'] + sorted(list(annotations['ClassName'].unique()))
        class_dict = {class_name: i for i, class_name in enumerate(class_names)}
        data = []
        for image_id, group in annotations.groupby("ImageID"):
            img_path = os.path.join(self.root, self.dataset_type, image_id + '.jpg')
            if os.path.isfile(img_path) is False:
                 logging.error(f'missing ImageID {image_id}.jpg - dropping from annotations')
                 continue
            boxes = group.loc[:, ["XMin", "YMin", "XMax", "YMax"]].values.astype(np.float32)
            # make labels 64 bits to satisfy the cross_entropy function
            labels = np.array([class_dict[name] for name in group["ClassName"]], dtype='int64')
            #print('found image {:s}  ({:d})'.format(img_path, len(data)))
            data.append({
                'image_id': image_id,
                'boxes': boxes,
                'labels': labels
            })
        print('num images:  {:d}'.format(len(data)))
        return data, class_names, class_dict

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.class_stat is None:
            self.class_stat = {name: 0 for name in self.class_names[1:]}
            for example in self.data:
                for class_index in example['labels']:
                    class_name = self.class_names[class_index]
                    self.class_stat[class_name] += 1
        content = ["Dataset Summary:"
                   f"Number of Images: {len(self.data)}",
                   f"Minimum Number of Images for a Class: {self.min_image_num}",
                   "Label Distribution:"]
        for class_name, num in self.class_stat.items():
            content.append(f"\t{class_name}: {num}")
        return "\n".join(content)

    def _read_image(self, image_id):
        image_file = self.root / self.dataset_type / f"{image_id}.jpg"
        image = cv2.imread(str(image_file))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _balance_data(self):
        logging.info('balancing data')
        label_image_indexes = [set() for _ in range(len(self.class_names))]
        for i, image in enumerate(self.data):
            for label_id in image['labels']:
                label_image_indexes[label_id].add(i)
        label_stat = [len(s) for s in label_image_indexes]
        self.min_image_num = min(label_stat[1:])
        sample_image_indexes = set()
        for image_indexes in label_image_indexes[1:]:
            image_indexes = np.array(list(image_indexes))
            sub = np.random.permutation(image_indexes)[:self.min_image_num]
            sample_image_indexes.update(sub)
        sample_data = [self.data[i] for i in sample_image_indexes]
        return sample_data





