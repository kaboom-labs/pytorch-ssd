from ..transforms.transforms import *

# NEW
import numpy as np
import albumentations as A
from copy import copy


        
class TrainAugmentation:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size of the final image
            mean: mean pixel value per channel (list of three)
            labels: int numpy 1d array that has category ids for each detection
        """
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image, boxes, labels):

        # if y_max == y_min, increase y_max by 2 pixels
        #problem_box = boxes[boxes[:,1]==boxes[:,3]]
        #fixed_box = copy(problem_box)
        #fixed_box[:,3] = fixed_box[:,1] + 2
        #boxes[boxes[:,1]==boxes[:,3]] = fixed_box

        boxes[:,0] = np.maximum(boxes[:,0], 0.0)
        boxes[:,1] = np.maximum(boxes[:,1], 0.0)
        boxes[:,2] = np.minimum(boxes[:,2], image.shape[1])
        boxes[:,3] = np.minimum(boxes[:,3], image.shape[0])

        transform = A.Compose([
            A.RandomRotate90(p=0.3),
            A.RandomSizedBBoxSafeCrop(height=self.size, width=self.size, p=0.8),
            A.Resize(height=self.size, width=self.size), # in case RandomSizedBBox SafeCrop doesn't crop, image still needs to be resized
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.1, hue=0.1, p=0.9),
            A.ToFloat(),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['class_categories'])) # min_visibility=0 is important because SSD needs at least one bounding box
        
        try:
            transformed = transform(image=image, bboxes=boxes, class_categories=labels)
            trans_image = transformed['image']
            trans_boxes = np.array(transformed['bboxes'], dtype=np.float32)
            trans_labels = np.array(transformed['class_categories'], dtype=np.long)
        except:
            import IPython; IPython.embed()

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)
        
        # convert to torch tensor
        out_image = torch.tensor(ch_trans_image, dtype=torch.float32)
        out_labels = np.array(trans_labels, dtype='int64')

        out_boxes = copy(trans_boxes)

        return out_image, out_boxes, out_labels


class TrainAugmentation_old:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, img, boxes, labels):
        """

        Args:
            img: the output of cv.imread in RGB layout.
            boxes: bounding boxes in the form of (x1, y1, x2, y2) in pixels.
            labels: 1-d array of labels of bounding boxes.
        """
        augmented = self.augment(img,boxes,labels)

        return self.augment(img, boxes, labels)


class TestTransform_oldest:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            ToPercentCoords(),
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor(),
        ])

    def __call__(self, image, boxes, labels):
        return self.transform(image, boxes, labels)

class TestTransform:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size of the final image
            mean: mean pixel value per channel (list of three)
            labels: int numpy 1d array that has category ids for each detection
        """
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image, boxes, labels):

        # if y_max == y_min, increase y_max by 2 pixels
        # prevents area zero boxes
        problem_box = boxes[boxes[:,1]==boxes[:,3]]
        fixed_box = copy(problem_box)
        fixed_box[:,3] = fixed_box[:,1] + 2
        boxes[boxes[:,1]==boxes[:,3]] = fixed_box

        # ensure that boxes don't extend beyond image
        boxes[:,0] = np.maximum(boxes[:,0], 0.0)
        boxes[:,1] = np.maximum(boxes[:,1], 0.0)
        boxes[:,2] = np.minimum(boxes[:,2], image.shape[1])
        boxes[:,3] = np.minimum(boxes[:,3], image.shape[0])

        transform = A.Compose([
            A.Resize(height=self.size, width=self.size), # in case RandomSizedBBox SafeCrop doesn't crop, image still needs to be resized
            A.ToFloat(),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['class_categories'])) # min_visibility=0 is important because SSD needs at least one bounding box
        
        transformed = transform(image=image, bboxes=boxes, class_categories=labels)
        trans_image = transformed['image']
        trans_boxes = np.array(transformed['bboxes'], dtype=np.float32)
        trans_labels = np.array(transformed['class_categories'], dtype=np.long)

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)
        
        # convert to torch tensor
        out_image = torch.tensor(ch_trans_image, dtype=torch.float32)
        out_labels = np.array(trans_labels, dtype='int64')

        out_boxes = copy(trans_boxes)

        return out_image, out_boxes, out_labels



        
class TestTransform_old:
    def __init__(self, size, mean=0, std=1.0):
        """
        Args:
            size: the size of the final image
            mean: mean pixel value per channel (list of three)
            labels: int numpy 1d array that has category ids for each detection
        """
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image, boxes, labels):

        # if y_max == y_min, increase y_max by 2 pixels
        problem_box = boxes[boxes[:,1]==boxes[:,3]]
        fixed_box = copy(problem_box)
        fixed_box[:,3] = fixed_box[:,1] + 2
        boxes[boxes[:,1]==boxes[:,3]] = fixed_box

        # @TODO clip any bounding boxes outside of image area

        transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.ToFloat(),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_categories']))
        
        transformed = transform(image=image, bboxes=boxes, class_categories=labels)
        trans_image = transformed['image']
        trans_boxes = np.array(transformed['bboxes'], dtype=np.float32)
        trans_labels = np.array(transformed['class_categories'], dtype=np.long)

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)
        
        # convert to torch tensor
        out_image = torch.tensor(ch_trans_image, dtype=torch.float32)

        out_labels = np.array(trans_labels, dtype='int64')

        # pixels->scale[0:1[ for boxes
        unscaled_boxes = copy(trans_boxes)
        width = image.shape[1]
        height = image.shape[0]
        unscaled_boxes[:,0]/=width
        unscaled_boxes[:,2]/=width
        unscaled_boxes[:,1]/=height
        unscaled_boxes[:,3]/=height
        out_boxes = copy(unscaled_boxes)

        return out_image, out_boxes, out_labels

class TestTransform_old:
    def __init__(self, size, mean=0.0, std=1.0):
        """
        Args:
            size: the size of the final image
            mean: mean pixel value per channel (list of three)
            labels: int numpy 1d array that has category ids for each detection
        """
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image, boxes, labels):
        transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.ToFloat(),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_categories']))
        transformed = transform(image=image, bboxes=boxes, class_categories=labels)
        trans_image = transformed['image']
        trans_boxes = np.array(transformed['bboxes'], dtype=np.float32)
        trans_labels = np.array(transformed['class_categories'], dtype=np.long)

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)

        return ch_trans_image, trans_boxes, trans_labels


class PredictionTransform_old:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image
    
class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        """
        Args:
            size: the size of the final image
            mean: mean pixel value per channel (list of three)
            labels: int numpy 1d array that has category ids for each detection
        """
        self.mean = mean
        self.std = std
        self.size = size

    def __call__(self, image):
        transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.ToFloat(),
            ])
        transformed = transform(image=image)
        trans_image = transformed['image']

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)

        return torch.Tensor(ch_trans_image)
