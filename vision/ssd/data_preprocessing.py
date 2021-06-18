from ..transforms.transforms import *

# NEW
import numpy as np
import albumentations as A


        
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
        # make sure bbox boundaries don't exceed image
        #import IPython; IPython.embed()
        #boxes[:,0][boxes<0] = 0 # x min
        #boxes[:,1][boxes<0] = 0 # y min
        #boxes[:,2][boxes>image.shape(1)] = image.shape(1) # x max
        #boxes[:,3][boxes>image.shape(0)] = image.shape(0) # y max

        #IPython.embed()


        main_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.7, contrast=0.2, saturation=0.1, hue=0.1, p=0.8),
            A.RandomSizedBBoxSafeCrop(height=self.size, width=self.size),
            A.ToFloat(),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['class_categories'])) # min_visibility=0 is important because SSD needs at least one bounding box
        
        minimal_transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.ToFloat(),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.0, label_fields=['class_categories']))

        #sometimes gets ValueError ("y_max is less than or equal to y_min...
        try:
            transformed = main_transform(image=image, bboxes=boxes, class_categories=labels)
            trans_image = transformed['image']
            trans_boxes = np.array(transformed['bboxes'], dtype=np.float32)
            trans_labels = np.array(transformed['class_categories'], dtype=np.long)

        except:
            import IPython; IPython.embed();exit(1)
            print("\nProblem with main transform. Falling back to minimal transform. From vision/ssd/data_preprocessing.py")
            transformed = minimal_transform(image=image, bboxes=boxes, class_categories=labels)
            trans_image = transformed['image']
            trans_boxes = np.array(transformed['bboxes'], dtype=np.float32)
            trans_labels = np.array(transformed['class_categories'], dtype=np.long)
            

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)
        
        # convert to torch tensor
        out_image = torch.tensor(ch_trans_image, dtype=torch.float32)
        
        if torch.isnan(out_image).any():
            print("NAN!")
            import IPython; IPython.embed()
        out_boxes = trans_boxes
        out_labels = np.array(trans_labels, dtype='int64')

        return out_image, out_boxes, out_labels


class TrainAugmentation_OLD:
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

        #augmented = self.augment(img,boxes,labels)

        return self.augment(img, boxes, labels)


class TestTransform_old:
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

        # add miniscule amount to bbox in case bbox area is 0; avoids error
        trans_boxes[:, 2] = trans_boxes[:, 2] + 1e-10
        trans_boxes[:, 3] = trans_boxes[:, 3] + 1e-10

        return ch_trans_image, trans_boxes, trans_labels


class PredictionTransform:
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

    def __call__(self, image, boxes, labels):
        transform = A.Compose([
            A.Resize(height=self.size, width=self.size),
            A.ToFloat(),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_categories']))
        transformed = transform(image=image, bboxes=boxes, class_categories=labels)
        trans_image = transformed['image']

        # cast image from (width, height, channel) to (channel, width, height)
        ch_trans_image = np.array(np.moveaxis(trans_image, -1, 0), dtype=np.float32)

        return ch_trans_images
