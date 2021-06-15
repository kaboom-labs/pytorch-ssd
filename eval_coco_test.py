'''
Run fix_coco_annotations.py before running this script

Generate files for upload to COCO test server.
'''

import torch
from torch.utils.data import DataLoader
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.coco_test import COCOTest 
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool, Timer, xyxy_to_xywh
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor

import os
import concurrent.futures
from tqdm import tqdm, trange
import cv2
from icecream import ic
import json

parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str,
                    help='Path to combo checkpoint .pth')
parser.add_argument("--dataset", type=str, help="The root directory of the COCO dataset.")
parser.add_argument("--label-file", type=str, help="The label file path.")
parser.add_argument("--num-workers", type=int, help="Parallel data fetching workers")
parser.add_argument("--use-cuda", type=str2bool, default=True)
parser.add_argument("--write-images", default='', type=str, help="The directory to store bounding box overlayed images")
parser.add_argument("--eval-dir", default="eval_results", type=str, help="The directory to store evaluation results.")
parser.add_argument('--mb2-width-mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
#DEVICE = torch.device("cpu")

timer = Timer()

# check output path
eval_path = pathlib.Path(args.eval_dir)
eval_path.mkdir(exist_ok=True)
write_images = pathlib.Path(args.write_images)
write_images.mkdir(exist_ok=True)

# instead of looking up COCO id and file names, 
# just cheaply convert id to filename
def cheap_id_to_filename(image_id):
    image_id = str(image_id)
    filename = str(0)*(12-len(image_id))+ image_id +'.jpg'
    return filename

# draw predicted bounding boxes on test images and save them
def write_bbox_images(results, path):
    for result in tqdm(results, desc="Writing images"):
        image_id = result[0]
        image_path = os.path.join(os.path.abspath(args.dataset),'test2017', cheap_id_to_filename(image_id))
        img = cv2.imread(image_path)
        if len(result[1]) > 0: #something is detected
            boxes = result[1]
            labels = result[2]
            probs = result[3]

            for i in range(len(boxes)):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,255,0), 4)
                cv2.putText(img, label,
                        (int(box[0]) + 20, int(box[1]) + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, #font scale
                        (255,0,255), #color
                        2) #line type
        out_path = os.path.join(path, cheap_id_to_filename(image_id))
        cv2.imwrite(out_path, img)
        tqdm.write(f"Drawing {len(probs)} objects to {out_path}")


# run image through SSD
def infer(image_id):
    image_filename = cheap_id_to_filename(image_id)
    image_path = os.path.join(os.path.abspath(args.dataset), 'test2017', image_filename)
    if not os.path.exists(image_path):
        print("Doesn't exist")
        exit(1)
    orig_image = cv2.imread(image_path)
    if orig_image.shape[2] == 1:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_GRAY2RGB)
    else:
        orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    boxes, labels, probs = predictor.predict(orig_image, 10, 0.4)

    return (image_id, boxes, labels, probs)

class_names = [name.strip() for name in open(args.label_file).readlines()]

if args.net == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif args.net == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif args.net == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif args.net == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
elif args.net == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), width_mult=args.mb2_width_mult, is_test=True)
else:
    logging.fatal("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    parser.print_help(sys.stderr)
    sys.exit(1)  

net_type = args.net

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_nredictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200, device=DEVICE)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200, device=DEVICE)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200, device=DEVICE)

dataset = COCOTest(args.dataset)

# load model
combo_checkpoint = torch.load(args.trained_model)
net_state_dict = combo_checkpoint['weights']
timer.start("Load Model")
net.load_state_dict(net_state_dict)
net = net.to(DEVICE)
print(f'It took {timer.end("Load Model")} seconds to load the model.')


# multithreaded inference 
print("Starting predictions")
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
    results = list(tqdm(executor.map(infer, dataset.image_ids), total=len(dataset.image_ids), desc='Running inference'))
#result = []
#for image_path in image_paths:
#    result += infer(image_path)

# format results from inference to COCO results format
# https://cocodataset.org/#format-results
coco_results = []
for result in results:
    num_detections = result[2].shape[0]
    for i in range(num_detections):
        image_id = int(result[0])
        category_id = int(result[2][i])
        x1,y1,x2,y2 = list(result[1][i])
        xyxy_bbox = [round(float(x1),2), round(float(y1),2), round(float(x2),2), round(float(y2),2)]
        xywh_bbox = xyxy_to_xywh(xyxy_bbox)
        score = round(float(result[3][i]),3)

        coco_results.append({
            "image_id":image_id,
            "category_id":category_id,
            "bbox":xywh_bbox,
            "score":score})

with open(os.path.join(args.eval_dir, 'detections_test-dev2017_mobilenetv2-ssd-lite_results.json'), 'w') as j:
    json.dump(coco_results, j)

# save images with predicted bounding boxes in args.
if args.write_images:
    args.write_images = os.path.abspath(args.write_images)
    write_bbox_images(results, write_images)
    
