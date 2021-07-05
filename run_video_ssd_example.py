from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer
import cv2
import sys
import torch

import matplotlib.pyplot as plt
import matplotlib.animation as animation


DEVICE = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

output_path = None

if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path> <optional: video output path>')
    sys.exit(0)
if len(sys.argv) < 5:
    print('Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path> <optional: video output path>')
    sys.exit(1)
elif len(sys.argv) < 6:
    net_type = sys.argv[1]
    model_path = sys.argv[2]
    label_path = sys.argv[3]
    image_path = sys.argv[4]
elif len(sys.argv) == 6:
    output_path = sys.argv[5]
else:
    print("too many arguments")
    sys.exit(1)

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == 'vgg16-ssd':
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd':
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == 'mb1-ssd-lite':
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == 'mb2-ssd-lite':
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == 'sq-ssd-lite':
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
    sys.exit(1)

combo_checkpoint = torch.load(model_path)
net_state_dict = combo_checkpoint['weights']
net.load_state_dict(net_state_dict)
net = net.to(DEVICE)

if net_type == 'vgg16-ssd':
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd':
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == 'mb1-ssd-lite':
    predictor = create_mobilenetv1_ssd_lite_nredictor(net, candidate_size=200)
elif net_type == 'mb2-ssd-lite':
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == 'sq-ssd-lite':
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

cap = cv2.VideoCapture(image_path)

# set video parameters
frame_width = int(cap.get(3))
frame_height =int(cap.get(3))

# define codec for video output
out = cv2.VideoWriter('video_ssd_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        print("Unable to read video source (camera) feed")
        exit(1)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(frame, 10, 0.4)

    for i in range(boxes.size(0)):
        box = boxes[i, :]

        box = box.numpy()
        
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 0), 4)
        # OpenCV needs python int, not numpy int. Manual casting fixes https://github.com/opencv/opencv/issues/15465
        
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.putText(frame, label,
                    (int(box[0]) + 20, int(box[1]) + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type

    # write video file if output path arg was passed in
    if output_path:
        out.write(frame)
    else:
        # display live frame
        cv2.imshow(f'frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
out.release()

cv2.destroyAllWindows()

