import time
import torch
from collections import OrderedDict


def str2bool(s):
    return s.lower() in ('true', '1')

class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval
        

def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))

def xyxy_to_xywh(xyxy):
    '''
    convert bounding boxes from
    x_min,y_min,x_max,y_max (pixels) [open_images / SSD]
    to
    x_left_top, y_left_top, width, height (pixels) [ COCO ]
    '''
    return [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]


def xywh_to_xyxy(xywh: list) -> list:
    '''
    DO NOT CONFUSE WITH `yolo_to_xyxy`, where the first two are x_center, y_center instead of x_min, y_min
    Convert bounding boxes from
    x_left_top, y_left_top, width, height (pixels)
    to
    x_min, y_min, x_max, y_max
    '''
    return [xywh[0], xywh[1], xywh[0]+xywh[2], xywh[1]+xywh[3]]

def yolo_to_xyxy(yolo: list) -> list:
    '''
    x_center, y_center, width, height
    to
    x_min, y_min, x_max, y_max
    '''
    x_center = yolo[0]
    y_center = yolo[1]
    width = yolo[2]
    height = yolo[3]
    
    x_min = x_center - ( width / 2 )
    y_min = y_center - ( height / 2 )
    x_max = x_center + ( width / 2 )
    y_max = y_center + ( width / 2 )

    return [x_min, y_min, x_max, y_max]

def yolo_to_xyxy(yolo: list) -> list:
    '''
    x_center, y_center, width, height
    to
    x_min, y_min, x_max, y_max
    '''
    x_center = yolo[0]
    y_center = yolo[1]
    width = yolo[2]
    height = yolo[3]

    x_min = x_center - ( width / 2 )
    y_min = y_center - ( height / 2 )
    x_max = x_center + ( width / 2 )
    y_max = y_center + ( height / 2 )

    return [x_min, y_min, x_max, y_max]

def xyxy_norm_to_abs(xyxy: list, height:int, width:int) -> list:
    '''
    Convert normalized xyxy (0~1)
    to absolute pixel xyxy (0~height or width of image)
    '''
    return [xyxy[0]*width, xyxy[1]*height, xyxy[2]*width, xyxy[3]*height]


def optimizer_to(optim,device):
    '''
    move optimizer to GPU. PyTorch doesn't have a '.to(device)' for optimizers as it does for nn.Modules
    code from: https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    '''
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
    return optim

def cuda_multi_to_single(net_state_dict):
    '''
    Remove 'module.' from model state dict if training on Single GPU.
    When model is trained on multi GPU, PyTorch adds the 'module.' name
    fixes https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
    '''
    if torch.cuda.device_count() <= 1:
        new_net_state_dict = OrderedDict()
        for k,v in net_state_dict.items():
            if k[:7] == 'module.':
                name = k[7:] # remove 'module.' which was added by nn.DataParallel
            else:
                name = k
            new_net_state_dict[name] = v
    return new_net_state_dict
