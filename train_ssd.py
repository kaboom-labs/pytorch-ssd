#
# train an SSD model on Pascal VOC, Open Images, or COCO datasets
#
import os
import sys
import logging
import argparse
import itertools
import torch
from collections import OrderedDict
import json

import time
import math
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels, optimizer_to, cuda_multi_to_single
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.datasets.coco_dataset import COCODataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

#DEBUG
from icecream import ic
from tqdm import tqdm, trange

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With PyTorch')

# Params for datasets
parser.add_argument("--dataset-type", default="coco", type=str,
                    help='Specify dataset type. Currently supports voc,open_images, and coco.')
parser.add_argument('--datasets', '--data', nargs='+', default=["coco-data"], help='Dataset directory path')
parser.add_argument('--balance-data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")

# Params for network
parser.add_argument('--net', default="mb2-ssd-lite",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze-base-net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze-net', action='store_true',
                    help="Freeze all the layers except the prediction head.")
parser.add_argument('--mb2-width-mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base-net', help='Pretrained base model')
parser.add_argument('--pretrained-ssd', default='', type=str, help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict files and other data to resume training from. Uses torch.load to load both net and optimizer state_dicts')

# Choose optimizer & parameters shared between SGD and Adam 
parser.add_argument('--optim-choose', default='Adam', type=str,
                    help='Choose optimizer from Adam, SGD')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    help='initial learning rate. Adam default=1e-3, SGD default=1e-3')
parser.add_argument('--weight-decay', default=0, type=float,
                    help='Weight decay. Adam default=0, SGD default=5e-4')

# Exclusive params for Adam
parser.add_argument('--betas', default=(0.9, 0.999), type=tuple,
                    help='Coefficients used for computing running averages of gradient and its square')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='term added to the denominator to improve numerical stability')
parser.add_argument('--amsgrad', action='store_true',
                    help='Whether to use AMSGrad variant of Adam')


# Exclusive params for SGD
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--dampening', default=0, type=float,
                    help='Dampening for momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='enables Nesterov momentum.')

# Params for learning rate scheduler
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update. Decays learning rate by this factor')
parser.add_argument('--base-net-lr', default=None, type=float,
                    help='initial learning rate for base net, or None to use --lr')
parser.add_argument('--extra-layers-lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')

# Scheduler
parser.add_argument('--scheduler', default="none", type=str,
                    help="Learning Rate Scheduler. It can be none, multi-step, cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t-max', default=100, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch-size', default=2, type=int,
                    help='Batch size for training')
parser.add_argument('--num-epochs', '--epochs', default=30, type=int,
                    help='the number epochs')
parser.add_argument('--num-workers', '--workers', default=6, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation-epochs', default=1, type=int,
                    help='the number epochs between running validation')
parser.add_argument('--debug-steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use-cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--checkpoint-folder', '--model-dir', default='models/',
                    help='Directory for saving checkpoint models')

# Tensorboard integration
parser.add_argument('--tensorboard', action='store_true',
                    help='Visualize training in tensoboard')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
                    
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA...")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    sub_epoch = math.ceil(len(loader)/debug_steps)*epoch # subdivide epoch into sub-epochs for reporting to tensorboard
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in tqdm(enumerate(loader), desc=f"Tensorboard @ http://localhost:6006/ | Batch", total=len(loader), leave=True):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        tqdm.write(
                f"Epoch: {epoch}/{args.num_epochs}\t Loss: {str(loss.item())[:5]}\t Regression Loss: {str(regression_loss.item())[:5]}\t Classification Loss: {str(classification_loss.item())[:5]}")

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps

            prev_loss = avg_loss
            prev_reg_loss = avg_reg_loss
            prev_clf_loss = avg_clf_loss

            #logging.info(
            #    f"Epoch: {epoch}/{args.num_epochs}, Step: {i}/{len(loader)}, " +
            #    f"Avg Loss: {avg_loss:.4f}, " +
            #    f"Avg Regression Loss {avg_reg_loss:.4f}, " +
            #    f"Avg Classification Loss: {avg_clf_loss:.4f}" +
            #)
            #tqdm.write(f"Epoch: {epoch}/{args.num_epochs}\t Avg Loss: {avg_loss:.3f}\t Avg Regression Loss: {avg_reg_loss:.3f}\t Avg Classification Loss: {avg_clf_loss:.3f}")

            writer.add_scalar("Average Loss", avg_loss, sub_epoch)
            writer.add_scalar("Average Regression Loss", avg_reg_loss, sub_epoch)
            writer.add_scalar("Average Classification Loss", avg_clf_loss, sub_epoch)
            sub_epoch += 1
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

if __name__ == '__main__':
    timer = Timer()

    # start tensorboard
    if args.tensorboard:
        os.system('tensorboard --logdir runs &')
            
    # if requested, load arguments from JSON file in parent folder of checkpoint 
    if args.resume:
        training_args_path = os.path.join(os.path.split(os.path.abspath(args.resume))[0], 'training_args.json')

        with open(training_args_path, 'r') as j:
            training_args = json.load(j)

        args.dataset_type = training_args['dataset_type']
        args.datasets = training_args['datasets']
        args.net= training_args['net']
        args.freeze_base_net = training_args['freeze_base_net']
        args.freeze_net = training_args['freeze_net']
        args.mb2_width_mult = training_args['mb2_width_mult']
        args.optim_choose = training_args['optim_choose']
        args.batch_size = training_args['batch_size']
        args.num_epochs = training_args['num_epochs']
        args.num_workers = training_args['num_workers']
        args.debug_steps = training_args['debug_steps']
        args.use_cuda = training_args['use_cuda']
        args.checkpoint_folder = training_args['checkpoint_folder']
        args.tensorboard = training_args['tensorboard']

    logging.info(args)

    # make sure that the checkpoint output dir exists
    if args.checkpoint_folder:
        args.checkpoint_folder = os.path.expanduser(args.checkpoint_folder)
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)

    # select the network architecture and config     
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    # create data transforms for train/test/val
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    # create training dataset
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        dataset_path = os.path.abspath(dataset_path)
        if args.dataset_type.lower() == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type.lower() == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.checkpoint_folder, "labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)
        elif args.dataset_type.lower() == 'coco':
            dataset = COCODataset(dataset_path,
                    transform=train_transform,
                    target_transform=target_transform,
                    dataset_type="train")
            label_file = os.path.join(args.checkpoint_folder, "labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
        
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
                           
    # create validation dataset                           
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(dataset_path, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
    elif args.dataset_type == 'coco':
        val_dataset = COCODataset(dataset_path,
                                        transform=test_transform, 
                                        target_transform=target_transform,
                                        dataset_type="val")
    logging.info(val_dataset)
    logging.info("Validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

                            
    # create the network
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    # freeze certain layers (if requested)
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    
    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    # define loss function and optimizer
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    # select optimizer and config
    if args.optim_choose.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    if args.optim_choose.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, dampening=args.dampening, nesterov=args.nesterov)

    # set learning rate policy
    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    elif args.scheduler == 'none':
        logging.info("No learning rate scheduler specified; if using SGD, a LR scheduler is recommended.")
        scheduler = None
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # load a previous model checkpoint (if requested)
    timer.start("Load Model")
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")

        combo_checkpoint = torch.load(args.resume)
        last_epoch = combo_checkpoint['epoch']
        net_state_dict = combo_checkpoint['weights']
        optimizer_state_dict = combo_checkpoint['optimizer']
        try:
            scheduler = combo_checkpoint['scheduler']
        except KeyError:
            scheduler = None

        net_state_dict = cuda_multi_to_single(net_state_dict)
        # load state dicts into model and optimizer
        net.load_state_dict(net_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        net = net.to(DEVICE)
        optimizer = optimizer_to(optimizer, DEVICE)
        if args.use_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()

    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    # move the model to GPU
    net.to(DEVICE)

    # DataParallel: automatically run on multiple GPUs
    logging.info(f"Using {torch.cuda.device_count()} GPUs!")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    # save training parameters to model directory
    # enables resuming training without having to copy all the arguments
    training_args = {} # save all training args that are required to continue training on same dataset; can be overriden at resume if need be.
    # 'initial values' for optimizers, etc are not included because they are saved as part of state_dict
    training_args.update({'dataset_type': args.dataset_type})
    training_args.update({'datasets': args.datasets})
    training_args.update({'balance-data': args.balance_data})
    training_args.update({'net': args.net})
    training_args.update({'freeze_base_net': args.freeze_base_net})
    training_args.update({'freeze_net': args.freeze_net})
    training_args.update({'mb2_width_mult': args.mb2_width_mult})
    training_args.update({'optim_choose': args.optim_choose})
    training_args.update({'batch_size': args.batch_size})
    training_args.update({'num_epochs': args.num_epochs})
    training_args.update({'num_workers': args.num_workers})
    training_args.update({'debug_steps': args.debug_steps})
    training_args.update({'use_cuda': args.use_cuda})
    training_args.update({'checkpoint_folder': args.checkpoint_folder})
    training_args.update({'tensorboard' : args.tensorboard})

    training_args_path = os.path.join(os.path.abspath(args.checkpoint_folder), 'training_args.json')

    with open(training_args_path, 'w') as j:
        json.dump(training_args, j)

    # train for the desired number of epochs
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    
    for epoch in trange(last_epoch + 1, args.num_epochs, initial=last_epoch+1, desc=f"Training {args.net} on {args.datasets[0][:9]}   | Epoch", total=args.num_epochs, leave=True):
        timer.start("Epoch")
        if scheduler is not None:
            scheduler.step()
        train(train_loader, net, criterion, optimizer, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            model_path = os.path.join(args.checkpoint_folder, f"COMBO_CHECKPOINT_{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")

            # save more comprehensive checkpoint that includes optimizer and scheduler.
            # to load just net weights (for eval), `torch.load(combo_checkpoint)['weights']`
            combo_checkpoint = {
                    'epoch': epoch,
                    'weights': net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    }
            if scheduler is not None: # possibility that no scheduler designated at all
                combo_checkpoint.update({'scheduler':scheduler})
            torch.save(combo_checkpoint, model_path)
            logging.info(f"Saved combo checkpoint to {model_path}")
            logging.info(f"Time elapsed in epoch {epoch} : {timer.end('Epoch'):.2f}")

    writer.flush()
    writer.close()

    logging.info("Task done, exiting program.")
