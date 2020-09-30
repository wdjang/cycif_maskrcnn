'''
<Env>

module load Anaconda3/5.0.1-fasrc01
module load python/3.6.3-fasrc01
module load cuda/10.1.243-fasrc01
source activate cellsegm

<Test>

python -m torch.distributed.launch --master_port=9588 --nproc_per_node=1 --use_env cellsegm_test.py --world-size 1 --test-only --use-channel dapi --vis-dir vis-dapi --resume ./trained_models_dapi/model_700.pth --root-path /n/pfister_lab2/Lab/vcg_biology/cycif/EMIT/TMA22/dearray/

python -m torch.distributed.launch --master_port=9588 --nproc_per_node=1 --use_env cellsegm_test.py --world-size 1 --test-only --use-channel dapi --vis-dir vis-dapi --resume ./trained_models_dapi/model_700.pth --root-path /n/pfister_lab2/Lab/vcg_biology/cycif/EMIT/TMA11/dearray/
'''
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
# import torchvision.models.detection
# import torchvision.models.detection.mask_rcnn
from mask_rcnn import maskrcnn_resnet50_fpn

# from coco_utils import get_cellsegm
from coco_utils import get_celltest

from engine import train_one_epoch, evaluate, test

import utils
import transforms as T

import pdb


def get_dataset(data_name, process_phase, root_path, use_channel):
    path_list = {
        "cellsegm": (root_path, get_celltest, 2),
    }
    root_path, dataset_func, num_classes = path_list[data_name]

    ret_dataset = dataset_func(root_path, use_channel=use_channel)
    return ret_dataset, num_classes

def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # Data loading code
    print("Loading data")
    dataset_test, num_classes = get_dataset(args.dataset,
                                            'test',
                                            args.root_path,
                                            args.use_channel)
    print("Creating data loaders")
    if args.distributed:
        sampler_test = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=sampler_test, num_workers=args.workers)
    print("Creating model")
    # maskrcnn_resnet50_fpn
    model = maskrcnn_resnet50_fpn(num_classes=num_classes,
                                  pretrained=args.pretrained)
    # set iou between boxes for nms: 0.7
    model.roi_heads.nms_thresh = 0.3
    # set the max num of rois: 1000
    model.roi_heads.detections_per_img = 1000
    # default: 0.05, 0.5
    model.roi_heads.score_thresh = 0.05
    model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    checkpoint = torch.load(args.resume, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])

    test(model, dataloader_test, device,
         is_vis=args.vis, draw_bbox=False, vis_dir=args.vis_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument(
        '--root-path', default='/n/pfister_lab2/Lab/vcg_biology/cycif/EMIT/TMA22/dearray/', help='dataset')
    parser.add_argument('--dataset', default='cellsegm', help='dataset')
    parser.add_argument(
        '--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=260, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02, type=float,
                        help='initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[160, 220], nargs='+', type=int,
                        help='decrease lr at epochs def: [16, 22]')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20,
                        type=int, help='print frequency')
    parser.add_argument('--output-dir', default='trained_models',
                        help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0,
                        type=int, help='start epoch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--vis",
        help="Save visualization result",
        action="store_true",
    )
    parser.add_argument(
        "--vis-dir",
        default="./vis_results",
        help="Location to save visualization result",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use-channel', default='dapi',
                        help='which channel to use; one of dapi/lamin/both')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir + '_' + args.use_channel)

    main(args)
