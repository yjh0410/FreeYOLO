from __future__ import division

import os
import argparse
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from engine import train_with_warmup, train_one_epoch, val_one_epoch

from config import build_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')

    # model
    parser.add_argument('-v', '--version', default='yolox_d53', type=str,
                        help='build yolox')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='Mosaic augmentation')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='MixUp augmentation')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    cfg = build_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(cfg, args, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, CollateFunc())

    # build model
    net = build_model(args=args, 
                      cfg=cfg,
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True,
                      coco_pretrained=args.coco_pretrained)
    model = net
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, 
                         min_size=cfg['test_min_size'], 
                         max_size=cfg['test_max_size'], 
                         device=device)
        model_copy.trainable = True
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # optimizer
    base_lr = 0.01 * cfg['batch_size'] / 64
    optimizer = build_optimizer(model=model_without_ddp,
                                base_lr=base_lr,
                                backbone_lr=base_lr,
                                name=cfg['optimizer'],
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                                              T_max=None, 
                                                              eta_min=None, 
                                                              last_epoch=None,)

    # warmup scheduler
    wp_iter = len(dataloader) * cfg['wp_epoch']
    warmup_scheduler = build_warmup(name=cfg['warmup'],
                                    base_lr=base_lr,
                                    wp_iter=wp_iter,
                                    warmup_factor=cfg['warmup_factor'])

    # warmup training loop
    train_with_warmup(args=args, 
                      device=device, 
                      model=model, 
                      cfg=cfg, 
                      base_lr=base_lr, 
                      dataloader=dataloader, 
                      optimizer=optimizer, 
                      warmup_scheduler=warmup_scheduler)

    # start training loop
    for epoch in range(cfg['max_epoch']):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        train_one_epoch(args=args, 
                        device=device, 
                        model=model, 
                        cfg=cfg, 
                        dataloader=dataloader, 
                        optimizer=optimizer, 
                        lr_scheduler=lr_scheduler)

        # evaluation
        val_one_epoch(args=args, 
                      model=model_without_ddp, 
                      evaluator=evaluator,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      epoch=epoch,
                      path_to_save=path_to_save)

        # close mosaic augmentation
        if args.mosaic and cfg['max_epoch'] - epoch == 5:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False


if __name__ == '__main__':
    train()
