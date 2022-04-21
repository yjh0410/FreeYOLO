from __future__ import division

import os
import math
import argparse
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import ModelEMA, CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from engine import train_with_warmup, train_one_epoch, val_one_epoch

from config import build_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOX')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')

    # model
    parser.add_argument('-v', '--version', default='yolox_d53', type=str,
                        help='build yolox')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Model EMA')

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
    print('World size: {}'.format(distributed_utils.get_world_size()))
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
    batch_size = cfg['batch_size'] * distributed_utils.get_world_size()
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc())

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
                         img_size=cfg['img_size'], 
                         device=device)
        model_copy.trainable = True
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # optimizer
    base_lr = cfg['base_lr'] * batch_size
    min_lr = base_lr * cfg['min_lr_ratio']
    optimizer = build_optimizer(model=model_without_ddp,
                                base_lr=base_lr,
                                name=cfg['optimizer'],
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    
    # warmup scheduler
    wp_iter = len(dataloader) * cfg['wp_epoch']
    warmup_scheduler = build_warmup(name=cfg['warmup'],
                                    base_lr=base_lr,
                                    wp_iter=wp_iter,
                                    warmup_factor=cfg['warmup_factor'])


    # start training loop
    best_map = -1.0
    for epoch in range(cfg['wp_epoch'] + cfg['max_epoch']):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        if epoch < cfg['wp_epoch']:
            # warmup training loop
            train_with_warmup(epoch=epoch,
                              args=args, 
                              device=device, 
                              ema=ema,
                              model=model, 
                              cfg=cfg, 
                              base_lr=base_lr, 
                              dataloader=dataloader, 
                              optimizer=optimizer, 
                              warmup_scheduler=warmup_scheduler)

            # evaluation
            best_map = val_one_epoch(args=args, 
                          model=ema.ema if args.ema else model_without_ddp, 
                          evaluator=evaluator,
                          optimizer=optimizer,
                          epoch=epoch,
                          best_map=best_map,
                          path_to_save=path_to_save)

        else:
            # use cos lr decay
            T_max = T_max=cfg['max_epoch'] - cfg['no_aug_epoch']
            if epoch > T_max:
                # Cos decay is done
                print('Cosine annealing is over !!')
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr
            else:
                tmp_lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*epoch / T_max))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = tmp_lr

            # train one epoch
            train_one_epoch(epoch=epoch,
                            args=args, 
                            device=device,
                            ema=ema, 
                            model=model, 
                            cfg=cfg, 
                            dataloader=dataloader, 
                            optimizer=optimizer)
        
            # evaluation
            best_map = val_one_epoch(args=args, 
                          model=ema.ema if args.ema else model_without_ddp, 
                          evaluator=evaluator,
                          optimizer=optimizer,
                          epoch=epoch,
                          best_map=best_map,
                          path_to_save=path_to_save)

        # close mosaic augmentation
        if args.mosaic and cfg['max_epoch'] - epoch == cfg['no_aug_epoch']:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False
        # close mixup augmentation
        if args.mixup and cfg['max_epoch'] - epoch == cfg['no_aug_epoch']:
            print('close Mixup Augmentation ...')
            dataloader.dataset.mixup = False


if __name__ == '__main__':
    train()
