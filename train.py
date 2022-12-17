from __future__ import division

import os
import math
import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import ModelEMA, CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from engine import train_with_warmup, train_one_epoch, val_one_epoch

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLO')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_first', action='store_true', default=False,
                        help='evaluate model before training.')
    parser.add_argument('--fp16', dest="fp16", action="store_true", default=False,
                        help="Adopting mix precision training.")
    
    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=16, type=int, 
                        help='batch size on a single GPU.')
    parser.add_argument('-accu', '--accumulate', default=1, type=int, 
                        help='gradient accumulate.')
    parser.add_argument('-lr', '--base_lr', default=0.01, type=float, 
                        help='base lr.')
    parser.add_argument('-mlr', '--min_lr_ratio', default=0.05, type=float, 
                        help='base lr.')

    # Epoch
    parser.add_argument('--max_epoch', default=300, type=int, 
                        help='max epoch.')
    parser.add_argument('--wp_epoch', default=1, type=int, 
                        help='warmup epoch.')
    parser.add_argument('--eval_epoch', default=10, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    
    # model
    parser.add_argument('-v', '--version', default='yolo_free_large', type=str,
                        help='build yolo')
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--pretrained', default=None, type=str,
                        help='load pretrained weight')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--ema', action='store_true', default=False,
                        help='Model EMA')
    parser.add_argument('--min_box_size', default=8.0, type=float,
                        help='min size of target bounding box.')
    parser.add_argument('--mosaic', default=None, type=float,
                        help='mosaic augmentation.')
    parser.add_argument('--mixup', default=None, type=float,
                        help='mixup augmentation.')

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
        # cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # amp
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # config
    cfg = build_config(args)

    # dataset and evaluator
    dataset, dataset_info, evaluator = build_dataset(cfg, args, device, is_train=True)
    num_classes = dataset_info[0]

    # dataloader
    dataloader = build_dataloader(args, dataset, args.batch_size, CollateFunc())

    # build model
    model, criterion = build_model(
        args=args, 
        cfg=cfg,
        device=device,
        num_classes=num_classes,
        trainable=True,
        )
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
                         img_size=cfg['test_size'], 
                         device=device)
        del model_copy
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # batch size
    world_size = distributed_utils.get_world_size()
    single_gpu_bs = args.batch_size
    accumulate = args.accumulate
    total_bs = single_gpu_bs * accumulate * world_size

    # learning rate
    base_lr = args.base_lr * (total_bs / 64)
    min_lr = base_lr * args.min_lr_ratio

    # optimizer
    optimizer, start_epoch = build_optimizer(cfg, model_without_ddp, base_lr, args.resume)
    
    # warmup scheduler
    wp_iter = len(dataloader) * args.wp_epoch
    warmup_scheduler = build_warmup(cfg=cfg, base_lr=base_lr, wp_iter=wp_iter)

    # EMA
    if args.ema and distributed_utils.get_rank() in [-1, 0]:
        print('Build ModelEMA ...')
        ema = ModelEMA(model, updates=start_epoch * len(dataloader))
    else:
        ema = None

    # start training loop
    best_map = -1.0
    lr_schedule=True
    total_epochs = args.wp_epoch + args.max_epoch

    # eval before training
    if args.eval_first and distributed_utils.is_main_process():
        # to check whether the evaluator can work
        model_eval = ema.ema if ema else model_without_ddp
        val_one_epoch(
            args=args, model=model_eval, evaluator=evaluator, optimizer=optimizer,
            epoch=0, best_map=best_map, path_to_save=path_to_save)

    # start to train
    for epoch in range(start_epoch, total_epochs):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)

        # train one epoch
        if epoch < args.wp_epoch:
            # warmup training loop
            train_with_warmup(epoch=epoch,
                              total_epochs=total_epochs,
                              args=args, 
                              device=device, 
                              ema=ema,
                              model=model,
                              criterion=criterion,
                              cfg=cfg, 
                              dataloader=dataloader, 
                              optimizer=optimizer, 
                              warmup_scheduler=warmup_scheduler,
                              scaler=scaler,
                              accumulate=accumulate)

        else:
            if epoch == args.wp_epoch:
                print('Warmup is Over !!!')
                warmup_scheduler.set_lr(optimizer, base_lr)
                
            # use cos lr decay
            T_max = total_epochs - cfg['no_aug_epoch']
            if epoch > T_max:
                print('Cosine annealing is over !!')
                lr_schedule = False
                for param_group in optimizer.param_groups:
                    param_group['lr'] = min_lr

            if lr_schedule:
                tmp_lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi*epoch / T_max))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = tmp_lr

            # train one epoch
            train_one_epoch(epoch=epoch,
                            total_epochs=total_epochs,
                            args=args, 
                            device=device,
                            ema=ema, 
                            model=model,
                            criterion=criterion,
                            cfg=cfg, 
                            dataloader=dataloader, 
                            optimizer=optimizer,
                            scaler=scaler,
                            accumulate=accumulate)
        
        # eval
        if (epoch % args.eval_epoch) == 0 or (epoch == total_epochs - 1):
            best_map = val_one_epoch(
                            args=args, 
                            model=ema.ema if ema else model_without_ddp, 
                            evaluator=evaluator,
                            optimizer=optimizer,
                            epoch=epoch,
                            best_map=best_map,
                            path_to_save=path_to_save)

        # close mosaic augmentation
        if dataloader.dataset.mosaic_prob > 0. and total_epochs - epoch == cfg['no_aug_epoch']:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic_prob = 0.
        # close mixup augmentation
        if dataloader.dataset.mixup_prob > 0. and total_epochs - epoch == cfg['no_aug_epoch']:
            print('close Mixup Augmentation ...')
            dataloader.dataset.mixup_prob = 0.


if __name__ == '__main__':
    train()
