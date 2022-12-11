import torch
import torch.distributed as dist

import time
import os
import numpy as np

from utils import distributed_utils


def rescale_image_targets(images, targets, new_img_size):
    """
        Deployed for Multi scale trick.
    """
    # During training phase, the shape of input image is square.
    old_img_size = images.shape[-1]
    # interpolate
    images = torch.nn.functional.interpolate(
                        input=images, 
                        size=new_img_size, 
                        mode='bilinear', 
                        align_corners=False)
    # rescale targets
    for tgt in targets:
        boxes = tgt["boxes"].clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
        tgt["boxes"] = boxes

    return images, targets


def train_with_warmup(epoch,
                      total_epochs,
                      args, 
                      device, 
                      ema,
                      model,
                      criterion,
                      cfg, 
                      dataloader, 
                      optimizer, 
                      warmup_scheduler,
                      scaler,
                      accumulate):
    epoch_size = len(dataloader)
    img_size = cfg['train_size']
    t0 = time.time()
    # train one epoch
    for iter_i, (images, targets) in enumerate(dataloader):
        ni = iter_i + epoch * epoch_size
        # warmup
        warmup_scheduler.warmup(ni, optimizer)

        # to device
        images = images.to(device)

        # multi scale
        # # choose a new image size
        if ni % 10 == 0 and cfg['random_size']:
            idx = np.random.randint(len(cfg['random_size']))
            img_size = cfg['random_size'][idx]
        # # rescale data with new image size
        if cfg['random_size']:
            images, targets = rescale_image_targets(images, targets, img_size)

        # inference
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images)
            # loss
            loss_dict = criterion(outputs=outputs, targets=targets)
            losses = loss_dict['losses']

        # reduce            
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

        # check loss
        if torch.isnan(losses):
            print('loss is NAN !!')
            continue

        if args.distributed:
            # gradient averaged between devices in DDP mode
            losses *= distributed_utils.get_world_size()

        # backward
        scaler.scale(losses).backward()

        # Optimize
        if ni % accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # ema
            if ema:
                ema.update(model)

        # display
        if distributed_utils.is_main_process() and iter_i % 10 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
            # basic infor
            log =  '[Epoch: {}/{}]'.format(epoch+1, total_epochs)
            log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
            log += '[lr: {:.6f}]'.format(cur_lr[0])
            # loss infor
            for k in loss_dict_reduced.keys():
                if k == 'losses' and args.distributed:
                    world_size = distributed_utils.get_world_size()
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k] / world_size)
                else:
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

            # other infor
            log += '[time: {:.2f}]'.format(t1 - t0)
            log += '[size: {}]'.format(img_size)

            # print log infor
            print(log, flush=True)
            
            t0 = time.time()


def train_one_epoch(epoch,
                    total_epochs,
                    args, 
                    device, 
                    ema,
                    model,
                    criterion,
                    cfg, 
                    dataloader, 
                    optimizer,
                    scaler,
                    accumulate):
    epoch_size = len(dataloader)
    img_size = cfg["train_size"]
    t0 = time.time()
    # train one epoch
    for iter_i, (images, targets) in enumerate(dataloader):
        ni = iter_i + epoch * epoch_size
        # to device
        images = images.to(device)

        # multi scale
        # # choose a new image size
        if ni % 10 == 0 and cfg['random_size']:
            idx = np.random.randint(len(cfg['random_size']))
            img_size = cfg['random_size'][idx]
        # # rescale data with new image size
        if cfg['random_size']:
            images, targets = rescale_image_targets(images, targets, img_size)

        # inference
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images)
            # loss
            loss_dict = criterion(outputs=outputs, targets=targets)
            losses = loss_dict['losses']

        # reduce            
        loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

        # check loss
        if torch.isnan(losses):
            print('loss is NAN !!')
            continue

        if args.distributed:
            # gradient averaged between devices in DDP mode
            losses *= distributed_utils.get_world_size()

        # backward
        scaler.scale(losses).backward()

        # Optimize
        if ni % accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # ema
            if ema:
                ema.update(model)

        # display
        if distributed_utils.is_main_process() and iter_i % 10 == 0:
            t1 = time.time()
            cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
            # basic infor
            log =  '[Epoch: {}/{}]'.format(epoch+1, total_epochs)
            log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
            log += '[lr: {:.6f}]'.format(cur_lr[0])
            # loss infor
            for k in loss_dict_reduced.keys():
                if k == 'losses' and args.distributed:
                    world_size = distributed_utils.get_world_size()
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k] / world_size)
                else:
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

            # other infor
            log += '[time: {:.2f}]'.format(t1 - t0)
            log += '[size: {}]'.format(img_size)

            # print log infor
            print(log, flush=True)
            
            t0 = time.time()


def val_one_epoch(args, 
                  model, 
                  evaluator,
                  optimizer,
                  epoch,
                  best_map,
                  path_to_save):
    # check evaluator
    if distributed_utils.is_main_process():
        if evaluator is None:
            print('No evaluator ... save model and go on training.')
            print('Saving state, epoch: {}'.format(epoch + 1))
            weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
            checkpoint_path = os.path.join(path_to_save, weight_name)
            torch.save({'model': model.state_dict(),
                        'mAP': -1.,
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args}, 
                        checkpoint_path)                      
            
        else:
            print('eval ...')
            # set eval mode
            model.trainable = False
            model.eval()

            # evaluate
            evaluator.evaluate(model)

            cur_map = evaluator.map
            if cur_map > best_map:
                # update best-map
                best_map = cur_map
                # save model
                print('Saving state, epoch:', epoch + 1)
                weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                checkpoint_path = os.path.join(path_to_save, weight_name)
                torch.save({'model': model.state_dict(),
                            'mAP': round(best_map*100, 1),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args}, 
                            checkpoint_path)                      

            # set train mode.
            model.trainable = True
            model.train()

    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    return best_map
