import torch
import torch.distributed as dist

import time
import os

from utils import distributed_utils
from utils.misc import get_total_grad_norm


def train_with_warmup(args, 
                      device, 
                      model, 
                      cfg, 
                      base_lr,
                      dataloader, 
                      optimizer, 
                      warmup_scheduler):
    epoch_size = len(dataloader)
    # start training loop
    for epoch in range(cfg['wp_epoch']):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            warmup_scheduler.warmup(ni, optimizer)

            # to device
            images = images.to(device)
            masks = masks.to(device)

            # inference
            loss_dict = model(images, mask=masks, targets=targets)
            losses = loss_dict['losses']

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            losses.backward()
            if args.grad_clip_norm > 0.:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad()

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                log = dict(
                    lr=round(cur_lr_dict['lr'], 6),
                    lr_bk=round(cur_lr_dict['lr_bk'], 6)
                )
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, cfg['max_epoch'])
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[gnorm: {:.2f}]'.format(total_norm)
                log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()
        
        print('Warmup is Over !!!')
        warmup_scheduler.set_lr(optimizer, base_lr, base_lr)


def train_one_epoch(args, 
                    device, 
                    model, 
                    cfg, 
                    dataloader, 
                    optimizer, 
                    lr_scheduler):
    epoch_size = len(dataloader)
    # start training loop
    for epoch in range(cfg['max_epoch']):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            # to device
            images = images.to(device)
            masks = masks.to(device)

            # inference
            loss_dict = model(images, mask=masks, targets=targets)
            losses = loss_dict['losses']

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            losses.backward()
            if args.grad_clip_norm > 0.:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad()

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                log = dict(
                    lr=round(cur_lr_dict['lr'], 6),
                    lr_bk=round(cur_lr_dict['lr_bk'], 6)
                )
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, cfg['max_epoch'])
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[gnorm: {:.2f}]'.format(total_norm)
                log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        lr_scheduler.step()


def val_one_epoch(args, 
                 model, 
                 evaluator,
                 optimizer,
                 lr_scheduler,
                 epoch,
                 path_to_save):
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch + 1))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
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
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch,
                                    'args': args}, 
                                    checkpoint_path)                      

                    # set train mode.
                    model.trainable = True
                    model.train()
        
            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()
