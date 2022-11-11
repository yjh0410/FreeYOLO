import torch
import torch.nn as nn
from torch import optim


def build_optimizer(cfg, model, base_lr=0.0, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # no decay
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(pg0, lr=base_lr, momentum=cfg['momentum'], nesterov=True)
    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(pg0, lr=base_lr, betas=(cfg['momentum'], 0.999))  # adjust beta1 to momentum
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(pg0, lr=base_lr, betas=(cfg['momentum'], 0.999))  # adjust beta1 to momentum
          
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": cfg['weight_decay']}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    del pg0, pg1, pg2

    start_epoch = 0
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume)
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)
        start_epoch = checkpoint.pop("epoch")
                        
                                
    return optimizer, start_epoch
