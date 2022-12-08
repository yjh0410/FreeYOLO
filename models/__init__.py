#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolo_free.loss import build_criterion
from .yolo_free.yolo_free import FreeYOLO


# build object detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)
    
    model = FreeYOLO(
        cfg=cfg,
        device=device, 
        num_classes=num_classes,
        trainable=trainable,
        conf_thresh=cfg['conf_thresh'],
        nms_thresh=cfg['nms_thresh'],
        topk=args.topk,
        no_decode=args.no_decode
        )

    # Load pretrained weight
    if args.pretrained is not None:
        print('Loading COCO pretrained weight ...')
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
                    print(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)

    # keep training
    if args.resume is not None:
        print('keep training: ', args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    if trainable:
        # build criterion for training
        criterion = build_criterion(cfg, device, num_classes)
        return model, criterion
    else:      
        return model
