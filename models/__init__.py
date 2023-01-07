#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from .yolo_free.build import build_yolo_free
from .yolo_anchor.build import build_yolo_anchor


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
    
    if args.version in ['yolo_free_nano', 'yolo_free_tiny',
                        'yolo_free_large', 'yolo_free_huge']:
        model, criterion = build_yolo_free(
            args, cfg, device, num_classes, trainable)
    elif args.version in ['yolo_anchor_nano', 'yolo_anchor_tiny',
                          'yolo_anchor_large', 'yolo_anchor_huge']:
        model, criterion = build_yolo_anchor(
            args, cfg, device, num_classes, trainable)

    if trainable:
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

        return model, criterion

    else:      
        return model
