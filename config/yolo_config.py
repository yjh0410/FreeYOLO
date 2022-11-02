# yolo config


yolo_config = {
    'yolo_free_large': {
        # input
        'train_size': 800,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
        'mosaic_prob': 0.5,
        'mixup_prob': 0.5,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'elannet',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 512,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'large', # 'large', 'tiny'
        'fpn_dim': [512, 1024, 512],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 250,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'accumulate': 1, # 4
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05, # 0.05
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_huge': {
        # input
        'train_size': 800,
        'test_size': 640,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640, 672,
                        704, 736, 768, 800],
        'mosaic_prob': 0.5,
        'mixup_prob': 0.5,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'elannet_huge',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 640,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'huge', # 'tiny', 'large', 'huge'
        'fpn_dim': [640, 1280, 640],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 320,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 250,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'accumulate': 1, # 4
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05, # 0.05
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_tiny': {
        # input
        'train_size': 640,
        'test_size': 416,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640],
        'mosaic_prob': 0.5,
        'mixup_prob': 0.1,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 256,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'tiny', # 'tiny', 'large', 'huge
        'fpn_dim': [128, 256, 256],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 250,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'accumulate': 1, # 4
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05, # 0.05
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

    'yolo_free_nano': {
        # input
        'train_size': 640,
        'test_size': 416,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640],
        'mosaic_prob': 0.5,
        'mixup_prob': 0.1,
        'format': 'RGB',
        'pixel_mean': [123.675, 116.28, 103.53],
        'pixel_std': [58.395, 57.12, 57.375],
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'elannet_nano',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'spp_block_dw',
        'neck_dim': 128,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'nano',
        'fpn_dim': [64, 128, 128],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'center_sampling_radius': 2.5,
                    'topk_candicate': 10},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 250,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'accumulate': 1, # 4
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05, # 0.05
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

}