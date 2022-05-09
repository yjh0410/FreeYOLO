# yolo config


yolo_config = {
    'yolo_x': {
        # input
        # If you have many GPUs, you could use large input size to enhance FreeYOLO.
        # 'train_size': 800,
        # 'test_size': 640,
        # 'random_size': [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # parameters affine
        'affine_params': {
            'degrees': 10.,
            'translate': 0.1,
            'shear': 2.0,
            'mosaic_scale': (0.1, 2.0),
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'cspdarknet',
        'depth': 1.33,
        'width': 1.25,
        'depthwise': False,
        'act_type': 'silu',
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': False,
        'fpn': 'yolo_pafpn',
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.01,
        'train_nms_thresh': 0.65,
        'test_nms_thresh': 0.45,
        # matcher
        'matcher': {'basic': {'object_sizes_of_interest': [[-1, 64], 
                                                           [64, 128], 
                                                           [128, float('inf')]],
                              'center_sampling_radius': 1.5}},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 5,
        },

    'yolo_l': {
        # input
        # If you have many GPUs, you could use large input size to enhance YOLO.
        # 'train_size': 800,
        # 'test_size': 640,
        # 'random_size': [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # parameters affine
        'affine_params': {
            'degrees': 10.,
            'translate': 0.1,
            'shear': 2.0,
            'mosaic_scale': (0.1, 2.0),
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'cspdarknet',
        'depth': 1.0,
        'width': 1.0,
        'depthwise': False,
        'act_type': 'silu',
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': False,
        'fpn': 'yolo_pafpn',
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.01,
        'train_nms_thresh': 0.65,
        'test_nms_thresh': 0.45,
        # matcher
        'matcher': {'basic': {'object_sizes_of_interest': [[-1, 64], 
                                                           [64, 128], 
                                                           [128, float('inf')]],
                              'center_sampling_radius': 1.5}},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 5,
        },

    'yolo_m': {
        # input
        # If you have many GPUs, you could use large input size to enhance YOLO.
        # 'train_size': 800,
        # 'test_size': 640,
        # 'random_size': [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic_prob': 1.0,
        'mixup_prob': 0.5,
        'format': 'RGB',
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # parameters affine
        'affine_params': {
            'degrees': 10.,
            'translate': 0.1,
            'shear': 2.0,
            'mosaic_scale': (0.1, 2.0),
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'cspdarknet',
        'depth': 0.67,
        'width': 0.75,
        'depthwise': False,
        'act_type': 'silu',
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': False,
        'fpn': 'yolo_pafpn',
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.01,
        'train_nms_thresh': 0.65,
        'test_nms_thresh': 0.45,
        # matcher
        'matcher': {'basic': {'object_sizes_of_interest': [[-1, 64], 
                                                           [64, 128], 
                                                           [128, float('inf')]],
                              'center_sampling_radius': 1.5}},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 5,
        },

    'yolo_s': {
        # input
        # If you have many GPUs, you could use large input size to enhance YOLO.
        # 'train_size': 800,
        # 'test_size': 640,
        # 'random_size': [448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # parameters affine
        'affine_params': {
            'degrees': 10.,
            'translate': 0.1,
            'shear': 2.0,
            'mosaic_scale': (0.5, 1.5),
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'cspdarknet',
        'depth': 0.33,
        'width': 0.5,
        'depthwise': False,
        'act_type': 'silu',
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': False,
        'fpn': 'yolo_pafpn',
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.01,
        'train_nms_thresh': 0.65,
        'test_nms_thresh': 0.45,
        # matcher
        'matcher': {'basic': {'object_sizes_of_interest': [[-1, 64], 
                                                           [64, 128], 
                                                           [128, float('inf')]],
                              'center_sampling_radius': 1.5}},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 5,
        },

    'yolo_t': {
        # input
        'train_size': 640,
        'test_size': 416,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic_prob': 1.0,
        'mixup_prob': 1.0,
        'format': 'RGB',
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # parameters affine
        'affine_params': {
            'degrees': 10.,
            'translate': 0.1,
            'shear': 2.0,
            'mosaic_scale': (0.5, 1.5),
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'cspdarknet',
        'depth': 0.33,
        'width': 0.375,
        'depthwise': False,
        'act_type': 'silu',
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': False,
        'fpn': 'yolo_pafpn',
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.01,
        'train_nms_thresh': 0.65,
        'test_nms_thresh': 0.45,
        # matcher
        'matcher': {'basic': {'object_sizes_of_interest': [[-1, 64], 
                                                           [64, 128], 
                                                           [128, float('inf')]],
                              'center_sampling_radius': 1.5}},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 32,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 5,
        },

    'yolo_n': {
        # input
        'train_size': 640,
        'test_size': 416,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic_prob': 0.5,
        'mixup_prob': 0.5,
        'format': 'RGB',
        'transforms': [{'name': 'DistortTransform',
                         'hue': 0.1,
                         'saturation': 1.5,
                         'exposure': 1.5},
                         {'name': 'RandomHorizontalFlip'},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'PadImage'}],
        # parameters affine
        'affine_params': {
            'degrees': 10.,
            'translate': 0.1,
            'shear': 2.0,
            'mosaic_scale': (0.5, 1.5),
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'cspdarknet',
        'depth': 0.33,
        'width': 0.25,
        'depthwise': True,
        'act_type': 'silu',
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': False,
        'fpn': 'yolo_pafpn',
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.01,
        'train_nms_thresh': 0.65,
        'test_nms_thresh': 0.45,
        # matcher
        'matcher': {'basic': {'object_sizes_of_interest': [[-1, 64], 
                                                           [64, 128], 
                                                           [128, float('inf')]],
                              'center_sampling_radius': 1.5}},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 64,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.05,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 5,
        },

}