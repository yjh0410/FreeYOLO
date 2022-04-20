# yolox config


yolox_config = {
    'yolox_d53': {
        # input
        'img_size': 640,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
        'mosaic': True,
        'mixup': True,
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
            'scales': (0.1, 2.0),  # mosaic scale
            'shear': 2.0,
            'mixup_scale': (0.5, 1.5)},
        # model
        'backbone': 'darknet53',
        'pretrained': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'use_spp': True,
        'fpn': 'yolo_fpn',
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        # post process
        'conf_thresh': 0.001,
        'train_nms_thresh': 0.6,
        'test_nms_thresh': 0.45,
        # scale range
        'object_sizes_of_interest': [[-1, 64], [64, 128], [128, float('inf')]],
        # matcher
        'matcher': 'matcher',
        'center_sampling_radius': 1.5,
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 300,
        'no_aug_epoch': 15,
        'batch_size': 8,
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