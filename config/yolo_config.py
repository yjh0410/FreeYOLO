# yolo config


yolo_config = {
    'free_yolo_csp_d53': {
        # input
        'train_size': 640,
        'test_size': 640,
        'random_size': [320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640],
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
                         {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                         {'name': 'ToTensor'},
                         {'name': 'Resize'},
                         {'name': 'Normalize'},
                         {'name': 'PadImage'}],
        # model
        'backbone': 'cspdarknet53',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block',
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'yolo_pafpn',
        'fpn_depth': 3,
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'object_sizes_of_interest': [[-1, 64], [64, 128], [128, float('inf')]],
                    'center_sampling_radius': 1.5},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 200,
        'no_aug_epoch': 15,
        'batch_size': 16,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
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