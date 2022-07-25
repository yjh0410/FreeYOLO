# yolo config


yolo_config = {
    'yolo_free_v1': {
        # input
        'train_size': 640,
        'test_size': 608,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640],
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
        'neck': 'spp_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_csp',
        'fpn_depth': 3,
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
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
        'max_epoch': 250,
        'no_aug_epoch': -1,
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

    'yolo_free_v2': {
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
        'backbone': 'cspdarknet53',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_csp',
        'fpn_depth': 3,
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

    'yolo_free_v3': {
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
        'pretrained': False,
        'stride': [8, 16, 32],  # P3, P4, P5
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
        'head_dim': [256, 512, 1024],
        'head_norm': 'BN',
        'head_act': 'silu',
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

    'yolo_free_tiny': {
        # input
        'train_size': 640,
        'test_size': 416,
        'random_size': [320, 352, 384, 416,
                        448, 480, 512, 544,
                        576, 608, 640],
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
        'backbone': 'elannet_tiny',
        'pretrained': False,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block',
        'neck_dim': 256,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'tiny', # 'large', 'tiny'
        'fpn_dim': [128, 256, 512],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head_dim': [128, 256, 512],
        'head_norm': 'BN',
        'head_act': 'lrelu',
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

    'yolo_anchor': {
        # input
        'train_size': 640,
        'test_size': 608,
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
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        'num_anchors': 3,  # number of anchor boxes on each level
        # neck
        'neck': 'spp_block_csp',
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_csp',
        'fpn_depth': 3,
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'iou_thresh': 0.5},
        # loss
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 5.0,
        # training configuration
        'max_epoch': 250,
        'no_aug_epoch': -1,
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

    'yolof': {
        # input
        'train_size': 640,
        'test_size': 608,
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
        'res5_dilation': True,
        'stride': 16,
        'anchor_size': [[16, 16],   [32, 32],   [64, 64],
                        [128, 128], [256, 256], [512, 512]],
        'ctr_clamp': 32,
        # neck
        'neck': 'dilated_encoder',
        'expand_ratio': 0.25,
        'dilation_list': [1, 2, 3, 4, 5, 6, 7, 8],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 512,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 4,
        'head_depthwise': False,
        # post process
        'conf_thresh': 0.01,
        'nms_thresh': 0.5,
        # matcher
        'matcher': {'topk_candidates': 4,
                    'igt': 0.7,
                    'iou_t': 0.15},
        # loss
        'alpha': 0.25,
        'gamma': 2.0,
        'loss_obj_weight': 1.0,
        'loss_cls_weight': 1.0,
        'loss_reg_weight': 1.0,
        # training configuration
        'max_epoch': 150,
        'no_aug_epoch': -1,
        'batch_size': 16,
        'base_lr': 0.01 / 64.,
        'min_lr_ratio': 0.01,
        # optimizer
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        # warmup strategy
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_epoch': 1,
        },

}