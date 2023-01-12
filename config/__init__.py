from .yolo_free_config import yolo_free_config
from .yolo_free_v2_config import yolo_free_v2_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if args.version in ['yolo_free_nano', 'yolo_free_tiny',
                        'yolo_free_large', 'yolo_free_huge']:
        cfg = yolo_free_config[args.version]

    elif args.version in ['yolo_free_v2_nano', 'yolo_free_v2_tiny',
                          'yolo_free_v2_large', 'yolo_free_v2_huge']:
        cfg = yolo_free_v2_config[args.version]

    return cfg
