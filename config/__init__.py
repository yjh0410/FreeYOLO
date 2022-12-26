from .yolo_free_config import yolo_free_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if 'yolo' in args.version:
        cfg = yolo_free_config[args.version]

    return cfg
