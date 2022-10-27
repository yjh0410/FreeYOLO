from .yolo_config import yolo_config


def build_config(args):
    print('==============================')
    print('Config: {} ...'.format(args.version.upper()))
    
    if 'yolo' in args.version:
        cfg = yolo_config[args.version]

    return cfg
