from .yolox_config import yolox_config


def build_config(args):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    if 'yolox' in args.version:
        cfg = yolox_config[args.version]

    return cfg
