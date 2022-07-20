from .cspdarknet53 import build_cspdarknet53
from .cspdarknets import build_cspd
from .elannet import build_elannet


def build_backbone(cfg, trainable=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # imagenet pretrained
    pretrained = cfg['pretrained'] and trainable
    try:
        res5_dilation = cfg['res5_dilation']
    except:
        res5_dilation=False

    if cfg['backbone'] == 'cspdarknet53':
        model, feat_dim = build_cspdarknet53(
            pretrained=pretrained,
            res5_dilation=res5_dilation
            )
    elif 'cspd' in cfg['backbone']:
        model, feat_dim = build_cspd(
            model_name=cfg['backbone'],
            pretrained=pretrained
        )

    elif cfg['backbone'] == 'elannet':
        model, feat_dim = build_elannet(
            pretrained=pretrained
        )
        
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
