from .cspdarknet53 import build_cspdarknet53
from .shufflenetv2 import build_shufflenetv2
from .elannet import build_elannet
from .cspdarknet_tiny import build_cspdarknet_tiny


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
            
    elif cfg['backbone'] == 'shufflenet_v2':
        model, feat_dim = build_shufflenetv2(
            model_size=cfg['model_size'],
            pretrained=pretrained
        )

    elif cfg['backbone'] == 'elannet':
        model, feat_dim = build_elannet(
            pretrained=pretrained
        )

    elif cfg['backbone'] == 'cspdarknet_tiny':
        model, feat_dim = build_cspdarknet_tiny(
            pretrained=pretrained
        )
        
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
