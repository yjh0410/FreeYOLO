from .cspdarknet import build_cspdarknet


def build_backbone(cfg, trainable=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # imagenet pretrained
    pretrained = cfg['pretrained'] and trainable
    try:
        res5_dilation = cfg['res5_dilation']
    except:
        res5_dilation=False

    if 'cspdarknet53' in cfg['backbone']:
        model, feat_dim = build_cspdarknet(
            pretrained=pretrained,
            res5_dilation=res5_dilation
            )

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
