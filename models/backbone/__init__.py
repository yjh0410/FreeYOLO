from .cspdarknet import build_cspdarknet


def build_backbone(cfg, trainable=False):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    # imagenet pretrained
    pretrained = cfg['pretrained'] and trainable

    if 'cspdarknet53' in cfg['backbone']:
        model, feat_dim = build_cspdarknet(pretrained=pretrained)

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
