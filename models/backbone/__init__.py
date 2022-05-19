from .cspdarknet import build_cspdarknet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if 'cspdarknet53' in cfg['backbone']:
        model, feat_dim = build_cspdarknet(pretrained=cfg['pretrained'])

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
