from .cspdarknet import build_cspdarknet


def build_backbone(cfg):
    print('==============================')
    print('Backbone: {}'.format(cfg['backbone'].upper()))

    if 'cspdarknet' in cfg['backbone']:
        model, feat_dim = build_cspdarknet(depth=cfg['depth'], 
                                           width=cfg['width'], 
                                           depthwise=cfg['depthwise'],
                                           act_type=cfg['act_type'])

    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dim
