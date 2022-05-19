from .spp import SPPBlockCSP


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'spp_block':
        neck = SPPBlockCSP(in_dim, 
                           out_dim, 
                           expand_ratio=cfg['expand_ratio'], 
                           pooling_size=cfg['pooling_size'],
                           act_type=cfg['neck_act'],
                           norm_type=cfg['neck_norm'],
                           depthwise=cfg['neck_depthwise'])

    return neck
