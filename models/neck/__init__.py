from .spp import SPPBlockCSP
from ..neck.yolopafpn import YoloPaFPN


def build_fpn(cfg, in_dims, out_dim):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'yolo_pafpn':
        fpn_net = YoloPaFPN(in_dims=in_dims,
                            out_dim=out_dim,
                            depth=cfg['fpn_depth'],
                            depthwise=cfg['fpn_depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])


    return fpn_net


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
