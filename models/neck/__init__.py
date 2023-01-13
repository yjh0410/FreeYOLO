from .spp import SPPBlockCSP, SPPF
from .elan_pafpn import ELAN_PaFPN


def build_fpn(cfg, in_dims, out_dim=None):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'elan_pafpn':
        fpn_net = ELAN_PaFPN(in_dims=in_dims,
                            out_dim=out_dim,
                            fpn_size=cfg['fpn_size'],
                            depthwise=cfg['fpn_depthwise'],
                            head_conv_elan=cfg['head_conv_elan'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])

    return fpn_net


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'spp_block_csp':
        neck = SPPBlockCSP(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    elif model == 'sppf':
        neck = SPPF(
            in_dim=in_dim, 
            out_dim=out_dim,
            expand_ratio=cfg['expand_ratio'],
            pooling_size=cfg['pooling_size']
            )

    return neck
    