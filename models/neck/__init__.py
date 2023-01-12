from .spp import SPPBlock, SPPBlockCSP, SPPF, SPPFBlockCSP
from .pafpn import PaFPNCSP, PaFPNELAN


def build_fpn(cfg, in_dims, out_dim):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'pafpn_csp':
        fpn_net = PaFPNCSP(in_dims=in_dims,
                            out_dim=out_dim,
                            depth=cfg['fpn_depth'],
                            depthwise=cfg['fpn_depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])

    elif model == 'pafpn_elan':
        fpn_net = PaFPNELAN(in_dims=in_dims,
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
    if model == 'spp_block':
        neck = SPPBlock(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )
            
    elif model == 'spp_block_csp':
        neck = SPPBlockCSP(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    elif model == 'sppf':
        neck = SPPF(in_dim, out_dim, pooling_size=cfg['pooling_size'])


    elif model == 'sppf_block_csp':
        neck = SPPBlockCSP(
            in_dim, out_dim, 
            expand_ratio=cfg['expand_ratio'], 
            pooling_size=cfg['pooling_size'],
            act_type=cfg['neck_act'],
            norm_type=cfg['neck_norm'],
            depthwise=cfg['neck_depthwise']
            )

    return neck
    