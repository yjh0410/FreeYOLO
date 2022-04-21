from .yolopafpn import YoloPaFPN



def build_fpn(cfg, in_dims):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'yolo_pafpn':
        fpn_net = YoloPaFPN(in_dims=in_dims,
                            out_dim=cfg['head_dim'],
                            depth=cfg['depth'],
                            width=cfg['width'],
                            depthwise=cfg['depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])


    return fpn_net

