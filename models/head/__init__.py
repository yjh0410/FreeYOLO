from ..head.yolopafpn import YoloPaFPN


def build_fpn(cfg, in_dims):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'yolo_pafpn':
        fpn_net = YoloPaFPN(in_dims=in_dims,
                            depth=cfg['fpn_depth'],
                            depthwise=cfg['fpn_depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'])


    return fpn_net

