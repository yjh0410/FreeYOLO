from .spp import SPP
from .yolofpn import YoloFPN
from .yolopafpn import YoloPaFPN


def build_neck(cfg, in_dim, out_dim):
    model = cfg['neck']
    print('==============================')
    print('Neck: {}'.format(model))
    # build neck
    if model == 'spp':
        neck = SPP(in_dim, 
                   out_dim, 
                   e=cfg['expand_ratio'], 
                   kernel_sizes=cfg['kernel_sizes'],
                   norm_type=cfg['neck_norm'],
                   act_type=cfg['neck_act'])

    return neck


def build_fpn(cfg, in_dims):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'yolo_fpn':
        fpn_net = YoloFPN(in_dims=in_dims,
                          out_dim=cfg['head_dim'],
                          norm_type=cfg['fpn_norm'],
                          act_type=cfg['fpn_act'],
                          spp=cfg['use_spp'])

    elif model == 'yolo_pafpn':
        fpn_net = YoloPaFPN(in_dims=in_dims,
                            out_dim=cfg['head_dim'],
                            depth=cfg['depth'],
                            depthwise=cfg['depthwise'],
                            norm_type=cfg['fpn_norm'],
                            act_type=cfg['fpn_act'],
                            spp=cfg['use_spp'])


    return fpn_net

