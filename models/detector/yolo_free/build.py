from .yolo_free import FreeYOLO
from .loss import build_criterion


# build object detector
def build_yolo_free(args, 
                    cfg,
                    device, 
                    num_classes=80, 
                    trainable=False):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    model = FreeYOLO(cfg=cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=cfg['conf_thresh'],
                        nms_thresh=cfg['nms_thresh'],
                        topk=args.topk)

    criterion = None
    # build criterion for training
    if trainable:
        criterion = build_criterion(cfg, device, num_classes)

    return model, criterion
