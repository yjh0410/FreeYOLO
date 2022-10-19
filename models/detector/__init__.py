import torch
from .yolo_free.build import build_yolo_free
from .yolo_anchor.build import build_yolo_anchor
from .yolo_tiny.build import build_yolo_tiny
from .yolo_nano.build import build_yolo_nano
from .yolof.build import build_yolof


# build object detector
def build_model(args, 
                cfg,
                device, 
                num_classes=80, 
                trainable=False, 
                coco_pretrained=None,
                resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))
    
    print('==============================')
    print('Model Configuration: \n', cfg)

    # build yolo
    if args.version == 'yolo_free':
        model, criterion = build_yolo_free(args, cfg, device, num_classes, trainable)

    elif args.version == 'yolo_anchor':
        model, criterion = build_yolo_anchor(args, cfg, device, num_classes, trainable)

    elif args.version == 'yolo_tiny':
        model, criterion = build_yolo_tiny(args, cfg, device, num_classes, trainable)

    elif args.version == 'yolo_nano':
        model, criterion = build_yolo_nano(args, cfg, device, num_classes, trainable)

    elif args.version == 'yolof':
        model, criterion = build_yolof(args, cfg, device, num_classes, trainable)

    # Load COCO pretrained weight
    if coco_pretrained is not None:
        print('Loading COCO pretrained weight ...')
        checkpoint = torch.load(coco_pretrained, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
                    print(k)
            else:
                checkpoint_state_dict.pop(k)
                print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)

    # keep training
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location='cpu')
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("model")
        model.load_state_dict(checkpoint_state_dict)

    if trainable:
        return model, criterion
    else:      
        return model
