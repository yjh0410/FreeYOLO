import argparse
import numpy as np
import time
import os
import torch

from dataset.transforms import ValTransforms
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from utils.com_flops_params import FLOPs_and_Params
from utils import fuse_conv_bn
from utils.misc import load_weight

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLO')
    # Model
    parser.add_argument('-v', '--version', default='yolo_free_large', type=str,
                        help='build yolo')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse conv and bn')
    parser.add_argument('--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('--nms_thresh', default=0.45, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='NMS threshold')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")

    # data root
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    # basic
    parser.add_argument('-size', '--img_size', default=608, type=int,
                        help='the min size of input image')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    # cuda
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')

    return parser.parse_args()


def test(net, device, img_size, testset, transform):
    # Step-1: Compute FLOPs and Params
    FLOPs_and_Params(model=net, 
                     img_size=img_size, 
                     device=device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            image, _ = testset.pull_image(index)

            orig_h, orig_w, _ = image.shape

            # prepare
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)

            # star time
            torch.cuda.synchronize()
            start_time = time.perf_counter()    

            # inference
            bboxes, scores, cls_inds = net(x)
            
            # rescale
            bboxes *= max(orig_h, orig_w)

            # end time
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            # print("detection time used ", elapsed, "s")
            if index > 1:
                total_time += elapsed
                count += 1
            
        print('- FPS :', 1.0 / (total_time / count))



if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    print('test on coco-val ...')
    data_dir = os.path.join(args.root, 'COCO')
    class_names = coco_class_labels
    class_indexs = coco_class_index
    num_classes = 80
    dataset = COCODataset(
                data_dir=data_dir,
                image_set='val2017',
                img_size=args.img_size)

    # config
    cfg = build_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load trained weight
    model = load_weight(device=device, 
                        model=model, 
                        path_to_ckpt=args.weight)


    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn.fuse_conv_bn(model)

    # transform
    transform = ValTransforms(img_size=args.img_size)

    # run
    test(net=model, 
        img_size=args.img_size,
        device=device, 
        testset=dataset,
        transform=transform
        )
