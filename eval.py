import argparse
import os

import torch

from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.coco_evaluator import COCOAPIEvaluator

from dataset.transforms import ValTransforms
from utils.misc import load_weight, TestTimeAugmentation

from config import build_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')
    # basic
    parser.add_argument('--min_size', default=800, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1333, type=int,
                        help='the min size of input image')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    # model
    parser.add_argument('-v', '--version', default='yolof50', type=str,
                        help='build YOLOF')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')

    return parser.parse_args()



def voc_test(model, data_dir, device, transform):
    evaluator = VOCAPIEvaluator(data_root=data_dir,
                                device=device,
                                transform=transform,
                                display=True)

    # VOC evaluation
    evaluator.evaluate(model)


def coco_test(model, data_dir, device, transform, test=False):
    if test:
        # test-dev
        print('test on test-dev 2017')
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=True,
                        transform=transform)

    else:
        # eval
        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        device=device,
                        testset=False,
                        transform=transform)

    # COCO evaluation
    evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    if args.dataset == 'voc':
        print('eval on voc ...')
        num_classes = 20
        data_dir = os.path.join(args.root, 'VOCdevkit')
    elif args.dataset == 'coco-val':
        print('eval on coco-val ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    elif args.dataset == 'coco-test':
        print('eval on coco-test-dev ...')
        num_classes = 80
        data_dir = os.path.join(args.root, 'COCO')
    else:
        print('unknow dataset !! we only support voc, coco-val, coco-test !!!')
        exit(0)


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

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None

    # transform
    transform = ValTransforms(min_size=cfg['test_min_size'], 
                              max_size=cfg['test_max_size'],
                              pixel_mean=cfg['pixel_mean'],
                              pixel_std=cfg['pixel_std'],
                              format=cfg['format'],
                              padding=cfg['val_padding'])

    # evaluation
    with torch.no_grad():
        if args.dataset == 'voc':
            voc_test(model, data_dir, device, transform)
        elif args.dataset == 'coco-val':
            coco_test(model, data_dir, device, transform, test=False)
        elif args.dataset == 'coco-test':
            coco_test(model, data_dir, device, transform, test=True)
