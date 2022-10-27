import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.voc import VOC_CLASSES, VOCDetection
from dataset.coco import coco_class_index, coco_class_labels, COCODataset
from dataset.transforms import ValTransforms
from utils.misc import load_weight, TestTimeAugmentation
from utils.vis_tools import visualize
from utils import fuse_conv_bn

from config import build_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLO')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--visual_threshold', default=0.35, type=float,
                        help='Final confidence threshold')

    # model
    parser.add_argument('-v', '--version', default='yolo_free', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                        help='fuse conv and bn')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')
    # TTA
    parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                        help='use test augmentation.')
    parser.add_argument('-fp', '--flip', action='store_true', default=False,
                        help='use flip in test augmentation.')
    parser.add_argument('--tta_min_size', default=640, type=int,
                        help='use flip in test augmentation.')
    parser.add_argument('--tta_max_size', default=640, type=int,
                        help='use flip in test augmentation.')

    return parser.parse_args()


@torch.no_grad()
def test(args,
         net, 
         device, 
         dataset,
         transforms=None,
         vis_thresh=0.4, 
         class_colors=None, 
         class_names=None, 
         class_indexs=None, 
         show=False,
         test_aug=None, 
         dataset_name='coco'):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        orig_h, orig_w, _ = image.shape

        # prepare
        x = transforms(image)[0]
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # inference
        if test_aug is not None:
            # test augmentation:
            bboxes, scores, labels = test_aug(x, net)
        else:
            bboxes, scores, labels = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale
        bboxes *= max(orig_h, orig_w)
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

        # vis detection
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            labels=labels,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=dataset_name)
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


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
        data_dir = os.path.join(args.root, 'VOCdevkit')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(data_dir=data_dir,
                               image_sets=[('2007', 'test')],
                               transform=None)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(data_dir=data_dir,
                              image_set='val2017',
                              transform=None)
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # config
    cfg = build_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)
    model = model.to(device).eval()

    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn(model)

    # TTA
    test_aug = TestTimeAugmentation(
        num_classes=num_classes,
        min_size=args.tta_min_size,
        max_size=args.tta_max_size,
        test_flip=args.flip) if args.test_aug else None

    # transform
    transform = ValTransforms(
        img_size=args.img_size,
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        format=cfg['format']
        )

    # run
    test(args=args,
        net=model, 
        device=device, 
        dataset=dataset,
        transforms=transform,
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        show=args.show,
        test_aug=test_aug,
        dataset_name=args.dataset)
