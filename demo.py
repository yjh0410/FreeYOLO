import argparse
import cv2
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from config.yolof_config import yolof_config
from dataset.coco import coco_class_index, coco_class_labels, COCODataset
from dataset.transforms import ValTransforms
from utils.misc import load_weight

from models.detector import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark Demo')

    # basic
    parser.add_argument('--min_size', default=800, type=int,
                        help='the min size of input image')
    parser.add_argument('--max_size', default=1333, type=int,
                        help='the min size of input image')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='data/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='data/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/images/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('--path_to_saveVid', default='data/videos/result.avi',
                        type=str, help='The path to save the detection results video')

    # model
    parser.add_argument('-v', '--version', default='yolof50', type=str,
                        help='build yolof')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=100, type=int,
                        help='NMS threshold')
    
    return parser.parse_args()
                    

def plot_bbox_labels(img, bbox, label, cls_color, test_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * test_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, test_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, class_colors, vis_thresh=0.3):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_color = class_colors[int(cls_inds[i])]
            cls_id = coco_class_index[int(cls_inds[i])]
            mess = '%s: %.2f' % (coco_class_labels[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, test_scale=ts)

    return img


def detect(net, 
           device, 
           transform, 
           vis_thresh, 
           mode='image', 
           path_to_img=None, 
           path_to_vid=None, 
           path_to_save=None):
    # class color
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    save_path = os.path.join(path_to_save, mode)
    os.makedirs(save_path, exist_ok=True)

    # ------------------------- Camera ----------------------------
    if mode == 'camera':
        print('use camera !!!')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            ret, frame = cap.read()
            if ret:
                if cv2.waitKey(1) == ord('q'):
                    break
                h, w, _ = frame.shape
                orig_size = np.array([[w, h, w, h]])

                # prepare
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes *= orig_size
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=h)

                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)
                cv2.imshow('detection result', frame_processed)
                cv2.waitKey(1)
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

    # ------------------------- Image ----------------------------
    elif mode == 'image':
        for i, img_id in enumerate(os.listdir(path_to_img)):
            image = cv2.imread(path_to_img + '/' + img_id, cv2.IMREAD_COLOR)
            h, w, _ = image.shape
            orig_size = np.array([[w, h, w, h]])

            # prepare
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)
            # inference
            t0 = time.time()
            bboxes, scores, cls_inds = net(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")

            # rescale
            bboxes *= orig_size
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=h)

            img_processed = visualize(img=image, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      cls_inds=cls_inds,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)

            cv2.imshow('detection', img_processed)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        save_path = os.path.join(save_path, 'det.avi')
        fps = 15.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                h, w, _ = frame.shape
                orig_size = np.array([[w, h, w, h]])

                # prepare
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, cls_inds = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes *= orig_size
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=h)

                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            cls_inds=cls_inds,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)

                frame_processed_resize = cv2.resize(frame_processed, save_size)
                out.write(frame_processed_resize)
                cv2.imshow('detection', frame_processed)
                cv2.waitKey(1)
            else:
                break
        video.release()
        out.release()
        cv2.destroyAllWindows()


def run():
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(0)

    # YOLOF config
    print('Model: ', args.version)
    cfg = yolof_config[args.version]

    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=80, 
                        trainable=False)

    # load trained weight
    model = load_weight(device=device, 
                        model=model, 
                        path_to_ckpt=args.weight)


    # transform
    transform = ValTransforms(min_size=cfg['test_min_size'], 
                              max_size=cfg['test_max_size'],
                              pixel_mean=cfg['pixel_mean'],
                              pixel_std=cfg['pixel_std'],
                              format=cfg['format'],
                              padding=cfg['val_padding'])

    # run
    detect(net=model, 
            device=device,
            transform=transform,
            mode=args.mode,
            path_to_img=args.path_to_img,
            path_to_vid=args.path_to_vid,
            path_to_save=args.path_to_save,
            vis_thresh=cfg['test_score_thresh'])


if __name__ == '__main__':
    run()
