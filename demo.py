import argparse
from ast import arg
import cv2
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dataset.transforms import ValTransforms
from utils.misc import load_weight
from utils.vis_tools import visualize

from config import build_config
from models import build_model



def parse_args():
    parser = argparse.ArgumentParser(description='FreeYOLO Demo')

    # basic
    parser.add_argument('-size', '--img_size', default=640, type=int,
                        help='the max size of input image')
    parser.add_argument('--mode', default='image',
                        type=str, help='Use the data from image, video or camera')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use cuda')
    parser.add_argument('--path_to_img', default='dataset/demo/images/',
                        type=str, help='The path to image files')
    parser.add_argument('--path_to_vid', default='dataset/demo/videos/',
                        type=str, help='The path to video files')
    parser.add_argument('--path_to_save', default='det_results/demos/',
                        type=str, help='The path to save the detection results')
    parser.add_argument('-vs', '--visual_threshold', default=0.35, type=float,
                        help='Final confidence threshold for visualization')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show visualization')

    # model
    parser.add_argument('-v', '--version', default='yolo_free_large', type=str,
                        help='build yolo')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=100, type=int,
                        help='topk candidates for testing')
    parser.add_argument("--no_decode", action="store_true", default=False,
                        help="not decode in inference or yes")

    return parser.parse_args()
                    

def detect(args,
           net, 
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
                orig_h, orig_w, _ = frame.shape

                # prepare
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)
                # inference
                t0 = time.time()
                bboxes, scores, labels = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes *= max(orig_h, orig_w)
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            labels=labels,
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
            orig_h, orig_w, _ = image.shape

            # prepare
            x = transform(image)[0]
            x = x.unsqueeze(0).to(device)

            # inference
            t0 = time.time()
            bboxes, scores, labels = net(x)
            t1 = time.time()
            print("detection time used ", t1-t0, "s")

            # rescale
            bboxes *= max(orig_h, orig_w)
            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

            img_processed = visualize(img=image, 
                                      bboxes=bboxes,
                                      scores=scores, 
                                      labels=labels,
                                      class_colors=class_colors,
                                      vis_thresh=vis_thresh)
            cv2.imwrite(os.path.join(save_path, str(i).zfill(6)+'.jpg'), img_processed)
            if args.show:
                cv2.imshow('detection', img_processed)
                cv2.waitKey(0)

    # ------------------------- Video ---------------------------
    elif mode == 'video':
        video = cv2.VideoCapture(path_to_vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        save_size = (640, 480)
        cur_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
        save_path = os.path.join(save_path, cur_time+'.avi')
        fps = 30.0
        out = cv2.VideoWriter(save_path, fourcc, fps, save_size)
        print(save_path)

        while(True):
            ret, frame = video.read()
            
            if ret:
                # ------------------------- Detection ---------------------------
                orig_h, orig_w, _ = frame.shape

                # prepare
                x = transform(frame)[0]
                x = x.unsqueeze(0).to(device)

                # inference
                t0 = time.time()
                bboxes, scores, labels = net(x)
                t1 = time.time()
                print("detection time used ", t1-t0, "s")

                # rescale
                bboxes *= max(orig_h, orig_w)
                bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=orig_w)
                bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=orig_h)

                frame_processed = visualize(img=frame, 
                                            bboxes=bboxes,
                                            scores=scores, 
                                            labels=labels,
                                            class_colors=class_colors,
                                            vis_thresh=vis_thresh)

                frame_processed_resize = cv2.resize(frame_processed, save_size)
                out.write(frame_processed_resize)
                if args.show:
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

    # config
    cfg = build_config(args)

    # build model
    model = build_model(args=args, 
                        cfg=cfg,
                        device=device, 
                        num_classes=80, 
                        trainable=False)

    # load trained weight
    model = load_weight(model=model, path_to_ckpt=args.weight)
    model.to(device).eval()

    # transform
    transform = ValTransforms(img_size=args.img_size)

    # run
    detect(args=args, net=model, 
            device=device,
            transform=transform,
            mode=args.mode,
            path_to_img=args.path_to_img,
            path_to_vid=args.path_to_vid,
            path_to_save=args.path_to_save,
            vis_thresh=args.visual_threshold)


if __name__ == '__main__':
    run()
