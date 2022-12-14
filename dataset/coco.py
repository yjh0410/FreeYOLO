import os
import random
import numpy as np
import time

import torch
from torch.utils.data import Dataset
import cv2

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")

try:
    from .transforms import mosaic_x4_augment, mosaic_x9_augment, mixup_augment
except:
    from transforms import mosaic_x4_augment, mosaic_x9_augment, mixup_augment


coco_class_labels = ('background',
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

coco_class_index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                    70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


class COCODataset(Dataset):
    """
    COCO dataset class.
    """
    def __init__(self, 
                 img_size=640,
                 data_dir=None, 
                 image_set='train2017',
                 transform=None,
                 mosaic_prob=0.,
                 mixup_prob=0.0,
                 trans_config=None):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            debug (bool): if True, only one data id is selected from the dataset
        """
        if image_set == 'train2017':
            self.json_file='instances_train2017.json'
        elif image_set == 'val2017':
            self.json_file='instances_val2017.json'
        elif image_set == 'test2017':
            self.json_file='image_info_test-dev2017.json'
        self.img_size = img_size
        self.image_set = image_set
        self.data_dir = data_dir
        self.coco = COCO(os.path.join(self.data_dir, 'annotations', self.json_file))
        self.ids = self.coco.getImgIds()
        self.class_ids = sorted(self.coco.getCatIds())
        # augmentation
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.trans_config = trans_config
        
        print('==============================')
        print('Image Set: {}'.format(image_set))
        print('Json file: {}'.format(self.json_file))
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')



    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target


    def load_image_target(self, index):
        anno_ids = self.coco.getAnnIds(imgIds=[int(index)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)

        # load an image
        img_file = os.path.join(self.data_dir, self.image_set,
                                '{:012}'.format(index) + '.jpg')
        image = cv2.imread(img_file)
        
        if self.json_file == 'instances_val5k.json' and image is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(index) + '.jpg')
            image = cv2.imread(img_file)

        assert image is not None

        height, width, channels = image.shape
        
        #load a target
        anno = []
        for label in annotations:
            if 'bbox' in label and label['area'] > 0:   
                xmin = np.max((0, label['bbox'][0]))
                ymin = np.max((0, label['bbox'][1]))
                xmax = np.min((width - 1, xmin + np.max((0, label['bbox'][2] - 1))))
                ymax = np.min((height - 1, ymin + np.max((0, label['bbox'][3] - 1))))
                if xmax > xmin and ymax > ymin:
                    label_ind = label['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    anno.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            # else:
            #     print('No bbox !!!')

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }
        
        return image, target


    def load_mosaic(self, index, load_4x=True):
        if load_4x:
            # load 4x mosaic image
            ids_list_ = self.ids[:index] + self.ids[index+1:]
            # random sample other indexs
            id1 = self.ids[index]
            id2, id3, id4 = random.sample(ids_list_, 3)
            ids = [id1, id2, id3, id4]

        else:
            # load 9x mosaic image
            ids_list_ = self.ids[:index] + self.ids[index+1:]
            # random sample other indexs
            id1 = self.ids[index]
            id2_9 = random.sample(ids_list_, 8)
            ids = [id1] + id2_9

        image_list = []
        target_list = []
        for id_ in ids:
            img_i, target_i = self.load_image_target(id_)
            image_list.append(img_i)
            target_list.append(target_i)

        if load_4x:
            image, target = mosaic_x4_augment(
                image_list, target_list, self.img_size, self.trans_config)
        else:
            image, target = mosaic_x9_augment(
                image_list, target_list, self.img_size, self.trans_config)

        return image, target

        
    def pull_item(self, index):
        # load a mosaic image
        mosaic = False
        if random.random() < self.mosaic_prob:
            mosaic = True
            if random.random() < 0.8:
                image, target = self.load_mosaic(index, True)
            else:
                image, target = self.load_mosaic(index, False)
            # MixUp
            if random.random() < self.mixup_prob:
                if random.random() < 0.8:
                    new_index = np.random.randint(0, len(self.ids))
                    new_image, new_target = self.load_mosaic(new_index, True)
                else:
                    new_index = np.random.randint(0, len(self.ids))
                    new_image, new_target = self.load_mosaic(new_index, False)

                image, target = mixup_augment(image, target, new_image, new_target)

        # load an image and target
        else:
            img_id = self.ids[index]
            image, target = self.load_image_target(img_id)

        # augment
        image, target = self.transform(image, target, mosaic)

        return image, target


    def pull_image(self, index):
        id_ = self.ids[index]
        img_file = os.path.join(self.data_dir, self.image_set,
                                '{:012}'.format(id_) + '.jpg')
        img = cv2.imread(img_file)

        if self.json_file == 'instances_val5k.json' and img is None:
            img_file = os.path.join(self.data_dir, 'train2017',
                                    '{:012}'.format(id_) + '.jpg')
            img = cv2.imread(img_file)

        return img, id_


    def pull_anno(self, index):
        id_ = self.ids[index]

        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        
        anno = []
        for label in annotations:
            if 'bbox' in label:
                xmin = np.max((0, label['bbox'][0]))
                ymin = np.max((0, label['bbox'][1]))
                xmax = xmin + label['bbox'][2]
                ymax = ymin + label['bbox'][3]
                
                if label['area'] > 0 and xmax >= xmin and ymax >= ymin:
                    label_ind = label['category_id']
                    cls_id = self.class_ids.index(label_ind)

                    anno.append([xmin, ymin, xmax, ymax, cls_id])  # [xmin, ymin, xmax, ymax, label_ind]
            else:
                print('No bbox !!')
        return anno


if __name__ == "__main__":
    import time
    import argparse
    from transforms import TrainTransforms, ValTransforms
    
    parser = argparse.ArgumentParser(description='FreeYOLO-Seg')

    # opt
    parser.add_argument('--root', default='D:\\python_work\\object-detection\\dataset\\COCO',
                        help='data root')

    args = parser.parse_args()
    
    img_size = 640
    trans_config = {
        'degrees': 0.0,
        'translate': 0.2,
        'scale': 0.9,
        'shear': 0.0,
        'perspective': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4
    }
    train_transform = TrainTransforms(
        trans_config=trans_config,
        img_size=img_size,
        min_box_size=8
        )

    val_transform = ValTransforms(
        img_size=img_size,
        )

    dataset = COCODataset(
        img_size=img_size,
        data_dir=args.root,
        image_set='val2017',
        transform=train_transform,
        mosaic_prob=0.5,
        mixup_prob=0.15,
        trans_config=trans_config
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        t0 = time.time()
        image, target = dataset.pull_item(i)
        print('load data time: {:6f}'.format((time.time() - t0)*1000))
        
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            color = class_colors[cls_id]
            # class name
            label = coco_class_labels[coco_class_index[cls_id]]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)