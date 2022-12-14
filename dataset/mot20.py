import os
import random
import numpy as np

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


mot_class_labels = ('person',)



class MOT20Dataset(Dataset):
    """
    MOT20 dataset class.
    """
    def __init__(self, 
                 img_size=640,
                 data_dir=None, 
                 image_set='train',
                 json_file='train_half.json',
                 transform=None,
                 mosaic_prob=0.0,
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
        self.img_size = img_size
        self.image_set = image_set
        self.json_file = json_file
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
        print('Json file: {}'.format(json_file))
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target


    def load_image_target(self, img_id):
        im_ann = self.coco.loadImgs(img_id)[0] 

        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=0)
        annotations = self.coco.loadAnns(anno_ids)

        # load an image
        img_file = os.path.join(
                self.data_dir, self.image_set, im_ann["file_name"])
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
        im_ann = self.coco.loadImgs(id_)[0] 
        img_file = os.path.join(
                self.data_dir, self.image_set, im_ann["file_name"])
        image = cv2.imread(img_file)

        return image, id_


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

    dataset = MOT20Dataset(
        img_size=img_size,
        data_dir=args.root,
        image_set='train',
        json_file='val_half.json',
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
        image, target = dataset.pull_item(i)
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
            label = mot_class_labels[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)