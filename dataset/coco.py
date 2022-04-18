import os
import random
import numpy as np

from torch.utils.data import Dataset
import cv2

try:
    from pycocotools.coco import COCO
except:
    print("It seems that the COCOAPI is not installed.")


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
                 mosaic=False,
                 mixup=False):
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
        self.mosaic = mosaic
        self.mixup = mixup
        if self.mosaic:
            print('use Mosaic Augmentation ...')
        if self.mixup:
            print('use Mixup Augmentation ...')


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
            else:
                print('No bbox !!!')

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }
        
        return image, target


    def load_mosaic(self, index):
        ids_list_ = self.ids[:index] + self.ids[index+1:]
        # random sample other indexs
        id1 = self.ids[index]
        id2, id3, id4 = random.sample(ids_list_, 3)
        ids = [id1, id2, id3, id4]

        img_lists = []
        tg_lists = []
        # load image and target
        for id_ in ids:
            img_i, target_i = self.load_image_target(id_)
            img_lists.append(img_i)
            tg_lists.append(target_i)

        mosaic_img = np.zeros([self.img_size*2, self.img_size*2, img_i.shape[2]], dtype=np.uint8)
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]
        # yc = xc = self.img_size

        mosaic_bboxes = []
        mosaic_labels = []
        for i in range(4):
            img_i, target_i = img_lists[i], tg_lists[i]
            bboxes_i = target_i["boxes"]
            labels_i = target_i["labels"]

            h0, w0, _ = img_i.shape
            s = np.random.randint(5, 21) / 10.

            # resize
            if np.random.randint(2):
                # keep aspect ratio
                r = self.img_size / max(h0, w0)
                if r != 1: 
                    img_i = cv2.resize(img_i, (int(w0 * r * s), int(h0 * r * s)))
            else:
                img_i = cv2.resize(img_i, (int(self.img_size * s), int(self.img_size * s)))
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            bboxes_i_ = bboxes_i.copy()
            if len(bboxes_i) > 0:
                # a valid target, and modify it.
                bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / w0 + padw)
                bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / h0 + padh)
                bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / w0 + padw)
                bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / h0 + padh)    

                mosaic_bboxes.append(bboxes_i_)
                mosaic_labels.append(labels_i)


        valid_bboxes = []
        valid_labels = []
        # check target
        if len(mosaic_bboxes) > 0:
            mosaic_bboxes = np.concatenate(mosaic_bboxes)
            mosaic_labels = np.concatenate(mosaic_labels)
            # Cutout/Clip targets
            np.clip(mosaic_bboxes, 0, 2 * self.img_size, out=mosaic_bboxes)

            # check boxes
            for box, label in zip(mosaic_bboxes, mosaic_labels):
                x1, y1, x2, y2 = box
                bw, bh = x2 - x1, y2 - y1
                if bw > 10. and bh > 10.:
                    valid_bboxes.append([x1, y1, x2, y2])
                    valid_labels.append(label)

        # guard against no boxes via resizing
        valid_bboxes = np.array(valid_bboxes).reshape(-1, 4)
        valid_labels = np.array(valid_labels).reshape(-1)
        mosaic_bboxes = np.array(valid_bboxes)
        mosaic_labels = np.array(valid_labels)

        # target
        mosaic_target = {
            "boxes": mosaic_bboxes,
            "labels": mosaic_labels,
            "orig_size": [self.img_size*2, self.img_size*2]
        }
        
        return mosaic_img, mosaic_target


    def pull_item(self, index):
        # load a mosaic image
        if self.mosaic:
            image, target = self.load_mosaic(index)

            if self.mixup:
                image2, target2 = self.load_mosaic(np.random.randint(0, len(self.ids)))
                r = np.random.beta(8.0, 8.0)  # mixup ratio, alpha=beta=8.0
                image = (image * r + image2 * (1 - r)).astype(np.uint8)
                target["boxes"]= np.concatenate((target["boxes"], target2["boxes"]), 0)
                target["labels"]= np.concatenate((target["labels"], target2["labels"]), 0)

        # load an image and target
        else:
            img_id = self.ids[index]
            image, target = self.load_image_target(img_id)

        # augment
        image, target = self.transform(image, target)
            
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
    from transforms import TrainTransforms, ValTransforms

    format = 'RGB'
    trans_config = [{'name': 'DistortTransform',
                     'hue': 0.1,
                     'saturation': 1.5,
                     'exposure': 1.5},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'ToTensor'},
                    {'name': 'Resize'},
                    {'name': 'PadImage'}]
    img_size = 640
    transform = TrainTransforms(trans_config=trans_config,
                                img_size=img_size,
                                format=format)
    dataset = COCODataset(img_size=img_size,
                          data_dir='E:\\python_work\\object_detection\\dataset\\COCO',
                          image_set='train2017',
                          transform=transform,
                          mosaic=True,
                          mixup=True)
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(80)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target = dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to BGR format
        if format == 'RGB':
            image = image[:, :, (2, 1, 0)].astype(np.uint8)
        elif format == 'BGR':
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
