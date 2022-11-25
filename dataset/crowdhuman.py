import os
import cv2
import json
import random
import numpy as np
import torch
import os.path as osp

try:
    from .transforms import  mosaic_x4_augment, mosaic_x9_augment, mixup_augment
except:
    from transforms import mosaic_x4_augment, mosaic_x9_augment, mixup_augment


CrowdHuman_CLASSES = ['person']


# CrowdHuman Detection
class CrowdHumanDetection(torch.utils.data.Dataset):

    def __init__(self, 
                 data_dir, 
                 img_size=640, 
                 image_set='val',                 
                 transform=None, 
                 mosaic_prob=0.0,
                 mixup_prob=0.0,
                 trans_config=None,
                 ignore_label=-1):
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_set = image_set
        self.img_folder = os.path.join(data_dir, 'Images')
        self.source = os.path.join(data_dir, 'annotation_{}.odgt'.format(image_set)) 

        self.records = self.load_json_lines(self.source)
        self.ignore_label = ignore_label

        # augmentation
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')


    def __getitem__(self, index):
        image, target = self.pull_item(index)
        
        return image, target


    def __len__(self):
        return len(self.records)


    def load_json_lines(self, fpath):
        assert os.path.exists(fpath)
        with open(fpath,'r') as fid:
            lines = fid.readlines()
        records = [json.loads(line.strip('\n')) for line in lines]
        return records


    def load_bbox(self, dict_input, key_name, key_box):
        assert key_name in dict_input
        if len(dict_input[key_name]) < 1:
            return np.empty([0, 5])
        else:
            assert key_box in dict_input[key_name][0]
        bbox = []
        for rb in dict_input[key_name]:
            if rb['tag'] == 'person':
                tag = 0
            else:
                tag = -1 # background
            if 'extra' in rb:
                if 'ignore' in rb['extra']:
                    if rb['extra']['ignore'] != 0:
                        tag = -1
            # check ttag
            if tag == self.ignore_label:
                continue
            else:
                bbox.append(np.hstack((rb[key_box], tag)))
        
        bboxes = np.vstack(bbox).astype(np.float64)
        # check bboxes
        keep = (bboxes[:, 2]>=0) * (bboxes[:, 3]>=0)
        bboxes = bboxes[keep, :]
        # [x1, y1, bw, bh] -> [x1, y1, x2, y2]
        bboxes[:, 2:4] += bboxes[:, :2]

        return bboxes


    def load_image_target(self, index):
        record = self.records[index]
        # load a image
        image_path = osp.join(self.img_folder, record['ID']+'.jpg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        # load a target
        anno = self.load_bbox(record, 'gtboxes', 'fbox')
        
        # Normalize bbox
        anno[:, [0, 2]] = np.clip(anno[:, [0, 2]], a_min=0., a_max=width)
        anno[:, [1, 3]] = np.clip(anno[:, [1, 3]], a_min=0., a_max=height)

        # check target
        if len(anno) == 0:
            anno = np.zeros([1, 5])
        else:
            anno = np.array(anno)

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }

        return image, target


    def load_mosaic(self, index, load_4x=False):
        if load_4x:
            # load a 4x mosaic image
            new_ids = np.arange(len(self.records)).tolist()
            new_ids = new_ids[:index] + new_ids[index+1:]
            # random sample other indexs
            id1 = index
            id2, id3, id4 = random.sample(new_ids, 3)
            ids = [id1, id2, id3, id4]
        else:
            # load a 9x mosaic image
            new_ids = np.arange(len(self.records)).tolist()
            new_ids = new_ids[:index] + new_ids[index+1:]
            # random sample other indexs
            id1 = index
            id2_9 = random.sample(new_ids, 8)
            ids = [id1] + id2_9

        image_list = []
        target_list = []
        # load image and target
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
                    new_index = np.random.randint(0, len(self.records))
                    new_image, new_target = self.load_mosaic(new_index, True)
                else:
                    new_index = np.random.randint(0, len(self.records))
                    new_image, new_target = self.load_mosaic(new_index, False)

                image, target = mixup_augment(image, target, new_image, new_target)
            
        # load an image and target
        else:
            image, target = self.load_image_target(index)

        # augment
        image, target = self.transform(image, target, mosaic)

        return image, target


    def pull_image(self, index):
        '''Returns the original image'''
        record = self.records[index]
        # image
        image_path = osp.join(self.img_folder, record['ID']+'.jpg')
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        return img, record


    def pull_anno(self, index):
        '''Returns the original annotation of image'''
        record = self.records[index]
        # load target
        target = self.load_bbox(record, 'gtboxes', 'fbox', class_names=CrowdHuman_CLASSES)
        
        return target, record


if __name__ == "__main__":
    from transforms import TrainTransforms, ValTransforms
    
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

    dataset = CrowdHumanDetection(
        data_dir='D:\\python_work\\object-detection\\dataset\\CrowdHuman',
        img_size=img_size,
        image_set='val',
        transform=train_transform,
        mosaic_prob=0.5,
        mixup_prob=0.15,
        trans_config=trans_config,
        ignore_label=-1
        )
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target= dataset.pull_item(i)
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
            label = 'face'
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
