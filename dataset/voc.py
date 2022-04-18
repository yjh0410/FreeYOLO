"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import torch.utils.data as data
import cv2
import numpy as np
import random
import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt if i % 2 == 0 else cur_pt
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [x1, y1, x2, y2, label_ind]

        return res  # [[x1, y1, x2, y2, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, 
                 img_size=640,
                 data_dir=None,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, 
                 target_transform=VOCAnnotationTransform(),
                 mosaic=False,
                 mixup=False):
        self.root = data_dir
        self.img_size = img_size
        self.image_set = image_sets
        self.target_transform = target_transform
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        # augmentation
        self.transform = transform
        self.mosaic = mosaic
        self.mixup = mixup
        if self.mosaic:
            print('use Mosaic Augmentation ...')
        if self.mixup:
            print('use Mixup Augmentation ...')


    def __getitem__(self, index):
        image, target = self.pull_item(index)
        return image, target


    def __len__(self):
        return len(self.ids)


    def load_image_target(self, img_id):
        # load an image
        image = cv2.imread(self._imgpath % img_id)
        height, width, channels = image.shape

        # laod an annotation
        anno = ET.parse(self._annopath % img_id).getroot()
        if self.target_transform is not None:
            anno = self.target_transform(anno)

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
            if len(valid_labels) == 0:
                    valid_bboxes.append([0., 0., 0., 0.])
                    valid_labels.append(0.)

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
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR), img_id


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt


if __name__ == "__main__":
    from transforms import TrainTransforms, ValTransforms
    # format = 'BGR'
    # pixel_mean = [103.53, 116.28, 123.675]
    # pixel_std = [1.0, 1.0, 1.0]

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

    dataset = VOCDetection(img_size=img_size,
                           data_dir='E:\\python_work\\object_detection\\dataset\\VOCdevkit',
                           transform=transform,
                           mosaic=True,
                           mixup=True)
    
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(20)]
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target= dataset.pull_item(i)
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to BGR format
        if format == 'RGB':
            # denormalize
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
            label = VOC_CLASSES[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
