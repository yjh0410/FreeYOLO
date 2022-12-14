from __future__ import division , print_function
"""WIDER Face Dataset Classes
author: swordli
"""
import cv2
import random
import numpy as np
import scipy.io
import os.path as osp
import torch.utils.data as data
import matplotlib.pyplot as plt
plt.switch_backend('agg')

try:
    from .transforms import  mosaic_x4_augment, mosaic_x9_augment, mixup_augment
except:
    from transforms import mosaic_x4_augment, mosaic_x9_augment, mixup_augment


WIDERFace_CLASSES = ['face']  # always index 0


class WIDERFaceDetection(data.Dataset):
    """WIDERFace Detection Dataset Object   
    http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/
    input is image, target is annotation
    Arguments:
        root (string): filepath to WIDERFace folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'WIDERFace')
    """

    def __init__(self, 
                 data_dir=None,
                 img_size=640,
                 image_set='train',
                 transform=None, 
                 mosaic_prob=0.0,
                 mixup_prob=0.0,
                 trans_config=None):
        self.data_dir = data_dir
        self.img_size = img_size
        self.image_set = image_set
        self.transform = transform

        # augmentation
        self.transform = transform
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.trans_config = trans_config
        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
        print('==============================')


        self.img_ids = list()
        self.label_ids = list()
        self.event_ids = list()

        if self.image_set == 'train':
            path_to_label = osp.join ( self.data_dir , 'wider_face_split' ) 
            path_to_image = osp.join ( self.data_dir , 'WIDER_train/images' )
            fname = "wider_face_train.mat"

        if self.image_set == 'val':
            path_to_label = osp.join ( self.data_dir , 'wider_face_split' ) 
            path_to_image = osp.join ( self.data_dir , 'WIDER_val/images' )
            fname = "wider_face_val.mat"

        if self.image_set == 'test':
            path_to_label = osp.join ( self.data_dir , 'wider_face_split' ) 
            path_to_image = osp.join ( self.data_dir , 'WIDER_test/images' )
            fname = "wider_face_test.mat"

        self.path_to_label = path_to_label
        self.path_to_image = path_to_image
        self.fname = fname
        self.f = scipy.io.loadmat(osp.join(self.path_to_label, self.fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
 
        self._load_widerface()


    def _load_widerface(self):

        error_bbox = 0 
        train_bbox = 0
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]

                if self.image_set in [ 'test']:
                    self.img_ids.append( osp.join(self.path_to_image, directory,  im_name + '.jpg') )
                    self.event_ids.append( directory )
                    self.label_ids.append([])
                    continue

                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                bboxes = []
                for i in range(face_bbx.shape[0]):
                    # filter bbox
                    if face_bbx[i][2] < 2 or face_bbx[i][3] < 2 or face_bbx[i][0] < 0 or face_bbx[i][1] < 0:
                        error_bbox +=1
                        #print (face_bbx[i])
                        continue 
                    train_bbox += 1 
                    xmin = float(face_bbx[i][0])
                    ymin = float(face_bbx[i][1])
                    xmax = float(face_bbx[i][2]) + xmin -1 	
                    ymax = float(face_bbx[i][3]) + ymin -1
                    bboxes.append([xmin, ymin, xmax, ymax, 0])

                if ( len(bboxes)==0 ):  #  filter bbox will make bbox none
                    continue
                self.img_ids.append( osp.join(self.path_to_image, directory,  im_name + '.jpg') )
                self.event_ids.append( directory )
                self.label_ids.append( bboxes )
                #yield DATA(os.path.join(self.path_to_image, directory,  im_name + '.jpg'), bboxes)
        print("Error bbox number to filter : %d,  bbox number: %d"  %(error_bbox , train_bbox))
        

    def __getitem__(self, index):
        image, target = self.pull_item(index)
        
        return image, target


    def __len__(self):
        return len(self.img_ids)


    def load_image_target(self, index):
        # load a target
        anno = self.label_ids[index]
        # load a image
        image = cv2.imread(self.img_ids[index])
        height, width, channels = image.shape

        # check target
        if len(anno) == 0:
            anno = np.zeros([1, 5])
        else:
            anno = np.array(anno)

        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [height, width]
        }
        print(target)
        
        return image, target


    def load_mosaic(self, index, load_4x=True):
        if load_4x:
            new_ids = np.arange(len(self.img_ids)).tolist()
            new_ids = new_ids[:index] + new_ids[index+1:]
            # random sample other indexs
            id1 = index
            id2, id3, id4 = random.sample(new_ids, 3)
            ids = [id1, id2, id3, id4]
        else:
            new_ids = np.arange(len(self.img_ids)).tolist()
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
                    new_index = np.random.randint(0, len(self.img_ids))
                    new_image, new_target = self.load_mosaic(new_index, True)
                else:
                    new_index = np.random.randint(0, len(self.img_ids))
                    new_image, new_target = self.load_mosaic(new_index, False)

                image, target = mixup_augment(image, target, new_image, new_target)

        # load an image and target
        else:
            image, target = self.load_image_target(index)

        # augment
        image, target = self.transform(image, target, mosaic)

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
        return cv2.imread(self.img_ids[index], cv2.IMREAD_COLOR), self.img_ids[index]


    def pull_event(self, index):
        return self.event_ids[index]


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
        img_id = self.img_ids[index]
        anno = self.label_ids[index]

        return img_id.split("/")[-1], anno


if __name__ == "__main__":
    import time
    import argparse
    from transforms import TrainTransforms, ValTransforms
    
    parser = argparse.ArgumentParser(description='FreeYOLO-Seg')

    # opt
    parser.add_argument('--root', default='D:\\python_work\\object-detection\\dataset\\WiderFace',
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

    dataset = WIDERFaceDetection(
        img_size=img_size,
        data_dir=args.root,
        image_set='val',
        transform=train_transform,
        mosaic_prob=0.5,
        mixup_prob=0.15,
        trans_config=trans_config
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
            label = WIDERFace_CLASSES[cls_id]
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, color, 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)