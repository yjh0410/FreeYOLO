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

try:
    from .transforms import  mosaic_augment, mixup_augment
except:
    from transforms import mosaic_augment, mixup_augment


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
                 image_sets='val',
                 transform=None,
                 color_augment=None, 
                 mosaic_prob=0.0,
                 mixup_prob=0.0):

        self.data_dir = data_dir
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform

        # augmentation
        self.transform = transform
        self.color_augment = color_augment
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        if self.mosaic_prob > 0.:
            print('use Mosaic Augmentation ...')
        if self.mixup_prob > 0.:
            print('use Mixup Augmentation ...')


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

                if self.image_set in [ 'test' , 'val']:
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
        
        return image, target


    def load_mosaic(self, index):
        new_ids = np.arange(len(self.img_ids)).tolist()
        new_ids = new_ids[:index] + new_ids[index+1:]
        # random sample other indexs
        id1 = index
        id2, id3, id4 = random.sample(new_ids, 3)
        ids = [id1, id2, id3, id4]

        image_list = []
        target_list = []
        # load image and target
        for id_ in ids:
            img_i, target_i = self.load_image_target(id_)
            image_list.append(img_i)
            target_list.append(target_i)

        image, target = mosaic_augment(image_list, target_list, self.img_size)
        
        return image, target


    def pull_item(self, index):
        # load a mosaic image
        if random.random() < self.mosaic_prob:
            image, target = self.load_mosaic(index)

            # MixUp
            if random.random() < self.mixup_prob:
                new_index = np.random.randint(0, len(self.img_ids))
                new_image, new_target = self.load_mosaic(new_index)

                image, target = mixup_augment(image, target, new_image, new_target)

            # augment
            image, target = self.color_augment(image, target)
            
        # load an image and target
        else:
            image, target = self.load_image_target(index)

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
        gt = self.target_transform(anno, 1, 1)
        return img_id.split("/")[-1], gt


if __name__ == "__main__":
    from transforms import BaseTransforms, TrainTransforms, ValTransforms

    img_size = 640
    format = 'RGB'
    pixel_mean = [123.675, 116.28, 103.53]
    pixel_std = [58.395, 57.12, 57.375]
    trans_config = [{'name': 'DistortTransform',
                     'hue': 0.1,
                     'saturation': 1.5,
                     'exposure': 1.5},
                    {'name': 'RandomHorizontalFlip'},
                    {'name': 'JitterCrop', 'jitter_ratio': 0.3},
                    {'name': 'ToTensor'},
                    {'name': 'Resize'},
                    {'name': 'Normalize'},
                    {'name': 'PadImage'}]
    transform = TrainTransforms(
        trans_config=trans_config,
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        format=format
        )
    color_augment = BaseTransforms(
        img_size=img_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        format=format
        )

    dataset = WIDERFaceDetection(
                           data_dir='E:\\python_work\\object_detection\\dataset\\WiderFace',
                           img_size=img_size,
                           transform=transform,
                           color_augment=color_augment,
                           mosaic_prob=0.5,
                           mixup_prob=0.5)
    
    np.random.seed(0)
    print('Data length: ', len(dataset))

    for i in range(1000):
        image, target= dataset[i]
        # to numpy
        image = image.permute(1, 2, 0).numpy()
        # to BGR format
        if format == 'RGB':
            # denormalize
            image = image * pixel_std + pixel_mean
            image = image[:, :, (2, 1, 0)].astype(np.uint8)
        elif format == 'BGR':
            image = image * pixel_std + pixel_mean
            image = image.astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        boxes = target["boxes"]
        labels = target["labels"]

        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cls_id = int(label)
            # class name
            label = 'face'
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            # put the test on the bbox
            cv2.putText(image, label, (int(x1), int(y1 - 5)), 0, 0.5, (0,0,255), 1, lineType=cv2.LINE_AA)
        cv2.imshow('gt', image)
        # cv2.imwrite(str(i)+'.jpg', img)
        cv2.waitKey(0)
