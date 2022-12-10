import random
import cv2
import math
import numpy as np
import torch
import torchvision.transforms.functional as F
import time


def refine_targets(target, img_size, min_box_size):
    # check target
    valid_bboxes = []
    valid_labels = []
    target_bboxes = target['boxes'].copy()
    target_labels = target['labels'].copy()

    if len(target_bboxes) > 0:
        # Cutout/Clip targets
        target_bboxes = np.clip(target_bboxes, 0, img_size)

        # check boxes
        target_bboxes_wh = target_bboxes[..., 2:] - target_bboxes[..., :2]
        min_tgt_boxes_size = np.min(target_bboxes_wh, axis=-1)

        keep = (min_tgt_boxes_size > min_box_size)

        valid_bboxes = target_bboxes[keep]
        valid_labels = target_labels[keep]
        
    else:
        valid_bboxes = target_bboxes
        valid_labels = target_labels

    # guard against no boxes via resizing
    valid_bboxes = valid_bboxes.reshape(-1, 4)
    valid_labels = valid_labels.reshape(-1)

    target['boxes'] = valid_bboxes
    target['labels'] = valid_labels

    return target


def random_perspective(image,
                       targets=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = image.shape[0] + border[0] * 2  # shape(h,w,c)
    width = image.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        targets[:, 1:5] = new

    return image, targets


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


# mosaic augment
def mosaic_x4_augment(image_list, target_list, img_size, affine_params=None):
    assert len(image_list) == 4

    mosaic_img = np.ones([img_size*2, img_size*2, image_list[0].shape[2]], dtype=np.uint8) * 114
    # mosaic center
    yc, xc = [int(random.uniform(-x, 2*img_size + x)) for x in [-img_size // 2, -img_size // 2]]
    # yc = xc = self.img_size

    mosaic_bboxes = []
    mosaic_labels = []
    for i in range(4):
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

        orig_h, orig_w, _ = img_i.shape

        # resize
        if np.random.randint(2):
            # keep aspect ratio
            r = img_size / max(orig_h, orig_w)
            if r != 1: 
                img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)))
        else:
            img_i = cv2.resize(img_i, (int(img_size), int(img_size)))
        h, w, _ = img_i.shape

        # place img in img4
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
        padw = x1a - x1b
        padh = y1a - y1b

        # labels
        bboxes_i_ = bboxes_i.copy()
        if len(bboxes_i) > 0:
            # a valid target, and modify it.
            bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padw)
            bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + padh)
            bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padw)
            bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + padh)    

            mosaic_bboxes.append(bboxes_i_)
            mosaic_labels.append(labels_i)

    if len(mosaic_bboxes) == 0:
        mosaic_bboxes = np.array([]).reshape(-1, 4)
        mosaic_labels = np.array([]).reshape(-1)
    else:
        mosaic_bboxes = np.concatenate(mosaic_bboxes)
        mosaic_labels = np.concatenate(mosaic_labels)

    # clip
    mosaic_bboxes = mosaic_bboxes.clip(0, img_size * 2)

    # random perspective
    mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
    mosaic_img, mosaic_targets = random_perspective(
        mosaic_img,
        mosaic_targets,
        affine_params['degrees'],
        translate=affine_params['translate'],
        scale=affine_params['scale'],
        shear=affine_params['shear'],
        perspective=affine_params['perspective'],
        border=[-img_size//2, -img_size//2]
        )

    # target
    mosaic_target = {
        "boxes": mosaic_targets[..., 1:],
        "labels": mosaic_targets[..., 0],
        "orig_size": [img_size, img_size]
    }

    return mosaic_img, mosaic_target


def mosaic_x9_augment(image_list, target_list, img_size, affine_params=None):
    assert len(image_list) == 9

    s = img_size
    mosaic_border = [-img_size//2, -img_size//2]
    mosaic_bboxes = []
    mosaic_labels = []
    for i in range(9):
        # Load image
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

        orig_h, orig_w, _ = img_i.shape

        # resize
        if np.random.randint(2):
            # keep aspect ratio
            r = img_size / max(orig_h, orig_w)
            if r != 1: 
                img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)))
        else:
            img_i = cv2.resize(img_i, (int(img_size), int(img_size)))
        h, w, _ = img_i.shape

        # place img in img9
        if i == 0:  # center
            mosaic_img = np.full((s * 3, s * 3, img_i.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            h0, w0 = h, w
            c = s, s, s + w, s + h  # xmin, ymin, xmax, ymax (base) coordinates
        elif i == 1:  # top
            c = s, s - h, s + w, s
        elif i == 2:  # top right
            c = s + wp, s - h, s + wp + w, s
        elif i == 3:  # right
            c = s + w0, s, s + w0 + w, s + h
        elif i == 4:  # bottom right
            c = s + w0, s + hp, s + w0 + w, s + hp + h
        elif i == 5:  # bottom
            c = s + w0 - w, s + h0, s + w0, s + h0 + h
        elif i == 6:  # bottom left
            c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
        elif i == 7:  # left
            c = s - w, s + h0 - h, s, s + h0
        elif i == 8:  # top left
            c = s - w, s + h0 - hp - h, s, s + h0 - hp

        padx, pady = c[:2]
        x1, y1, x2, y2 = [max(x, 0) for x in c]  # allocate coords

        # Image
        mosaic_img[y1:y2, x1:x2] = img_i[y1 - pady:, x1 - padx:]
        hp, wp = h, w  # height, width previous

        # Labels
        bboxes_i_ = bboxes_i.copy()
        if len(bboxes_i) > 0:
            # a valid target, and modify it.
            bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padx)
            bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + pady)
            bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padx)
            bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + pady)    

            mosaic_bboxes.append(bboxes_i_)
            mosaic_labels.append(labels_i)

    if len(mosaic_bboxes) == 0:
        mosaic_bboxes = np.array([]).reshape(-1, 4)
        mosaic_labels = np.array([]).reshape(-1)
    else:
        mosaic_bboxes = np.concatenate(mosaic_bboxes)
        mosaic_labels = np.concatenate(mosaic_labels)

    # Offset
    yc, xc = [int(random.uniform(0, s)) for _ in mosaic_border]  # mosaic center x, y
    mosaic_img = mosaic_img[yc:yc + 2 * s, xc:xc + 2 * s]
    mosaic_bboxes[..., [0, 2]] -= xc
    mosaic_bboxes[..., [1, 3]] -= yc

    # clip
    mosaic_bboxes = mosaic_bboxes.clip(0, img_size * 2)

    # random perspective
    mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
    mosaic_img, mosaic_targets = random_perspective(
        mosaic_img,
        mosaic_targets,
        affine_params['degrees'],
        translate=affine_params['translate'],
        scale=affine_params['scale'],
        shear=affine_params['shear'],
        perspective=affine_params['perspective'],
        border=mosaic_border
        )

    # target
    mosaic_target = {
        "boxes": mosaic_targets[..., 1:],
        "labels": mosaic_targets[..., 0],
        "orig_size": [img_size, img_size]
    }

    return mosaic_img, mosaic_target


# mixup augment
def mixup_augment(origin_image, origin_target, new_image, new_target):
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    mixup_image = r * origin_image.astype(np.float32) + \
                  (1.0 - r)* new_image.astype(np.float32)
    mixup_image = mixup_image.astype(np.uint8)
    
    cls_labels = new_target["labels"].copy()
    box_labels = new_target["boxes"].copy()

    mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
    mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)

    mixup_target = {
        "boxes": mixup_bboxes,
        "labels": mixup_labels,
        'orig_size': mixup_image.shape[:2]
    }
    
    return mixup_image, mixup_target
    


# TrainTransform
class TrainTransforms(object):
    def __init__(self, 
                 trans_config=None,
                 img_size=640, 
                 min_box_size=8):
        self.trans_config = trans_config
        self.img_size = img_size
        self.min_box_size = min_box_size


    def __call__(self, image, target, mosaic=False):
        # resize
        img_h0, img_w0 = image.shape[:2]

        r = self.img_size / max(img_h0, img_w0)
        if r != 1: 
            img = cv2.resize(image, (int(img_w0 * r), int(img_h0 * r)))
        else:
            img = image

        # rescale bboxes
        if target is not None:
            img_h, img_w = img.shape[:2]

        if not mosaic:
            # rescale bbox
            boxes_ = target["boxes"].copy()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # spatial augment
            target_ = np.concatenate(
                (target['labels'][..., None], target['boxes']), axis=-1)
            img, target_ = random_perspective(
                img, target_,
                degrees=self.trans_config['degrees'],
                translate=self.trans_config['translate'],
                scale=self.trans_config['scale'],
                shear=self.trans_config['shear'],
                perspective=self.trans_config['perspective']
                )
            target['boxes'] = target_[..., 1:]
            target['labels'] = target_[..., 0]
        
        # hsv augment
        augment_hsv(img, hgain=self.trans_config['hsv_h'], 
                    sgain=self.trans_config['hsv_s'], 
                    vgain=self.trans_config['hsv_v'])

        # random flip
        if random.random() < 0.5:
            w = img.shape[1]
            img = np.fliplr(img).copy()
            boxes = target['boxes'].copy()
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
            target["boxes"] = boxes

        # refine target
        target = refine_targets(target, self.img_size, self.min_box_size)
        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        assert max(img_h0, img_w0) <= self.img_size

        pad_image = torch.ones([img_tensor.size(0), self.img_size, self.img_size]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor

        return pad_image, target


# ValTransform
class ValTransforms(object):
    def __init__(self, 
                 img_size=640):
        self.img_size =img_size


    def __call__(self, image, target=None, mosaic=False):
        # resize
        img_h0, img_w0 = image.shape[:2]

        r = self.img_size / max(img_h0, img_w0)
        if r != 1: 
            img = cv2.resize(image, (int(img_w0 * r), int(img_h0 * r)))
        else:
            img = image

        img_h, img_w = img.shape[:2]

        # to tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()

        # rescale bboxes
        if target is not None:
            # rescale bbox
            boxes_ = target["boxes"].copy()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

            # refine target
            target = refine_targets(target, self.img_size, 8)

            # to tensor
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        # pad img
        img_h0, img_w0 = img_tensor.shape[1:]
        assert max(img_h0, img_w0) <= self.img_size

        if img_h0 > img_w0:
            pad_img_h = self.img_size
            pad_img_w = (img_w0 // 32 + 1) * 32
        elif img_h0 < img_w0:
            pad_img_h = (img_h0 // 32 + 1) * 32
            pad_img_w = self.img_size
        else:
            pad_img_h = self.img_size
            pad_img_w = self.img_size
        pad_image = torch.ones([img_tensor.size(0), pad_img_h, pad_img_w]).float() * 114.
        pad_image[:, :img_h0, :img_w0] = img_tensor

        return pad_image, target

