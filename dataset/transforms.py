import random
import cv2
import math
import numpy as np
import torch
import torchvision.transforms.functional as F


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


def mosaic_augment(image_list, target_list, img_size, affine_params):
    # prepare empty mosaic image
    mosaic_img = np.zeros([img_size*2, img_size*2, image_list[0].shape[2]], dtype=np.uint8)
    # mosaic center
    yc, xc = [int(random.uniform(-x, 2*img_size + x)) 
                for x in [-img_size // 2, -img_size // 2]]
    # yc = xc = img_size

    mosaic_bboxes = []
    mosaic_labels = []
    for i in range(4):
        img_i, target_i = image_list[i], target_list[i]
        bboxes_i = target_i["boxes"]
        labels_i = target_i["labels"]

        h0, w0, _ = img_i.shape

        # resize
        if np.random.randint(2):
            # keep aspect ratio
            r = img_size / max(h0, w0)
            if r != 1: 
                img_i = cv2.resize(img_i, (int(w0 * r), int(h0 * r)))
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
            bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / w0 + padw)
            bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / h0 + padh)
            bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / w0 + padw)
            bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / h0 + padh)    

            mosaic_bboxes.append(bboxes_i_)
            mosaic_labels.append(labels_i)

    mosaic_bboxes = np.concatenate(mosaic_bboxes)
    mosaic_labels = np.concatenate(mosaic_labels)
    mosaic_tgts = np.concatenate([mosaic_bboxes.clip(0, img_size * 2), 
                                  mosaic_labels[..., None]], 
                                  axis=-1)
    # affine
    mosaic_img, mosaic_tgts = random_affine(
        mosaic_img,
        mosaic_tgts,
        target_size=(img_size*2, img_size*2),
        degrees=affine_params['degrees'],
        translate=affine_params['translate'],
        scales=affine_params['mosaic_scale'],
        shear=affine_params['shear'],
    )
    mosaic_img = cv2.resize(mosaic_img, (img_size, img_size))
    mosaic_bboxes = mosaic_tgts[..., :4] / 2.0
    mosaic_labels = mosaic_tgts[..., 4]

    # check target
    valid_bboxes = []
    valid_labels = []
    if len(mosaic_bboxes) > 0:
        # Cutout/Clip targets
        np.clip(mosaic_bboxes, 0, img_size, out=mosaic_bboxes)

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
        "orig_size": [img_size, img_size]
    }
    
    return mosaic_img, mosaic_target


def mixup_augment(origin_image, origin_target, new_image, new_target, img_size, mixup_scale):
    jit_factor = random.uniform(*mixup_scale)
    FLIP = random.uniform(0, 1) > 0.5

    cp_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 114

    cp_scale_ratio = min(img_size / new_image.shape[0], img_size / new_image.shape[1])
    resized_img = cv2.resize(
        new_image,
        (int(new_image.shape[1] * cp_scale_ratio), int(new_image.shape[0] * cp_scale_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )

    cp_img[
        : int(new_image.shape[0] * cp_scale_ratio), : int(new_image.shape[1] * cp_scale_ratio)
    ] = resized_img

    cp_img = cv2.resize(
        cp_img,
        (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
    )
    cp_scale_ratio *= jit_factor

    if FLIP:
        cp_img = cp_img[:, ::-1, :]

    origin_h, origin_w = cp_img.shape[:2]
    target_h, target_w = origin_image.shape[:2]
    padded_img = np.zeros(
        (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=cp_img.dtype
    )
    padded_img[:origin_h, :origin_w] = cp_img

    x_offset, y_offset = 0, 0
    if padded_img.shape[0] > target_h:
        y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
    if padded_img.shape[1] > target_w:
        x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
    padded_cropped_img = padded_img[
        y_offset: y_offset + target_h, x_offset: x_offset + target_w
    ]

    cp_bboxes = new_target["boxes"].copy()
    cp_bboxes_origin_np = np.zeros_like(cp_bboxes)
    cp_bboxes_origin_np[:, 0::2] = np.clip(cp_bboxes[:, 0::2] * cp_scale_ratio, 0, origin_w)
    cp_bboxes_origin_np[:, 1::2] = np.clip(cp_bboxes[:, 1::2] * cp_scale_ratio, 0, origin_h)

    if FLIP:
        cp_bboxes_origin_np[:, 0::2] = (
            origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
        )

    cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
    cp_bboxes_transformed_np[:, 0::2] = np.clip(
        cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
    )
    cp_bboxes_transformed_np[:, 1::2] = np.clip(
        cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
    )

    mixup_image = 0.5 * origin_image.astype(np.float32) + 0.5 * padded_cropped_img.astype(np.float32)
    mixup_image = mixup_image.astype(np.uint8)
    
    cls_labels = new_target["labels"].copy()
    box_labels = cp_bboxes_transformed_np

    mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
    mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)
    mixup_target = {
        "boxes": mixup_bboxes,
        "labels": mixup_labels,
        'orig_size': mixup_image.shape[:2]
    }

    return mixup_image, mixup_target
    

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# Convert ndarray to tensor
class ToTensor(object):
    def __init__(self, format='RGB'):
        self.format = format

    def __call__(self, image, target=None):
        # check color format
        if self.format == 'RGB':
            # BGR -> RGB
            image = image[..., (2, 1, 0)]
            # [H, W, C] -> [C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        elif self.format == 'BGR':
            # keep BGR format
            image = image
            # [H, W, C] -> [C, H, W]
            image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        else:
            print('Unknown color format !!')
            exit()
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()

        return image, target


# DistortTransform
class DistortTransform(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, hue=0.1, saturation=1.5, exposure=1.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def __call__(self, image: np.ndarray, target=None) -> np.ndarray:
        """
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        dhue = np.random.uniform(low=-self.hue, high=self.hue)
        dsat = self._rand_scale(self.saturation)
        dexp = self._rand_scale(self.exposure)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = np.asarray(image, dtype=np.float32) / 255.
        image[:, :, 1] *= dsat
        image[:, :, 2] *= dexp
        H = image[:, :, 0] + dhue * 179 / 255.

        if dhue > 0:
            H[H > 1.0] -= 1.0
        else:
            H[H < 0.0] += 1.0

        image[:, :, 0] = H
        image = (image * 255).clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = np.asarray(image, dtype=np.uint8)

        return image, target

    def _rand_scale(self, upper_bound):
        """
        Calculate random scaling factor.

        Args:
            upper_bound (float): range of the random scale.
        Returns:
            random scaling factor (float) whose range is
            from 1 / s to s .
        """
        scale = np.random.uniform(low=1, high=upper_bound)
        if np.random.rand() > 0.5:
            return scale
        return 1 / scale


# JitterCrop
class JitterCrop(object):
    """Jitter and crop the image and box."""

    def __init__(self, jitter_ratio):
        super().__init__()
        self.jitter_ratio = jitter_ratio

    def crop(self, image, pleft, pright, ptop, pbot, output_size):
        oh, ow = image.shape[:2]

        swidth, sheight = output_size

        src_rect = [pleft, ptop, swidth + pleft,
                    sheight + ptop]  # x1,y1,x2,y2
        img_rect = [0, 0, ow, oh]
        # rect intersection
        new_src_rect = [max(src_rect[0], img_rect[0]),
                        max(src_rect[1], img_rect[1]),
                        min(src_rect[2], img_rect[2]),
                        min(src_rect[3], img_rect[3])]
        dst_rect = [max(0, -pleft),
                    max(0, -ptop),
                    max(0, -pleft) + new_src_rect[2] - new_src_rect[0],
                    max(0, -ptop) + new_src_rect[3] - new_src_rect[1]]

        # crop the image
        cropped = np.zeros([sheight, swidth, 3], dtype=image.dtype)
        cropped[:, :, ] = np.mean(image, axis=(0, 1))
        cropped[dst_rect[1]:dst_rect[3], dst_rect[0]:dst_rect[2]] = \
            image[new_src_rect[1]:new_src_rect[3],
            new_src_rect[0]:new_src_rect[2]]

        return cropped


    def __call__(self, image, target=None):
        oh, ow = image.shape[:2]
        dw = int(ow * self.jitter_ratio)
        dh = int(oh * self.jitter_ratio)
        pleft = np.random.randint(-dw, dw)
        pright = np.random.randint(-dw, dw)
        ptop = np.random.randint(-dh, dh)
        pbot = np.random.randint(-dh, dh)

        swidth = ow - pleft - pright
        sheight = oh - ptop - pbot
        output_size = (swidth, sheight)
        # crop image
        cropped_image = self.crop(image=image,
                                  pleft=pleft, 
                                  pright=pright, 
                                  ptop=ptop, 
                                  pbot=pbot,
                                  output_size=output_size)
        # crop bbox
        if target is not None:
            bboxes = target['boxes'].copy()
            coords_offset = np.array([pleft, ptop], dtype=np.float32)
            bboxes[..., [0, 2]] = bboxes[..., [0, 2]] - coords_offset[0]
            bboxes[..., [1, 3]] = bboxes[..., [1, 3]] - coords_offset[1]
            swidth, sheight = output_size

            bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], 0, swidth - 1)
            bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], 0, sheight - 1)
            target['boxes'] = bboxes

        return cropped_image, target


# RandomHFlip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = image[:, ::-1]
            if target is not None:
                h, w = target["orig_size"]
                if "boxes" in target:
                    boxes = target["boxes"].copy()
                    boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target


# Resize tensor image
class Resize(object):
    def __init__(self, img_size=640):
        self.img_size = img_size

    def __call__(self, image, target=None):
        # Resize the longest side of the image to the specified max size
        img_h0, img_w0 = image.shape[1:]

        r = self.img_size / max(img_h0, img_w0)
        if r != 1: 
            image = F.resize(image, (int(img_h0 * r), int(img_w0 * r)))

        # rescale bboxes
        if target is not None:
            img_h, img_w = image.shape[1:]
            # rescale bbox
            boxes_ = target["boxes"].clone()
            boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
            boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
            target["boxes"] = boxes_

        return image, target


# Pad tensor image
class PadImage(object):
    def __init__(self, img_size=640, adaptive=False) -> None:
        self.img_size = img_size
        self.adapative = adaptive

    def __call__(self, image, target=None):
        img_h0, img_w0 = image.shape[1:]
        assert max(img_h0, img_w0) <= self.img_size

        if self.adapative:
            if img_h0 > img_w0:
                pad_img_h = self.img_size
                pad_img_w = (img_w0 // 32 + 1) * 32
            elif img_h0 < img_w0:
                pad_img_h = (img_h0 // 32 + 1) * 32
                pad_img_w = self.img_size
            else:
                pad_img_h = self.img_size
                pad_img_w = self.img_size
            pad_image = torch.zeros([image.size(0), pad_img_h, pad_img_w]).float()
        else:
            pad_image = torch.zeros([image.size(0), self.img_size, self.img_size]).float()
        pad_image[:, :img_h0, :img_w0] = image

        return pad_image, target


# TrainTransform
class TrainTransforms(object):
    def __init__(self, 
                 trans_config=None,
                 img_size=640,
                 format='RGB'):
        self.trans_config = trans_config
        self.img_size = img_size
        self.format = format
        self.transforms = Compose(self.build_transforms(trans_config))


    def build_transforms(self, trans_config):
        transform = []
        for t in trans_config:
            if t['name'] == 'DistortTransform':
                transform.append(DistortTransform(hue=t['hue'], 
                                                  saturation=t['saturation'], 
                                                  exposure=t['exposure']))
            elif t['name'] == 'RandomHorizontalFlip':
                transform.append(RandomHorizontalFlip())
            elif t['name'] == 'JitterCrop':
                transform.append(JitterCrop(jitter_ratio=t['jitter_ratio']))
            elif t['name'] == 'ToTensor':
                transform.append(ToTensor(format=self.format))
            elif t['name'] == 'Resize':
                transform.append(Resize(img_size=self.img_size))
            elif t['name'] == 'PadImage':
                transform.append(PadImage(img_size=self.img_size))
        
        return transform


    def __call__(self, image, target):
        return self.transforms(image, target)


# ValTransform
class ValTransforms(object):
    def __init__(self, 
                 img_size=640,
                 format='RGB'):
        self.img_size = img_size
        self.format = format
        self.transforms = Compose([
            ToTensor(),
            Resize(img_size=img_size),
            PadImage(img_size=img_size, adaptive=True)
        ])


    def __call__(self, image, target=None):
        return self.transforms(image, target)
