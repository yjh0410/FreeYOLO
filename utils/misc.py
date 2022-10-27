import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import os
import math
from copy import deepcopy

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from dataset.voc import VOCDetection
from dataset.coco import COCODataset
from dataset.transforms import BaseTransforms, TrainTransforms, ValTransforms


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms']
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = TrainTransforms(trans_config=trans_config,
                                      img_size=cfg['train_size'],
                                      format=cfg['format'])
    val_transform = ValTransforms(img_size=cfg['test_size'],
                                  format=cfg['format'])
    color_augment = BaseTransforms(
        img_size=cfg['train_size'],
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        format=cfg['format'],
        min_box_size=args.min_box_size
    )
    train_transform = TrainTransforms(
        trans_config=trans_config,
        img_size=cfg['train_size'],
        format=cfg['format'],
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std'],
        min_box_size=args.min_box_size
        )
    val_transform = ValTransforms(
        img_size=cfg['test_size'],
        format=cfg['format'],
        pixel_mean=cfg['pixel_mean'],
        pixel_std=cfg['pixel_std']
        )
        
    # dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20

        # dataset
        dataset = VOCDetection(
            img_size=cfg['train_size'],
            data_dir=data_dir,
            transform=train_transform,
            color_augment=color_augment,
            mosaic_prob=cfg['mosaic_prob'],
            mixup_prob=cfg['mixup_prob']
            )
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        # dataset
        dataset = COCODataset(
            img_size=cfg['train_size'],
            data_dir=data_dir,
            image_set='train2017',
            transform=train_transform,
            color_augment=color_augment,
            mosaic_prob=cfg['mosaic_prob'],
            mixup_prob=cfg['mixup_prob']
            )
        # evaluator
        evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                     device=device,
                                     transform=val_transform)

    else:
        print('unknow dataset !! Only support voc, coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=True)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler_train,
                            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    
    return dataloader
    

def load_weight(model, path_to_ckpt):
    # check ckpt file
    if path_to_ckpt is None:
        print('no weight file ...')

        return model

    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    print('Finished loading model!')

    return model


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def replace_module(module, replaced_module_type, new_module_type, replace_func=None) -> nn.Module:
    """
    Replace given type in module to a new type. mostly used in deploy.

    Args:
        module (nn.Module): model to apply replace operation.
        replaced_module_type (Type): module type to be replaced.
        new_module_type (Type)
        replace_func (function): python function to describe replace logic. Defalut value None.

    Returns:
        model (nn.Module): module that already been replaced.
    """

    def default_replace_func(replaced_module_type, new_module_type):
        return new_module_type()

    if replace_func is None:
        replace_func = default_replace_func

    model = module
    if isinstance(module, replaced_module_type):
        model = replace_func(replaced_module_type, new_module_type)
    else:  # recurrsively replace
        for name, child in module.named_children():
            new_child = replace_module(child, replaced_module_type, new_module_type)
            if new_child is not child:  # child is already replaced
                model.add_module(name, new_child)

    return model


def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-10, xx2 - xx1)
        h = np.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


# Model EMA
class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


class CollateFunc(object):
    def __call__(self, batch):
        targets = []
        images = []

        for sample in batch:
            image = sample[0]
            target = sample[1]

            images.append(image)
            targets.append(target)

        images = torch.stack(images, 0) # [B, C, H, W]

        return images, targets


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)

        # Sinkhorn iterations
        for i in range(self.max_iter):
            v = self.eps * \
                (torch.log(
                    nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * \
                (torch.log(
                    mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


# test time augmentation(TTA)
class TestTimeAugmentation(object):
    def __init__(self,
                 num_classes=80, nms_thresh=0.4,
                 min_size = 608, max_size = 608,
                 test_flip=False):
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.scales = np.arange(min_size, max_size+32, 32)
        self.test_flip = test_flip
        

    def nms(self, dets, scores, nms_thresh=0.4):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def __call__(self, x, model):
        # x: Tensor -> [B, C, H, W]
        bboxes_list = []
        scores_list = []
        labels_list = []

        # multi scale
        for s in self.scales:
            h0, w0 = x.shape[-2:]
            if max(h0, w0) == s:
                x_scale = x
            else:
                # keep aspect ratio
                r = s / max(h0, w0)
                w1 = int(w0 * r / 32.) * 32
                h1 = int(h0 * r / 32.) * 32
                x_scale = F.interpolate(
                    input=x,
                    size=(h1, w1),
                    mode='bilinear',
                    align_corners=False)

            bboxes, scores, labels = model(x_scale)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

            # Flip
            if self.test_flip:
                x_flip = torch.flip(x_scale, [-1])
                bboxes, scores, labels = model(x_flip)
                bboxes = bboxes.copy()
                bboxes[:, 0::2] = 1.0 - bboxes[:, 2::-2]
                bboxes_list.append(bboxes)
                scores_list.append(scores)
                labels_list.append(labels)

        bboxes = np.concatenate(bboxes_list)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        # rescale
        bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]] * w0, a_min=0., a_max=w0)
        bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]] * h0, a_min=0., a_max=h0)

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # scale
        bboxes[..., [0, 2]] /= w0
        bboxes[..., [1, 3]] /= h0

        return bboxes, scores, labels