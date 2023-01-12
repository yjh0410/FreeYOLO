import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

import os
import math
from copy import deepcopy

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator
from evaluator.crowdhuman_evaluator import CrowdHumanEvaluator
from evaluator.widerface_evaluator import WiderFaceEvaluator
from evaluator.mot_evaluator import MOTEvaluator
from evaluator.ourdataset_evaluator import OurDatasetEvaluator

from dataset.voc import VOCDetection, VOC_CLASSES
from dataset.coco import COCODataset, coco_class_index, coco_class_labels
from dataset.widerface import WiderFaceDataset, widerface_class_labels
from dataset.crowdhuman import CrowdHumanDataset, crowd_class_labels
from dataset.mot17 import MOT17Dataset, mot_class_labels
from dataset.mot20 import MOT20Dataset, mot_class_labels
from dataset.ourdataset import OurDataset, our_class_labels

from dataset.transforms import TrainTransforms, ValTransforms


def build_dataset(cfg, args, device, is_train=False):
    # transform
    print('==============================')
    print('TrainTransforms: {}'.format(cfg['trans_config']))
    train_transform = TrainTransforms(
        trans_config=cfg['trans_config'],
        img_size=cfg['train_size'],
        min_box_size=args.min_box_size
        )
    val_transform = ValTransforms(img_size=cfg['test_size'])
    
    # dataset params
    transform = train_transform if is_train else None
    trans_config=cfg['trans_config'] if is_train else None

    # mosaic prob.
    if args.mosaic is not None:
        mosaic_prob=args.mosaic if is_train else 0.0
    else:
        mosaic_prob=cfg['mosaic_prob'] if is_train else 0.0

    # mixup prob.
    if args.mixup is not None:
        mixup_prob=args.mixup if is_train else 0.0
    else:
        mixup_prob=cfg['mixup_prob']  if is_train else 0.0

    # dataset
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        class_names = VOC_CLASSES
        class_indexs = None

        # dataset
        dataset = VOCDetection(
            img_size=cfg['train_size'],
            data_dir=data_dir,
            image_sets=[('2007', 'trainval'), ('2012', 'trainval')] if is_train else [('2007', 'test')],
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = VOCAPIEvaluator(
                data_dir=data_dir,
                device=device,
                transform=val_transform)
        else:
            evaluator = None

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        class_names = coco_class_labels
        class_indexs = coco_class_index

        # dataset
        dataset = COCODataset(
            img_size=cfg['train_size'],
            data_dir=data_dir,
            image_set='train2017' if is_train else 'val2017',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = COCOAPIEvaluator(
                data_dir=data_dir,
                device=device,
                transform=val_transform
                )
        else:
            evaluator = None

    elif args.dataset == 'widerface':
        data_dir = os.path.join(args.root, 'WiderFace')
        num_classes = 1
        class_names = widerface_class_labels
        class_indexs = None

        # dataset
        dataset = WiderFaceDataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train' if is_train else 'val',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = WiderFaceEvaluator(
                data_dir=data_dir,
                device=device,
                image_set='val',
                transform=val_transform
            )
        else:
            evaluator = None

    elif args.dataset == 'crowdhuman':
        data_dir = os.path.join(args.root, 'CrowdHuman')
        num_classes = 1
        class_names = crowd_class_labels
        class_indexs = None

        # dataset
        dataset = CrowdHumanDataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train' if is_train else 'val',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = CrowdHumanEvaluator(
                data_dir=data_dir,
                device=device,
                image_set='val',
                transform=val_transform
            )
        else:
            evaluator = None

    elif args.dataset == 'mot17_half':
        data_dir = os.path.join(args.root, 'MOT17')
        num_classes = 1
        class_names = mot_class_labels
        class_indexs = None

        # dataset
        dataset = MOT17Dataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train',
            json_file='train_half.json' if is_train else 'val_half.json',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = MOTEvaluator(
                data_dir=data_dir,
                device=device,
                dataset='mot17',
                transform=val_transform
                )
        else:
            evaluator = None

    elif args.dataset == 'mot17':
        data_dir = os.path.join(args.root, 'MOT17')
        num_classes = 1
        class_names = mot_class_labels
        class_indexs = None

        # dataset
        dataset = MOT17Dataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train',
            json_file='train.json',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        evaluator = None

    elif args.dataset == 'mot17_test':
        data_dir = os.path.join(args.root, 'MOT17')
        num_classes = 1
        class_names = mot_class_labels
        class_indexs = None

        # dataset
        dataset = MOT17Dataset(
                data_dir=data_dir,
                image_set='test',
                json_file='test.json',
                transform=None)
        # evaluator
        evaluator = None

    elif args.dataset == 'mot20_half':
        data_dir = os.path.join(args.root, 'MOT20')
        num_classes = 1
        class_names = mot_class_labels
        class_indexs = None

        # dataset
        dataset = MOT20Dataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train',
            json_file='train_half.json' if is_train else 'val_half.json',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = MOTEvaluator(
                data_dir=data_dir,
                device=device,
                dataset='mot20',
                transform=val_transform
                )
        else:
            evaluator = None

    elif args.dataset == 'mot20':
        data_dir = os.path.join(args.root, 'MOT20')
        num_classes = 1
        class_names = mot_class_labels
        class_indexs = None

        # dataset
        dataset = MOT20Dataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train',
            json_file='train.json',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        evaluator = None

    elif args.dataset == 'mot20_test':
        data_dir = os.path.join(args.root, 'MOT20')
        num_classes = 1
        class_names = mot_class_labels
        class_indexs = None

        # dataset
        dataset = MOT20Dataset(
                data_dir=data_dir,
                image_set='test',
                json_file='test.json',
                transform=None)
        # evaluator
        evaluator = None

    elif args.dataset == 'ourdataset':
        data_dir = os.path.join(args.root, 'OurDataset')
        class_names = our_class_labels
        num_classes = len(our_class_labels)
        class_indexs = None

        # dataset
        dataset = OurDataset(
            data_dir=data_dir,
            img_size=cfg['train_size'],
            image_set='train' if is_train else 'val',
            transform=transform,
            mosaic_prob=mosaic_prob,
            mixup_prob=mixup_prob,
            trans_config=trans_config
            )
        # evaluator
        if is_train:
            evaluator = OurDatasetEvaluator(
                data_dir=data_dir,
                device=device,
                image_set='val',
                transform=val_transform
            )
        else:
            evaluator = None

    else:
        print('unknow dataset !!')
        exit(0)

    print('==============================')
    print('Dataset name: {}'.format(args.dataset))
    print('Dataset size: {}'.format(len(dataset)))

    return dataset, (num_classes, class_names, class_indexs), evaluator


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
    checkpoint_state_dict = checkpoint.pop("model")
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


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='none'):
    p = torch.sigmoid(logits)
    # bce loss
    ce_loss = F.binary_cross_entropy_with_logits(
        input=logits, target=targets, reduction="none")
    # focal weight
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    # focal loss
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


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
