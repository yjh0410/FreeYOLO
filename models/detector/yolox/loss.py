import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import Matcher, OTA_Matcher
from utils.box_ops import *
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 loss_obj_weight=1.0, 
                 loss_cls_weight=1.0,
                 loss_reg_weight=1.0,
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        if cfg['matcher'] == 'matcher':
            self.matcher = Matcher(cfg,
                                   num_classes=num_classes,
                                   box_weights=[1., 1., 1., 1.])
        elif cfg['matcher'] == 'ota_matcher':
            self.matcher = OTA_Matcher(cfg, 
                                       num_classes, 
                                       box_weights=[1., 1., 1., 1.])


    # The origin loss of FCOS
    def __call__(self, outputs, targets, anchors=None):
        """
            outputs['pred_obj']: (Tensor) [B, M, 1]
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchors: (List of Tensor) List[Tensor[M, 4]], len(anchors) == num_fpn_levels
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        gt_objectness, gt_classes, gt_shifts_deltas = self.matcher(fpn_strides = fpn_strides,
                                                                    anchors = anchors,
                                                                    targets = targets)

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = torch.cat(outputs['pred_obj'], dim=1).view(-1, 1)
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)

        gt_classes = gt_classes.flatten().to(device)
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4).to(device)
        gt_centerness = gt_centerness.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # TO DO:
        # regression loss
        ious = None
        loss_bboxes = None

        # objectness loss
        loss_objectness = None

        # classification loss
        loss_labels = None

        # total loss
        losses = self.loss_obj_weight * loss_objectness + \
                 self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes

        loss_dict = dict(
                loss_objectness = loss_objectness,
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                losses = losses
        )

        return loss_dict
    

    # Compute loss of FCOS with OTA matcher
    def ota_losses(self,
                 outputs, 
                 targets, 
                 anchors=None):
        """
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_reg']: (Tensor) [B, M, 4]
            outputs['pred_ctn']: (Tensor) [B, M, 1]
            outputs['strides']: (List) [8, 16, 32, ...] stride of the model output
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
            anchors: (List of Tensor) List[Tensor[M, 4]], len(anchors) == num_fpn_levels
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        gt_classes, gt_shifts_deltas, gt_ious = self.matcher(fpn_strides = fpn_strides, 
                                                             anchors = anchors, 
                                                             pred_cls_logits = outputs['pred_cls'], 
                                                             pred_deltas = outputs['pred_reg'], 
                                                             targets = targets)

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        pred_iou = torch.cat(outputs['pred_ctn'], dim=1).view(-1, 1)
        masks = torch.cat(outputs['mask'], dim=1).view(-1)

        gt_classes = gt_classes.flatten().to(device)
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4).to(device)
        gt_ious = gt_ious.view(-1, 1).to(device)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # cls loss
        valid_idxs = (gt_classes >= 0) & masks
        loss_labels = self.loss_labels(
            pred_cls[valid_idxs],
            gt_classes_target[valid_idxs],
            num_boxes=num_foreground)

        # box loss
        loss_bboxes = self.loss_bboxes(
            pred_delta[foreground_idxs],
            gt_shifts_deltas[foreground_idxs],
            num_boxes=num_foreground)

        # iou loss
        loss_ious = F.binary_cross_entropy_with_logits(pred_iou[foreground_idxs], 
                                                       gt_ious[foreground_idxs], 
                                                       reduction='none')
        loss_ious = loss_ious.sum() / num_foreground

        # total loss
        losses = self.loss_obj_weight * loss_labels + \
                 self.loss_cls_weight * loss_bboxes + \
                 self.loss_reg_weight * loss_ious

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                loss_ious = loss_ious,
                losses = losses
        )

        return loss_dict


    def __call__(self,
                 outputs, 
                 targets, 
                 anchors=None):
        if self.cfg['matcher'] == 'matcher':
            return self.basic_losses(outputs, targets, anchors)
        elif self.cfg['matcher'] == 'ota_matcher':
            return self.ota_losses(outputs, targets, anchors)

    
if __name__ == "__main__":
    pass
