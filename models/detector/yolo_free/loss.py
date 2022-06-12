import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import Matcher
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 loss_obj_weight=1.0, 
                 loss_cls_weight=1.0,
                 loss_reg_weight=1.0,
                 num_classes=80,
                 matcher='fcos'):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        # loss
        self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')
        # matcher
        matcher_config = cfg['matcher']
        self.matcher = Matcher(
            num_classes=num_classes,
            center_sampling_radius=matcher_config['center_sampling_radius'],
            object_sizes_of_interest=matcher_config['object_sizes_of_interest']
            )


    def __call__(self, outputs, targets):
        """
            outputs['pred_obj']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_reg']: List(Tensor) [B, M, 4]
            outputs['anchors']:  List(Tensor) [M, 2]
            outputs['strides']: List(Int) [8, 16, 32] output strides
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        # label assignment
        gt_objectness, gt_classes, gt_deltas = self.matcher(
            fpn_strides = fpn_strides,
            anchors = anchors,
            targets = targets
            )

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = torch.cat(outputs['pred_obj'], dim=1).view(-1)
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        
        gt_objectness = gt_objectness.flatten().to(device).float()
        gt_classes = gt_classes.flatten().to(device).long()
        gt_deltas = gt_deltas.view(-1, 4).to(device).float()

        foreground_idxs = (gt_objectness > 0)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        gt_classes_target = torch.zeros_like(pred_cls)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # TO DO:
        # objectness loss
        loss_objectness = self.obj_lossf(pred_obj, gt_objectness)
        loss_objectness = loss_objectness.sum() / num_foreground

        # regression loss
        matched_pred_delta = pred_delta[foreground_idxs]
        matched_tgt_delta = gt_deltas[foreground_idxs]
        ious = get_ious(matched_pred_delta,
                        matched_tgt_delta,
                        box_mode="ltrb",
                        iou_type='giou')
        loss_bboxes = (1.0 - ious).sum() / num_foreground

        # classification loss
        matched_pred_cls = pred_cls[foreground_idxs]
        matched_tgt_cls = gt_classes_target[foreground_idxs] * ious.unsqueeze(1).clamp(0.)
        loss_labels = self.cls_lossf(matched_pred_cls, matched_tgt_cls)
        loss_labels = loss_labels.sum() / num_foreground

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
    

if __name__ == "__main__":
    pass
