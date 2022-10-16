import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import YoloMatcher
from utils.box_ops import get_ious
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
        # loss
        self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')
        # matcher
        self.matcher = YoloMatcher(
            num_classes=num_classes,
            num_anchors=cfg['num_anchors'],
            iou_thresh=cfg['matcher']['iou_thresh'],
            anchor_size=cfg['anchor_size']
            )


    def __call__(self, outputs, targets):
        """
            outputs['pred_obj']: (Tensor) [B, M, 1]
            outputs['pred_cls']: (Tensor) [B, M, C]
            outputs['pred_box']: (Tensor) [B, M, 4]
            outputs['fmp_sizes']: (List) [[H, W], ...]
            outputs['strides']: (List) [8, 16, 32, ...]
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        fmp_sizes = outputs['fmp_sizes']
        (
            gt_objectness, 
            gt_classes, 
            gt_bboxes
            ) = self.matcher(fmp_sizes=fmp_sizes, 
                             fpn_strides=fpn_strides, 
                             targets=targets)
        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = torch.cat(outputs['pred_obj'], dim=1).view(-1)
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_box = torch.cat(outputs['pred_box'], dim=1).view(-1, 4)
        
        gt_objectness = gt_objectness.flatten().to(device).float()
        gt_classes = gt_classes.view(-1, self.num_classes).to(device).float()
        gt_bboxes = gt_bboxes.view(-1, 4).to(device).float()

        foreground_idxs = (gt_objectness > 0)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # objectness loss
        loss_objectness = self.obj_lossf(pred_obj, gt_objectness)
        loss_objectness = loss_objectness.sum() / num_foreground

        # regression loss
        matched_pred_box = pred_box[foreground_idxs]
        matched_tgt_box = gt_bboxes[foreground_idxs]
        ious = get_ious(matched_pred_box,
                        matched_tgt_box,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_bboxes = (1.0 - ious).sum() / num_foreground

        # classification loss
        matched_pred_cls = pred_cls[foreground_idxs]
        matched_tgt_cls = gt_classes[foreground_idxs] * ious.unsqueeze(1).clamp(0.)
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
    

def build_criterion(cfg, device, num_classes):
    criterion = Criterion(
        cfg=cfg,
        device=device,
        loss_obj_weight=cfg['loss_obj_weight'],
        loss_cls_weight=cfg['loss_cls_weight'],
        loss_reg_weight=cfg['loss_reg_weight'],
        num_classes=num_classes
        )

    return criterion

    
if __name__ == "__main__":
    pass
