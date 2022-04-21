import torch
import torch.nn as nn
from .matcher import Matcher, SimOTA_Matcher
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
        if cfg['matcher'] == 'matcher':
            self.matcher = Matcher(cfg,
                                   num_classes=num_classes,
                                   box_weights=[1., 1., 1., 1.])
        elif cfg['matcher'] == 'ota_matcher':
            self.matcher = SimOTA_Matcher(cfg, 
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
        pred_obj = torch.cat(outputs['pred_obj'], dim=1).view(-1)
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)
        pred_delta = torch.cat(outputs['pred_reg'], dim=1).view(-1, 4)
        
        gt_objectness = gt_objectness.flatten().to(device).float()
        gt_classes = gt_classes.flatten().to(device).long()
        gt_shifts_deltas = gt_shifts_deltas.view(-1, 4).to(device).float()

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
        matched_tgt_delta = gt_shifts_deltas[foreground_idxs]
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
