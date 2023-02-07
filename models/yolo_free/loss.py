import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
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
        matcher_config = cfg['matcher']
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=matcher_config['center_sampling_radius'],
            topk_candidate=matcher_config['topk_candicate']
            )


    def __call__(self, outputs, targets):        
        """
            outputs['pred_obj']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        # preds: [B, M, C]
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)

        # label assignment
        cls_targets = []
        box_targets = []
        obj_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                # There is no valid gt
                cls_target = obj_preds.new_zeros((0, self.num_classes))
                box_target = obj_preds.new_zeros((0, 4))
                obj_target = obj_preds.new_zeros((num_anchors, 1))
                fg_mask = obj_preds.new_zeros(num_anchors).bool()
            else:
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.matcher(
                    fpn_strides = fpn_strides,
                    anchors = anchors,
                    pred_obj = obj_preds[batch_idx],
                    pred_cls = cls_preds[batch_idx], 
                    pred_box = box_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )

                obj_target = fg_mask.unsqueeze(-1)
                cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)
                box_target = tgt_bboxes[matched_gt_inds]

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)

        # objectness loss
        loss_obj = self.obj_lossf(obj_preds.view(-1, 1), obj_targets.float())
        loss_obj = loss_obj.sum() / num_foregrounds
        
        # classification loss
        matched_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.cls_lossf(matched_cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / num_foregrounds

        # regression loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        ious = get_ious(matched_box_preds,
                        box_targets,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_box = (1.0 - ious).sum() / num_foregrounds

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_reg_weight * loss_box

        loss_dict = dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
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