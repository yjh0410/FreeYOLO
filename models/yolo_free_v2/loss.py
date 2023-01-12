import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import SimOTA
from utils.box_ops import get_ious
from utils.misc import sigmoid_focal_loss
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction


    def forward(self, logits, targets):
        return sigmoid_focal_loss(
            logits, targets, self.alpha, self.gamma, self.reduction)


class Criterion(object):
    def __init__(self, cfg, device, num_classes=80):
        self.cfg = cfg
        self.device = device
        self.num_classes = num_classes
        self.loss_iou_weight = cfg['loss_iou_weight']
        self.loss_cls_weight = cfg['loss_cls_weight']
        self.loss_reg_weight = cfg['loss_reg_weight']
        # loss
        self.cls_lossf = SigmoidFocalLoss(reduction='none')
        self.iou_lossf = nn.BCEWithLogitsLoss(reduction='none')
        # matcher
        matcher_config = cfg['matcher']
        self.matcher = SimOTA(
            num_classes=num_classes,
            center_sampling_radius=matcher_config['center_sampling_radius'],
            topk_candidate=matcher_config['topk_candicate']
            )


    def __call__(self, outputs, targets):        
        """
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_box']: List(Tensor) [B, M, 4]
            outputs['pred_iou']: List(Tensor) [B, M, 1]
            outputs['strides']: List(Int) [8, 16, 32] output stride
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]
        """
        bs = outputs['pred_cls'][0].shape[0]
        device = outputs['pred_cls'][0].device
        fpn_strides = outputs['strides']
        anchors = outputs['anchors']
        num_anchors = sum([ab.shape[0] for ab in anchors])
        # preds: [B, M, C]
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        box_preds = torch.cat(outputs['pred_box'], dim=1)
        iou_preds = torch.cat(outputs['pred_iou'], dim=1)

        # label assignment
        cls_targets = []
        box_targets = []
        iou_targets = []
        fg_masks = []

        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                # There is no valid gt
                cls_target = cls_preds.new_zeros((num_anchors, self.num_classes))
                box_target = cls_preds.new_zeros((0, 4))
                iou_target = cls_preds.new_zeros((0, 1))
                fg_mask = cls_preds.new_zeros(num_anchors).bool()
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
                    pred_cls = cls_preds[batch_idx], 
                    pred_box = box_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )
                # cls target: [M, C]
                gt_cls = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_preds.new_zeros((num_anchors, self.num_classes)).float()
                cls_target[fg_mask] = gt_cls.float()
                # box target: [Mp, 4]
                box_target = tgt_bboxes[matched_gt_inds]
                # iou target: [Mp,]
                iou_target = pred_ious_this_matching

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            iou_targets.append(iou_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        box_targets = torch.cat(box_targets, 0)
        iou_targets = torch.cat(iou_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)

        # classification loss
        cls_preds = cls_preds.view(-1, self.num_classes)
        loss_labels = self.cls_lossf(cls_preds, cls_targets)
        loss_labels = loss_labels.sum() / num_foregrounds

        # regression loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        ious = get_ious(matched_box_preds,
                        box_targets,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_bboxes = (1.0 - ious).sum() / num_foregrounds

        # objectness loss
        matched_iou_preds = iou_preds.view(-1)[fg_masks]
        loss_ious = self.iou_lossf(matched_iou_preds, iou_targets.float())
        loss_ious = loss_ious.sum() / num_foregrounds
        
        # total loss
        losses = self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes + \
                 self.loss_iou_weight * loss_ious

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                loss_ious = loss_ious,
                losses = losses
        )

        return loss_dict
    

def build_criterion(cfg, device, num_classes):
    criterion = Criterion(cfg=cfg, device=device, num_classes=num_classes)

    return criterion


if __name__ == "__main__":
    pass