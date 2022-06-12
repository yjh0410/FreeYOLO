import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import Matcher, SimOTA
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
        print('==============================')
        print('Matcher: {}'.format(matcher))
        if matcher == 'fcos':
            matcher_config = cfg['matcher']['fcos']
            self.matcher = Matcher(
                num_classes=num_classes,
                center_sampling_radius=matcher_config['center_sampling_radius'],
                object_sizes_of_interest=matcher_config['object_sizes_of_interest']
                )
        elif matcher == 'simota':
            matcher_config = cfg['matcher']['simota']
            self.matcher = SimOTA(
                num_classes=num_classes,
                center_sampling_radius=matcher_config['center_sampling_radius'],
                topk_candidate=matcher_config['topk_candidate']
                )


    def basic_loss(self, outputs, targets):
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


    def simota_loss(self, outputs, targets):        
        """
            outputs['pred_obj']: List(Tensor) [B, M, 1]
            outputs['pred_cls']: List(Tensor) [B, M, C]
            outputs['pred_reg']: List(Tensor) [B, M, 4]
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
        reg_preds = torch.cat(outputs['pred_reg'], dim=1)

        # label assignment
        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_foregrounds = 0.0
        for batch_idx in range(bs):
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:
                num_anchors = sum([ab.shape[0] for ab in anchors])
                # There is no valid gt
                cls_target = obj_preds.new_zeros((0, self.num_classes))
                reg_target = obj_preds.new_zeros((0, 4))
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
                    pred_reg = reg_preds[batch_idx],
                    tgt_labels = tgt_labels,
                    tgt_bboxes = tgt_bboxes
                    )
                num_foregrounds += num_fg_img

                obj_target = fg_mask.unsqueeze(-1)
                cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)
                reg_target = tgt_bboxes[matched_gt_inds]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)
        num_foregrounds = max(num_foregrounds / get_world_size(), 1)

        # objectness loss
        loss_objectness = self.obj_lossf(obj_preds.view(-1, 1), obj_targets.float())
        loss_objectness = loss_objectness.sum() / num_foregrounds
        
        # classification loss
        matched_cls_preds = cls_preds.view(-1, self.num_classes)[fg_masks]
        loss_labels = self.cls_lossf(matched_cls_preds, cls_targets)
        loss_labels = loss_labels.sum() / num_foregrounds

        # regression loss
        matched_reg_preds = reg_preds.view(-1, 4)[fg_masks]
        ious = get_ious(matched_reg_preds,
                        reg_targets,
                        box_mode="xyxy",
                        iou_type='giou')
        loss_bboxes = (1.0 - ious).sum() / num_foregrounds

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
        if self.cfg['label_assignment'] == 'fcos':
            return self.basic_loss(outputs, targets)
        elif self.cfg['label_assignment'] == 'simota':
            return self.simota_loss(outputs, targets)
    

if __name__ == "__main__":
    pass
