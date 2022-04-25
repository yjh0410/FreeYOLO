import torch
import torch.nn as nn
import torch.nn.functional as F
from .matcher import Matcher, OTA, SimOTA
from utils.box_ops import get_ious
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 matcher,
                 loss_obj_weight=1.0, 
                 loss_cls_weight=1.0,
                 loss_reg_weight=1.0,
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.matcher_name = matcher
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        # loss
        self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')
        # matcher
        matcher_cfg = cfg['matcher'][matcher]
        if matcher == 'basic':
            self.matcher = Matcher(
                num_classes=num_classes,
                center_sampling_radius=matcher_cfg['center_sampling_radius'],
                object_sizes_of_interest=matcher_cfg['object_sizes_of_interest'])
        elif matcher == 'ota':
            self.matcher = OTA(
                num_classes, 
                center_sampling_radius=matcher_cfg['center_sampling_radius'],
                topk_candidate=matcher_cfg['topk_candidate'])
        elif matcher == 'sim_ota':
            self.matcher = SimOTA(
                num_classes, 
                center_sampling_radius=matcher_cfg['center_sampling_radius'],
                topk_candidate=matcher_cfg['topk_candidate'])


    def basic_loss(self, outputs, targets, anchors=None):
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
        (
            gt_objectness, 
            gt_classes, 
            gt_shifts_deltas
            ) = self.matcher(fpn_strides = fpn_strides,
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
    

    def ota_loss(self, outputs, targets, anchors=None):
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
        (
            gt_objectness, 
            gt_classes, 
            gt_shifts_deltas
            ) = self.matcher(fpn_strides = fpn_strides,
                             anchors = anchors,
                             obj_preds = outputs['pred_obj'],
                             cls_preds = outputs['pred_cls'],
                             reg_preds = outputs['pred_reg'],
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
    

    def sim_ota_loss(self, outputs, targets, anchors=None):
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
        bs = outputs['pred_cls'][0].shape[0]

        # List[F, Mi, 2] -> [M, 2] -> [B, M, 2]
        anchors_ = torch.cat(anchors, dim=0).unsqueeze(0).repeat(bs, 1, 1)
        num_anchors = anchors_.shape[1]

        # List[B, Mi, C] -> [B, M, C]
        obj_preds = torch.cat(outputs['pred_obj'], dim=1)
        cls_preds = torch.cat(outputs['pred_cls'], dim=1)
        reg_preds = torch.cat(outputs['pred_reg'], dim=1)
        box_x1y1_preds = anchors_ - reg_preds[..., :2]
        box_x2y2_preds = anchors_ + reg_preds[..., 2:]
        box_preds = torch.cat([box_x1y1_preds, box_x2y2_preds], dim=-1)
        del anchors_

        cls_targets = []
        reg_targets = []
        obj_targets = []
        fg_masks = []

        num_foregrounds = 0.0
        for batch_idx in range(bs):
            tgt_cls_per_image = targets[batch_idx]["labels"].to(device)
            tgt_box_per_image = targets[batch_idx]["boxes"].to(device)
            num_gt = len(tgt_cls_per_image)
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((num_anchors, 1))
                fg_mask = outputs.new_zeros(num_anchors).bool()
            else:
                (
                    gt_matched_classes,
                    fg_mask,
                    pred_ious_this_matching,
                    matched_gt_inds,
                    num_fg_img,
                ) = self.matcher(fpn_strides = fpn_strides,
                                 anchors = anchors,
                                 pred_obj_per_image = obj_preds[batch_idx], 
                                 pred_cls_per_image = cls_preds[batch_idx], 
                                 pred_box_per_image = box_preds[batch_idx],
                                 tgt_cls_per_image = tgt_cls_per_image,
                                 tgt_box_per_image = tgt_box_per_image)
                num_foregrounds += num_fg_img

                obj_target = fg_mask.unsqueeze(-1)
                cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)
                cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)
                reg_target = tgt_box_per_image[matched_gt_inds]

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
        loss_labels = self.obj_lossf(matched_cls_preds, cls_targets)
        loss_labels = loss_labels.sum() / num_foregrounds

        # regression loss
        matched_box_preds = box_preds.view(-1, 4)[fg_masks]
        ious = get_ious(matched_box_preds,
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
        if self.matcher_name == 'basic':
            return self.basic_loss(outputs, targets, anchors)
        elif self.matcher_name == 'ota':
            return self.ota_loss(outputs, targets, anchors)
        elif self.matcher_name == 'sim_ota':
            return self.sim_ota_loss(outputs, targets, anchors)


if __name__ == "__main__":
    pass
