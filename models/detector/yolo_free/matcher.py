import math
import torch
import torch.nn.functional as F
from utils.box_ops import *


# FCOS Matcher
class Matcher(object):
    def __init__(self, 
                 num_classes,
                 center_sampling_radius,
                 object_sizes_of_interest):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.object_sizes_of_interest = object_sizes_of_interest


    def get_deltas(self, anchors, boxes):
        """
        Get box regression transformation deltas (dl, dt, dr, db) that can be used
        to transform the `anchors` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, anchors)`` is true.

        Args:
            anchors (Tensor): anchors, e.g., feature map coordinates
            boxes (Tensor): target of the transformation, e.g., ground-truth
                boxes.
        """
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(boxes, torch.Tensor), type(boxes)
        deltas = torch.cat((anchors - boxes[..., :2], 
                            boxes[..., 2:] - anchors), dim=-1)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, targets):
        """
            fpn_strides: (List) List[8, 16, 32, ...] stride of network output.
            anchors: (List of Tensor) List[F, M, 2], F = num_fpn_levels
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        gt_objectness = []
        gt_classes = []
        gt_anchors_deltas = []
        device = anchors[0].device

        # List[F, M, 2] -> [M, 2]
        anchors_over_all_feature_maps = torch.cat(anchors, dim=0).to(device)

        for targets_per_image in targets:
            # generate object_sizes_of_interest: List[[M, 2]]
            object_sizes_of_interest = [anchors_i.new_tensor(scale_range).unsqueeze(0).expand(anchors_i.size(0), -1) 
                                        for anchors_i, scale_range in zip(anchors, self.object_sizes_of_interest)]
            # List[F, M, 2] -> [M, 2], M = M1 + M2 + ... + MF
            object_sizes_of_interest = torch.cat(object_sizes_of_interest, dim=0)
            # [N, 4]
            tgt_box = targets_per_image['boxes'].to(device)
            # [N,]
            tgt_cls = targets_per_image['labels'].to(device)
            # [N,]
            tgt_obj = torch.ones_like(tgt_cls)
            # [N, M, 4], M = M1 + M2 + ... + MF
            deltas = self.get_deltas(anchors_over_all_feature_maps, tgt_box.unsqueeze(1))
            
            has_gt = True
            if tgt_box.shape[0] == 0:
                has_gt = False  # no gt
            elif tgt_box.max().item() == 0.:
                has_gt = False  # no valid bbox

            if has_gt:
                if self.center_sampling_radius > 0:
                    # bbox centers: [N, 2]
                    centers = (tgt_box[..., :2] + tgt_box[..., 2:]) * 0.5

                    is_in_boxes = []
                    for stride, anchors_i in zip(fpn_strides, anchors):
                        radius = stride * self.center_sampling_radius
                        # [N, 4]
                        center_boxes = torch.cat((
                            torch.max(centers - radius, tgt_box[:, :2]),
                            torch.min(centers + radius, tgt_box[:, 2:]),
                        ), dim=-1)
                        # [N, Mi, 4]
                        center_deltas = self.get_deltas(anchors_i, center_boxes.unsqueeze(1))
                        # [N, Mi]
                        is_in_boxes.append(center_deltas.min(dim=-1).values > 0)
                    # [N, M], M = M1 + M2 + ... + MF
                    is_in_boxes = torch.cat(is_in_boxes, dim=1)
                else:
                    # no center sampling, it will use all the locations within a ground-truth box
                    # [N, M], M = M1 + M2 + ... + MF
                    is_in_boxes = deltas.min(dim=-1).values > 0

                # [N, M], M = M1 + M2 + ... + MF
                max_deltas = deltas.max(dim=-1).values
                # limit the regression range for each location
                is_cared_in_the_level = \
                    (max_deltas >= object_sizes_of_interest[None, :, 0]) & \
                    (max_deltas <= object_sizes_of_interest[None, :, 1])

                # [N,]
                tgt_box_area = (tgt_box[:, 2] - tgt_box[:, 0]) * (tgt_box[:, 3] - tgt_box[:, 1])
                # [N,] -> [N, 1] -> [N, M]
                gt_positions_area = tgt_box_area.unsqueeze(1).repeat(
                    1, anchors_over_all_feature_maps.size(0))
                gt_positions_area[~is_in_boxes] = math.inf
                gt_positions_area[~is_cared_in_the_level] = math.inf

                # if there are still more than one objects for a position,
                # we choose the one with minimal area
                # [M,], each element is the index of ground-truth
                try:
                    positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)
                except:
                    print(gt_positions_area.shape, tgt_box.shape, tgt_box.shape[0])

                # ground truth objectness [M,]
                tgt_obj_i = tgt_obj[gt_matched_idxs]
                # anchors with area inf are treated as background.
                tgt_obj_i[positions_min_area == math.inf] = 0

                # ground truth classification [M,]
                tgt_cls_i = tgt_cls[gt_matched_idxs]
                # anchors with area inf are treated as background.
                tgt_cls_i[positions_min_area == math.inf] = self.num_classes

                # ground truth regression [M, 4]
                tgt_reg_i = self.get_deltas(anchors_over_all_feature_maps, tgt_box[gt_matched_idxs])

                gt_objectness.append(tgt_obj_i)
                gt_classes.append(tgt_cls_i)
                gt_anchors_deltas.append(tgt_reg_i)
                
            else:
                num_anchors = anchors_over_all_feature_maps.shape[0]
                tgt_obj_i = torch.zeros(num_anchors, device=device)
                tgt_cls_i = torch.zeros(num_anchors, device=device) + self.num_classes
                tgt_reg_i = torch.zeros([num_anchors, 4], device=device)

                gt_objectness.append(tgt_obj_i)
                gt_classes.append(tgt_cls_i)
                gt_anchors_deltas.append(tgt_reg_i)

        # [B, M], [B, M], [B, M, 4]
        return torch.stack(gt_objectness), torch.stack(gt_classes), torch.stack(gt_anchors_deltas)


# YOLOX SimOTA
class SimOTA(object):
    def __init__(self, 
                 num_classes,
                 center_sampling_radius,
                 topk_candidate
                 ) -> None:
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate


    @torch.no_grad()
    def __call__(self, 
                 fpn_strides, 
                 anchors, 
                 pred_obj, 
                 pred_cls, 
                 pred_reg, 
                 tgt_labels,
                 tgt_bboxes):
        # [M,]
        strides = torch.cat([torch.ones_like(anchor_i[:, 0]) * stride_i
                                for stride_i, anchor_i in zip(fpn_strides, anchors)], dim=-1)
        # List[F, M, 2] -> [M, 2]
        anchors = torch.cat(anchors, dim=0)
        num_anchor = anchors.shape[0]        
        num_gt = len(tgt_labels)

        fg_mask, is_in_boxes_and_center = \
            self.get_in_boxes_info(
                tgt_bboxes,
                anchors,
                strides,
                num_anchor,
                num_gt
                )

        obj_preds_ = pred_obj[fg_mask]   # [Mp, 1]
        cls_preds_ = pred_cls[fg_mask]   # [Mp, C]
        reg_preds_ = pred_reg[fg_mask]   # [Mp, 4]
        num_in_boxes_anchor = reg_preds_.shape[0]

        # [N, Mp]
        pair_wise_ious, _ = box_iou(tgt_bboxes, reg_preds_)
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        # [N, C] -> [N, Mp, C]
        gt_cls = (
            F.one_hot(tgt_labels.long(), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )

        with torch.cuda.amp.autocast(enabled=False):
            score_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            ) # [N, Mp, C]
            pair_wise_cls_loss = F.binary_cross_entropy(
                score_preds_, gt_cls, reduction="none"
            ).sum(-1) # [N, Mp]
        del score_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        ) # [N, Mp]

        (
            num_fg,
            gt_matched_classes,         # [num_fg,]
            pred_ious_this_matching,    # [num_fg,]
            matched_gt_inds,            # [num_fg,]
        ) = self.dynamic_k_matching(
            cost,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask
            )
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
        )


    def get_in_boxes_info(
        self,
        gt_bboxes,   # [N, 4]
        anchors,     # [M, 2]
        strides,     # [M,]
        num_anchors, # M
        num_gt,      # N
        ):
        # anchor center
        x_centers = anchors[:, 0]
        y_centers = anchors[:, 1]

        # [M,] -> [1, M] -> [N, M]
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        # [N,] -> [N, 1] -> [N, M]
        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors) # x1
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors) # y1
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors) # x2
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors) # y2

        b_l = x_centers - gt_bboxes_l
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center
        center_radius = self.center_sampling_radius

        # [N, 2]
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        
        # [1, M]
        center_radius_ = center_radius * strides.unsqueeze(0)

        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # x1
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_ # y1
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # x2
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_ # y2

        c_l = x_centers - gt_bboxes_l
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    
    def dynamic_k_matching(
        self, 
        cost, 
        pair_wise_ious, 
        gt_classes, 
        num_gt, 
        fg_mask
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
