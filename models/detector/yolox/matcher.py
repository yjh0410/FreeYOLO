import math
import torch
import torch.nn.functional as F
from utils.box_ops import box_iou
from utils.misc import SinkhornDistance


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

            has_gt = (len(tgt_cls) > 0)
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
                positions_min_area, gt_matched_idxs = gt_positions_area.min(dim=0)

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
                tgt_obj_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device)
                tgt_cls_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device) + self.num_classes
                tgt_reg_i = torch.zeros([anchors_over_all_feature_maps.shape[0], 4], device=device)

                gt_objectness.append(tgt_obj_i)
                gt_classes.append(tgt_cls_i)
                gt_anchors_deltas.append(tgt_reg_i)

        # [B, M], [B, M], [B, M, 4]
        return torch.stack(gt_objectness), torch.stack(gt_classes), torch.stack(gt_anchors_deltas)


class OTA(object):
    def __init__(self, 
                 num_classes,
                 center_sampling_radius,
                 topk_candidate,
                 eps=0.1,
                 max_iter=50) -> None:
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=max_iter)


    @torch.no_grad()
    def get_ious_and_iou_loss(self,
                            inputs,
                            targets,
                            weight=None,
                            box_mode="xyxy",
                            loss_type="iou",
                            reduction="none"):
        """
        Compute iou loss of type ['iou', 'giou', 'linear_iou']

        Args:
            inputs (tensor): pred values
            targets (tensor): target values
            weight (tensor): loss weight
            box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
            loss_type (str): 'giou' or 'iou' or 'linear_iou'
            reduction (str): reduction manner

        Returns:
            loss (tensor): computed iou loss.
        """
        if box_mode == "ltrb":
            inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
            targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
        elif box_mode != "xyxy":
            raise NotImplementedError

        eps = torch.finfo(torch.float32).eps

        inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
            * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
        targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
            * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                    - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                    - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = targets_area + inputs_area - area_intersect
        ious = area_intersect / area_union.clamp(min=eps)

        if loss_type == "iou":
            loss = -ious.clamp(min=eps).log()
        elif loss_type == "linear_iou":
            loss = 1 - ious
        elif loss_type == "giou":
            g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
                - torch.min(inputs[..., 0], targets[..., 0])
            g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
                - torch.min(inputs[..., 1], targets[..., 1])
            ac_uion = g_w_intersect * g_h_intersect
            gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
            loss = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None:
            loss = loss * weight.view(loss.size())
            if reduction == "mean":
                loss = loss.sum() / max(weight.sum().item(), eps)
        else:
            if reduction == "mean":
                loss = loss.mean()
        if reduction == "sum":
            loss = loss.sum()

        return ious, loss


    def get_deltas(self, anchors, bboxes):
        """
        Get box regression transformation deltas (dl, dr) that can be used
        to transform the `anchors` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, anchors)`` is true.

        Args:
            anchors (Tensor): anchors, e.g., feature map coordinates
            bboxes (Tensor): target of the transformation, e.g., ground-truth bboxes.
        """
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(anchors, torch.Tensor), type(anchors)

        deltas = torch.cat((anchors - bboxes[..., :2], bboxes[..., 2:] - anchors), dim=-1)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, obj_preds, cls_preds, reg_preds, targets):
        gt_objectness = []
        gt_classes = []
        gt_anchors_deltas = []
        device = anchors[0].device

        # List[F, M, 2] -> [M, 2]
        anchors_over_all_feature_maps = torch.cat(anchors, dim=0)

        # [B, M, C]
        obj_preds = torch.cat(obj_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)

        for target, obj_pred, cls_pred, reg_pred in zip(targets, obj_preds, cls_preds, reg_preds):
            # [N,]
            tgt_cls = target["labels"].to(device)
            # [N, 4]
            tgt_box = target["boxes"].to(device)
            # [N,]
            tgt_obj = torch.ones_like(tgt_cls)

            # check target
            if tgt_box.max().item == 0.:
                # There is no valid gt
                tgt_obj_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device)
                tgt_cls_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device) + self.num_classes
                tgt_reg_i = torch.zeros([anchors_over_all_feature_maps.shape[0], 4], device=device)

                gt_objectness.append(tgt_obj_i)
                gt_classes.append(tgt_cls_i)
                gt_anchors_deltas.append(tgt_reg_i)

            else:
                # [N, M, 4], N is the number of targets, M is the number of all anchors
                deltas = self.get_deltas(anchors_over_all_feature_maps, tgt_box.unsqueeze(1))
                # [N, M]
                is_in_bboxes = deltas.min(dim=-1).values > 0.01

                # targets bbox centers: [N, 2]
                centers = (tgt_box[:, :2] + tgt_box[:, 2:]) * 0.5
                is_in_centers = []
                for stride, anchors_i in zip(fpn_strides, anchors):
                    radius = stride * self.center_sampling_radius
                    center_bboxes = torch.cat((
                        torch.max(centers - radius, tgt_box[:, :2]),
                        torch.min(centers + radius, tgt_box[:, 2:]),
                    ), dim=-1)
                    # [N, Mi, 2]
                    center_deltas = self.get_deltas(anchors_i, center_bboxes.unsqueeze(1))
                    is_in_centers.append(center_deltas.min(dim=-1).values > 0)
                # [N, M], M = M1 + M2 + ... + MF
                is_in_centers = torch.cat(is_in_centers, dim=1)

                del centers, center_bboxes, deltas, center_deltas

                # [N, M]
                is_in_bboxes = (is_in_bboxes & is_in_centers)

                num_gt = len(tgt_cls)               # N
                num_anchor = len(anchors_over_all_feature_maps)   # M
                shape = (num_gt, num_anchor, -1)                  # [N, M, -1]

                tgt_cls_ = F.one_hot(tgt_cls, self.num_classes).float()

                with torch.no_grad():
                    cls_pred_ = torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())
                    loss_cls = F.binary_cross_entropy_with_logits(
                        cls_pred_.unsqueeze(0).expand(shape),     # [M, C] -> [1, M, C] -> [N, M, C]
                        tgt_cls_.unsqueeze(1).expand(shape),   # [N, C] -> [N, 1, C] -> [N, M, C]
                        reduction='none'
                    ).sum(dim=-1) # [N, M, C] -> [N, M]
                    loss_cls_bg = F.binary_cross_entropy_with_logits(
                        cls_pred_,
                        torch.zeros_like(cls_pred_),
                        reduction='none'
                    ).sum(dim=-1) # [M, C] -> [M]

                    # [N, M, 4]
                    tgt_delta = self.get_deltas(anchors_over_all_feature_maps, tgt_box.unsqueeze(1))

                    # compute iou and iou loss between pred deltas and tgt deltas
                    ious, loss_delta = self.get_ious_and_iou_loss(
                        reg_pred.unsqueeze(0).expand(shape), # [M, 4] -> [1, M, 4] -> [N, M, 4]
                        tgt_delta,
                        box_mode="ltrb",
                        loss_type='iou'
                    ) # [N, M]

                    loss = loss_cls + 3.0 * loss_delta + 1e6 * (1 - is_in_bboxes.float())

                    # Performing Dynamic k Estimation, top_candidates = 20
                    topk_ious, _ = torch.topk(ious * is_in_bboxes.float(), self.topk_candidate, dim=1)
                    mu = ious.new_ones(num_gt + 1)
                    mu[:-1] = torch.clamp(topk_ious.sum(1).int(), min=1).float()
                    mu[-1] = num_anchor - mu[:-1].sum()
                    nu = ious.new_ones(num_anchor)
                    loss = torch.cat([loss, loss_cls_bg.unsqueeze(0)], dim=0)

                    # Solving Optimal-Transportation-Plan pi via Sinkhorn-Iteration.
                    _, pi = self.sinkhorn(mu, nu, loss)

                    # Rescale pi so that the max pi for each gt equals to 1.
                    rescale_factor, _ = pi.max(dim=1)
                    pi = pi / rescale_factor.unsqueeze(1)

                    # matched_gt_inds: [M,]
                    max_assigned_units, matched_gt_inds = torch.max(pi, dim=0)
                    # fg_mask: [M,]
                    fg_mask = matched_gt_inds != num_gt
                    # [M,]
                    tgt_cls_i = tgt_cls.new_ones(num_anchor) * self.num_classes
                    tgt_cls_i[fg_mask] = tgt_cls[matched_gt_inds[fg_mask]]
                    gt_classes.append(tgt_cls_i)

                    # ground truth objectness [M,]
                    # [M,]
                    tgt_obj_i = tgt_obj.new_ones(num_anchor) * 0
                    tgt_obj_i[fg_mask] = tgt_obj[matched_gt_inds[fg_mask]]
                    gt_objectness.append(tgt_obj_i)

                    # [M, 4]
                    tgt_delta_i = tgt_delta.new_zeros((num_anchor, 4))
                    tgt_delta_i[fg_mask] = \
                        tgt_delta[matched_gt_inds[fg_mask], torch.arange(num_anchor)[fg_mask]]
                    gt_anchors_deltas.append(tgt_delta_i)

        return (
                    torch.stack(gt_objectness),     # [B, M]
                    torch.stack(gt_classes),        # [B, M]
                    torch.stack(gt_anchors_deltas)  # [B, M, 4]
                    )


class SimOTA(object):
    def __init__(self, 
                 num_classes,
                 center_sampling_radius,
                 topk_candidate
                 ):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius
        self.topk_candidate = topk_candidate


    @torch.no_grad()
    def get_ious_and_iou_loss(self,
                            inputs,
                            targets,
                            weight=None,
                            box_mode="xyxy",
                            loss_type="iou",
                            reduction="none"):
        """
        Compute iou loss of type ['iou', 'giou', 'linear_iou']

        Args:
            inputs (tensor): pred values
            targets (tensor): target values
            weight (tensor): loss weight
            box_mode (str): 'xyxy' or 'ltrb', 'ltrb' is currently supported.
            loss_type (str): 'giou' or 'iou' or 'linear_iou'
            reduction (str): reduction manner

        Returns:
            loss (tensor): computed iou loss.
        """
        if box_mode == "ltrb":
            inputs = torch.cat((-inputs[..., :2], inputs[..., 2:]), dim=-1)
            targets = torch.cat((-targets[..., :2], targets[..., 2:]), dim=-1)
        elif box_mode != "xyxy":
            raise NotImplementedError

        eps = torch.finfo(torch.float32).eps

        inputs_area = (inputs[..., 2] - inputs[..., 0]).clamp_(min=0) \
            * (inputs[..., 3] - inputs[..., 1]).clamp_(min=0)
        targets_area = (targets[..., 2] - targets[..., 0]).clamp_(min=0) \
            * (targets[..., 3] - targets[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(inputs[..., 2], targets[..., 2])
                    - torch.max(inputs[..., 0], targets[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(inputs[..., 3], targets[..., 3])
                    - torch.max(inputs[..., 1], targets[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = targets_area + inputs_area - area_intersect
        ious = area_intersect / area_union.clamp(min=eps)

        if loss_type == "iou":
            loss = -ious.clamp(min=eps).log()
        elif loss_type == "linear_iou":
            loss = 1 - ious
        elif loss_type == "giou":
            g_w_intersect = torch.max(inputs[..., 2], targets[..., 2]) \
                - torch.min(inputs[..., 0], targets[..., 0])
            g_h_intersect = torch.max(inputs[..., 3], targets[..., 3]) \
                - torch.min(inputs[..., 1], targets[..., 1])
            ac_uion = g_w_intersect * g_h_intersect
            gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
            loss = 1 - gious
        else:
            raise NotImplementedError
        if weight is not None:
            loss = loss * weight.view(loss.size())
            if reduction == "mean":
                loss = loss.sum() / max(weight.sum().item(), eps)
        else:
            if reduction == "mean":
                loss = loss.mean()
        if reduction == "sum":
            loss = loss.sum()

        return ious, loss


    def get_deltas(self, anchors, bboxes):
        """
        Get box regression transformation deltas (dl, dr) that can be used
        to transform the `anchors` into the `boxes`. That is, the relation
        ``boxes == self.apply_deltas(deltas, anchors)`` is true.

        Args:
            anchors (Tensor): anchors, e.g., feature map coordinates
            bboxes (Tensor): target of the transformation, e.g., ground-truth bboxes.
        """
        assert isinstance(anchors, torch.Tensor), type(anchors)
        assert isinstance(anchors, torch.Tensor), type(anchors)

        deltas = torch.cat((anchors - bboxes[..., :2], bboxes[..., 2:] - anchors), dim=-1)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, obj_preds, cls_preds, reg_preds, targets):
        gt_objectness = []
        gt_classes = []
        gt_anchors_deltas = []
        device = anchors[0].device

        # List[F, M, 2] -> [M, 2]
        anchors_over_all_feature_maps = torch.cat(anchors, dim=0)

        # [B, M, C]
        obj_preds = torch.cat(obj_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        reg_preds = torch.cat(reg_preds, dim=1)

        for target, obj_pred, cls_pred, reg_pred in zip(targets, obj_preds, cls_preds, reg_preds):
            # [N,]
            tgt_cls = target["labels"].to(device)
            # [N, 4]
            tgt_box = target["boxes"].to(device)
            # [N,]
            tgt_obj = torch.ones_like(tgt_cls)

            # check target
            if tgt_box.max().item == 0.:
                # There is no valid gt
                tgt_obj_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device)
                tgt_cls_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device) + self.num_classes
                tgt_reg_i = torch.zeros([anchors_over_all_feature_maps.shape[0], 4], device=device)

                gt_objectness.append(tgt_obj_i)
                gt_classes.append(tgt_cls_i)
                gt_anchors_deltas.append(tgt_reg_i)

            else:            
                # [N, M, 4]
                deltas = self.get_deltas(anchors_over_all_feature_maps, tgt_box.unsqueeze(1))
                # [N, M]
                is_in_bboxes = (deltas.min(dim=-1).values > 0.0)
                # [M,]
                is_in_boxes_all = (is_in_bboxes.sum(dim=0) > 0)

                # targets bbox centers: [N, 2]
                centers = (tgt_box[:, :2] + tgt_box[:, 2:]) * 0.5
                is_in_centers = []
                for stride, anchors_i in zip(fpn_strides, anchors):
                    radius = stride * self.center_sampling_radius
                    center_bboxes = torch.cat((
                        torch.max(centers - radius, tgt_box[:, :2]),
                        torch.min(centers + radius, tgt_box[:, 2:]),
                    ), dim=-1)
                    # [N, Mi, 2]
                    center_deltas = self.get_deltas(anchors_i, center_bboxes.unsqueeze(1))
                    is_in_centers.append(center_deltas.min(dim=-1).values > 0)
                # [N, M], M = M1 + M2 + ... + MF
                is_in_centers = torch.cat(is_in_centers, dim=1)
                # [M,]
                is_in_centers_all = is_in_centers.sum(dim=0) > 0

                del centers, center_bboxes, deltas, center_deltas

                # posotive candidates: [M,]
                is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all
                fg_mask = is_in_boxes_anchor

                # both in bboxes and center: [Mp,]
                is_in_boxes_and_center = (
                    is_in_bboxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
                )

                obj_pred_ = obj_pred[fg_mask]                      # [Mp, 1]
                cls_pred_ = cls_pred[fg_mask]                      # [Mp, C]
                reg_pred_ = reg_pred[fg_mask]                      # [Mp, 4]
                anchors_ = anchors_over_all_feature_maps[fg_mask]   # [Mp, 2]
                num_in_boxes_anchor = obj_pred_.shape[0]
                num_gt = len(tgt_cls)
                num_anchor = obj_pred.shape[0]
                shape = (num_gt, num_in_boxes_anchor, -1)          # [N, Mp, -1]

                # [N, Mp, 4]
                tgt_delta_ = self.get_deltas(anchors_, tgt_box.unsqueeze(1))
                pair_wise_ious, pair_wise_ious_loss = self.get_ious_and_iou_loss(
                    reg_pred_.unsqueeze(0).expand(shape), # [M, 4] -> [1, M, 4] -> [N, M, 4]
                    tgt_delta_,
                    box_mode="ltrb",
                    loss_type='iou'
                ) # [N, M]

                tgt_cls_ = F.one_hot(tgt_cls, self.num_classes).float()

                with torch.cuda.amp.autocast(enabled=False):
                    scores_ = torch.sqrt(obj_pred_.sigmoid() * cls_pred_.sigmoid())
                    pair_wise_cls_loss = F.binary_cross_entropy_with_logits(
                        scores_.unsqueeze(0).expand(shape),     # [Mp, C] -> [1, Mp, C] -> [N, Mp, C]
                        tgt_cls_.unsqueeze(1).expand(shape),    # [N, C] -> [N, 1, C] -> [N, Mp, C]
                        reduction='none'
                    ).sum(dim=-1) # [N, Mp, C] -> [N, Mp]
                del scores_

                # [N, Mp]
                cost = (
                    pair_wise_cls_loss
                    + 3.0 * pair_wise_ious_loss
                    + 1e6 * (1 - is_in_boxes_and_center.float())
                )

                # Dynamic k Estimation
                matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
                n_candidate_k = min(self.topk_candidate, pair_wise_ious.size(1))
                topk_ious, _ = torch.topk(pair_wise_ious, n_candidate_k, dim=1)
                dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1) # [N,]
                dynamic_ks = dynamic_ks.tolist()
                
                # Dynamic K matching
                for gt_idx in range(num_gt):
                    try:
                        _, pos_idx = torch.topk(
                            cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
                        )
                        matching_matrix[gt_idx][pos_idx] = 1
                    except:
                        print(is_in_boxes_anchor.sum())
                        print(is_in_bboxes.sum())
                        print(is_in_centers.sum())
                        print(tgt_box)

                del topk_ious, dynamic_ks, pos_idx

                anchor_matching_gt = matching_matrix.sum(0)
                if (anchor_matching_gt > 1).sum() > 0:
                    _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                    matching_matrix[:, anchor_matching_gt > 1] *= 0
                    matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
                # [Mp,]
                fg_mask_inboxes = matching_matrix.sum(0) > 0

                # [M,]
                fg_mask[fg_mask.clone()] = fg_mask_inboxes

                # [Mp,]
                matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

                # ground truth objectness [M,]
                tgt_obj_i = tgt_obj.new_ones(num_anchor) * 0
                tgt_obj_i[fg_mask] = tgt_obj[matched_gt_inds]
                gt_objectness.append(tgt_obj_i)

                # ground truth classification [M,]               
                tgt_cls_i = tgt_cls.new_ones(num_anchor) * self.num_classes
                tgt_cls_i[fg_mask] = tgt_cls[matched_gt_inds]
                gt_classes.append(tgt_cls_i)

                # ground truth regression [M, 4]               
                tgt_delta = self.get_deltas(anchors_over_all_feature_maps, tgt_box.unsqueeze(1))
                tgt_delta_i = tgt_delta.new_zeros((num_anchor, 4))
                tgt_delta_i[fg_mask] = \
                    tgt_delta[matched_gt_inds, torch.arange(num_anchor)[fg_mask]]
                gt_anchors_deltas.append(tgt_delta_i)

        return (
                    torch.stack(gt_objectness),     # [B, M]
                    torch.stack(gt_classes),        # [B, M]
                    torch.stack(gt_anchors_deltas)  # [B, M, 4]
                    )

