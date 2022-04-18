import math
import torch
import torch.nn.functional as F
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss, SinkhornDistance


@torch.no_grad()
def get_ious_and_iou_loss(inputs,
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


class Matcher(object):
    def __init__(self, 
                 cfg,
                 num_classes,
                 box_weights=[1, 1, 1, 1]):
        self.num_classes = num_classes
        self.center_sampling_radius = cfg['center_sampling_radius']
        self.object_sizes_of_interest = cfg['object_sizes_of_interest']
        self.box_weightss = box_weights


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
        deltas = torch.cat((anchors - boxes[..., :2], boxes[..., 2:] - anchors),
                           dim=-1) * anchors.new_tensor(self.box_weightss)
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
        gt_classes = []
        gt_anchors_deltas = []
        gt_centerness = []
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
            # [N, C]
            tgt_cls = targets_per_image['labels'].to(device)
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

                # ground truth box regression
                # [M, 4]
                gt_anchors_reg_deltas_i = self.get_deltas(
                    anchors_over_all_feature_maps, tgt_box[gt_matched_idxs])

                # [M,]
                tgt_cls_i = tgt_cls[gt_matched_idxs]
                # anchors with area inf are treated as background.
                tgt_cls_i[positions_min_area == math.inf] = self.num_classes

                # ground truth centerness
                left_right = gt_anchors_reg_deltas_i[:, [0, 2]]
                top_bottom = gt_anchors_reg_deltas_i[:, [1, 3]]
                # [M,]
                gt_centerness_i = torch.sqrt(
                    (left_right.min(dim=-1).values / left_right.max(dim=-1).values).clamp_(min=0)
                    * (top_bottom.min(dim=-1).values / top_bottom.max(dim=-1).values).clamp_(min=0)
                )

                gt_classes.append(tgt_cls_i)
                gt_anchors_deltas.append(gt_anchors_reg_deltas_i)
                gt_centerness.append(gt_centerness_i)
                
            else:
                tgt_cls_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device) + self.num_classes
                gt_anchors_reg_deltas_i = torch.zeros([anchors_over_all_feature_maps.shape[0], 4], device=device)
                gt_centerness_i = torch.zeros(anchors_over_all_feature_maps.shape[0], device=device)

                gt_classes.append(tgt_cls_i.long())
                gt_anchors_deltas.append(gt_anchors_reg_deltas_i.float())
                gt_centerness.append(gt_centerness_i.float())

        # [B, M], [B, M, 4], [B, M]
        return torch.stack(gt_classes), torch.stack(gt_anchors_deltas), torch.stack(gt_centerness)


class OTA_Matcher(object):
    def __init__(self, 
                 cfg,
                 num_classes,
                 box_weights=[1.0, 1.0, 1.0, 1.0]) -> None:
        self.num_classes = num_classes
        self.box_weights = box_weights
        self.center_sampling_radius = cfg['center_sampling_radius']
        self.sinkhorn = SinkhornDistance(eps=cfg['eps'], max_iter=cfg['max_iter'])
        self.topk_candidate = cfg['topk_candidate']


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

        deltas = torch.cat((anchors - bboxes[..., :2], bboxes[..., 2:] - anchors),
                           dim=-1) * anchors.new_tensor(self.box_weights)
        return deltas


    @torch.no_grad()
    def __call__(self, fpn_strides, anchors, pred_cls_logits, pred_deltas, targets):
        gt_classes = []
        gt_anchors_deltas = []
        gt_ious = []
        assigned_units = []
        device = anchors[0].device

        # List[F, M, 2] -> [M, 2]
        anchors_over_all_feature_maps = torch.cat(anchors, dim=0)

        # [B, M, C]
        pred_cls_logits = torch.cat(pred_cls_logits, dim=1)
        pred_deltas = torch.cat(pred_deltas, dim=1)

        for tgt_per_image, pred_cls_per_image, pred_deltas_per_image in zip(targets, pred_cls_logits, pred_deltas):
            tgt_labels_per_images = tgt_per_image["labels"].to(device)
            tgt_bboxes_per_images = tgt_per_image["boxes"].to(device)

            # [N, M, 4], N is the number of targets, M is the number of all anchors
            deltas = self.get_deltas(anchors_over_all_feature_maps, tgt_bboxes_per_images.unsqueeze(1))
            # [N, M]
            is_in_bboxes = deltas.min(dim=-1).values > 0.01

            # targets bbox centers: [N, 2]
            centers = (tgt_bboxes_per_images[:, :2] + tgt_bboxes_per_images[:, 2:]) * 0.5
            is_in_centers = []
            for stride, anchors_i in zip(fpn_strides, anchors):
                radius = stride * self.center_sampling_radius
                center_bboxes = torch.cat((
                    torch.max(centers - radius, tgt_bboxes_per_images[:, :2]),
                    torch.min(centers + radius, tgt_bboxes_per_images[:, 2:]),
                ), dim=-1)
                # [N, Mi, 2]
                center_deltas = self.get_deltas(anchors_i, center_bboxes.unsqueeze(1))
                is_in_centers.append(center_deltas.min(dim=-1).values > 0)
            # [N, M], M = M1 + M2 + ... + MF
            is_in_centers = torch.cat(is_in_centers, dim=1)

            del centers, center_bboxes, deltas, center_deltas

            # [N, M]
            is_in_bboxes = (is_in_bboxes & is_in_centers)

            num_gt = len(tgt_labels_per_images)               # N
            num_anchor = len(anchors_over_all_feature_maps)   # M
            shape = (num_gt, num_anchor, -1)                  # [N, M, -1]

            gt_classes_per_image = F.one_hot(tgt_labels_per_images, self.num_classes).float()

            with torch.no_grad():
                loss_cls = sigmoid_focal_loss(
                    pred_cls_per_image.unsqueeze(0).expand(shape),     # [M, C] -> [1, M, C] -> [N, M, C]
                    gt_classes_per_image.unsqueeze(1).expand(shape),   # [N, C] -> [N, 1, C] -> [N, M, C]
                ).sum(dim=-1) # [N, M, C] -> [N, M]

                loss_cls_bg = sigmoid_focal_loss(
                    pred_cls_per_image,
                    torch.zeros_like(pred_cls_per_image),
                ).sum(dim=-1) # [M, C] -> [M]

                # [N, M, 4]
                gt_delta_per_image = self.get_deltas(anchors_over_all_feature_maps, tgt_bboxes_per_images.unsqueeze(1))

                # compute iou and iou loss between pred deltas and tgt deltas
                ious, loss_delta = get_ious_and_iou_loss(
                    pred_deltas_per_image.unsqueeze(0).expand(shape), # [M, 4] -> [1, M, 4] -> [N, M, 4]
                    gt_delta_per_image,
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
                # [M,]
                gt_classes_i = tgt_labels_per_images.new_ones(num_anchor) * self.num_classes
                # fg_mask: [M,]
                fg_mask = matched_gt_inds != num_gt
                gt_classes_i[fg_mask] = tgt_labels_per_images[matched_gt_inds[fg_mask]]
                gt_classes.append(gt_classes_i)
                assigned_units.append(max_assigned_units)

                # [M, 4]
                gt_anchors_deltas_per_image = gt_delta_per_image.new_zeros((num_anchor, 4))
                gt_anchors_deltas_per_image[fg_mask] = \
                    gt_delta_per_image[matched_gt_inds[fg_mask], torch.arange(num_anchor)[fg_mask]]
                gt_anchors_deltas.append(gt_anchors_deltas_per_image)

                # [M,]
                gt_ious_per_image = ious.new_zeros((num_anchor, 1))
                gt_ious_per_image[fg_mask] = ious[matched_gt_inds[fg_mask],
                                                  torch.arange(num_anchor)[fg_mask]].unsqueeze(1)
                gt_ious.append(gt_ious_per_image)


        # [B, M, C]
        gt_classes = torch.stack(gt_classes)
        # [B, M, 4]
        gt_anchors_deltas = torch.stack(gt_anchors_deltas)
        # [B, M,]
        gt_ious = torch.stack(gt_ious)

        return gt_classes, gt_anchors_deltas, gt_ious
