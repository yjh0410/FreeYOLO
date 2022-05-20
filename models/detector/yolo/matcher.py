import math
import torch


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
                has_gt == False  # no gt
            elif tgt_box.max().item() == 0.:
                has_gt == False  # no valid bbox

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
                    print(gt_positions_area.shape)

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
