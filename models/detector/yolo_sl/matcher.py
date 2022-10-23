import math
import numpy as np
import torch


class YoloMatcher(object):
    def __init__(self, num_classes, num_anchors, anchor_size, iou_thresh):
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.iou_thresh = iou_thresh
        self.anchor_boxes = np.array(
            [[0., 0., anchor[0], anchor[1]]
            for anchor in anchor_size]
            )  # [KA, 4]


    def compute_iou(self, anchor_boxes, gt_box):
        """
            anchor_boxes : ndarray -> [KA, 4] (cx, cy, bw, bh).
            gt_box : ndarray -> [1, 4] (cx, cy, bw, bh).
        """
        # anchors: [KA, 4]
        anchors = np.zeros_like(anchor_boxes)
        anchors[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5  # x1y1
        anchors[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5  # x2y2
        anchors_area = anchor_boxes[..., 2] * anchor_boxes[..., 3]
        
        # gt_box: [1, 4] -> [KA, 4]
        gt_box = np.array(gt_box).reshape(-1, 4)
        gt_box = np.repeat(gt_box, anchors.shape[0], axis=0)
        gt_box_ = np.zeros_like(gt_box)
        gt_box_[..., :2] = gt_box[..., :2] - gt_box[..., 2:] * 0.5  # x1y1
        gt_box_[..., 2:] = gt_box[..., :2] + gt_box[..., 2:] * 0.5  # x2y2
        gt_box_area = np.prod(gt_box[..., 2:] - gt_box[..., :2], axis=1)

        # intersection
        inter_w = np.minimum(anchors[:, 2], gt_box_[:, 2]) - \
                  np.maximum(anchors[:, 0], gt_box_[:, 0])
        inter_h = np.minimum(anchors[:, 3], gt_box_[:, 3]) - \
                  np.maximum(anchors[:, 1], gt_box_[:, 1])
        inter_area = inter_w * inter_h
        
        # union
        union_area = anchors_area + gt_box_area - inter_area

        # iou
        iou = inter_area / union_area
        iou = np.clip(iou, a_min=1e-10, a_max=1.0)
        
        return iou


    @torch.no_grad()
    def __call__(self, fmp_size, stride, targets):
        """
            fmp_size: (List) [fmp_h, fmp_w]
            stride: (Int) -> 32 stride of network output.
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """
        # prepare
        bs = len(targets)
        fmp_h, fmp_w = fmp_size
        gt_objectness = torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 1]) 
        gt_classes = torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, self.num_classes]) 
        gt_bboxes = torch.zeros([bs, fmp_h, fmp_w, self.num_anchors, 4]) 

        for bi in range(bs):
            targets_per_image = targets[bi]
            # [N,]
            tgt_cls = targets_per_image["labels"].numpy()
            # [N, 4]
            tgt_box = targets_per_image['boxes'].numpy()

            for box, label in zip(tgt_box, tgt_cls):
                # get a bbox coords
                label = int(label)
                x1, y1, x2, y2 = box.tolist()
                # xyxy -> cxcywh
                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                gt_box = [0, 0, bw, bh]

                # check target
                if bw < 1. or bh < 1.:
                    # invalid target
                    continue

                # compute IoU
                iou = self.compute_iou(self.anchor_boxes, gt_box)
                iou_mask = (iou > self.iou_thresh)

                label_assignment_results = []
                if iou_mask.sum() == 0:
                    # We assign the anchor box with highest IoU score.
                    anchor_idx = np.argmax(iou)

                    # compute the grid cell
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, anchor_idx])
                else:            
                    for iou_ind, iou_m in enumerate(iou_mask):
                        if iou_m:
                            anchor_idx = iou_ind  # anchor index

                            # compute the gride cell
                            xc_s = xc / stride
                            yc_s = yc / stride
                            grid_x = int(xc_s)
                            grid_y = int(yc_s)

                            label_assignment_results.append([grid_x, grid_y, anchor_idx])

                # label assignment
                for result in label_assignment_results:
                    grid_x, grid_y, anchor_idx = result
                    x1s, y1s = x1 / stride, y1 / stride
                    x2s, y2s = x2 / stride, y2 / stride
                    fmp_h, fmp_w = fmp_size

                    # check
                    is_in_box = (grid_y >= y1s and grid_y < y2s) and (grid_x >= x1s and grid_x < x2s)
                    is_valid = (grid_y >= 0 and grid_y < fmp_h) and (grid_x >= 0 and grid_x < fmp_w)

                    if is_in_box and is_valid:
                        gt_objectness[bi, grid_y, grid_x, anchor_idx, 0] = 1.0
                        gt_classes[bi, grid_y, grid_x, anchor_idx, label] = 1.0
                        gt_bboxes[bi, grid_y, grid_x, anchor_idx] = torch.as_tensor([x1, y1, x2, y2])
        # [B, M, C]
        gt_objectness = gt_objectness.view(bs, -1, 1).float()
        gt_classes = gt_classes.view(bs, -1, self.num_classes).float()
        gt_bboxes = gt_bboxes.view(bs, -1, 4).float()

        return gt_objectness, gt_classes, gt_bboxes
