import torch
import numpy as np
from .nms import multiclass_nms


class PostProcessor(object):
    def __init__(self, img_size, strides, num_classes, conf_thresh=0.15, nms_thresh=0.5):
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.strides = strides

        # generate anchors
        self.anchors = self.generate_anchors()
        self.expand_strides = strides


    def generate_anchors(self):
        """
            fmp_size: (List) [H, W]
        """
        all_anchors = []
        for stride in self.strides:
            # generate grid cells
            fmp_h, fmp_w = self.img_size // stride, self.img_size // stride
            anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
            anchor_xy *= stride
            all_anchors.append(anchor_xy.to(self.device))

        anchors = torch.cat(all_anchors, dim=0)

        return anchors


    def decode_boxes(self, anchors, pred_regs):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors[..., :2] + pred_regs[..., :2] * self.expand_strides
        # size of bbox
        pred_box_wh = pred_regs[..., 2:].exp() * self.expand_strides

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def __call__(self, predictions):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        reg_preds = predictions[..., :4]
        obj_preds = predictions[..., 4:5]
        cls_preds = predictions[..., 5:]
        scores = obj_preds * cls_preds

        # scores & labels
        labels = np.argmax(scores, axis=1)                      # [M,]
        scores = scores[(np.arange(scores.shape[0]), labels)]   # [M,]

        # bboxes
        bboxes = self.decode_boxes(self.anchors, reg_preds)     # [M, 4]    

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, True)

        return bboxes, scores, labels
