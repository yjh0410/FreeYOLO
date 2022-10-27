import torch
import numpy as np
from .misc import nms


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


    def __call__(self, obj_preds, cls_preds, reg_preds):
        """
        Input:
            obj_preds: (ndarray) [n_anchors_all, 1]
            cls_preds: (ndarray) [n_anchors_all, C]
            reg_preds: (ndarray) [n_anchors_all, 4]
        """
        scores = obj_preds * cls_preds
        scores, labels = np.max(scores), np.argmax(scores, axis=1)
        bboxes = self.decode_boxes(self.anchors, reg_preds)

        # thresh
        keep = np.where(scores > self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = nms(c_bboxes, c_scores, )
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels
