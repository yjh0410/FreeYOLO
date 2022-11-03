# designed for demo

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
        self.anchors, self.expand_strides = self.generate_anchors()


    def generate_anchors(self):
        """
            fmp_size: (List) [H, W]
        """
        all_anchors = []
        all_expand_strides = []
        for stride in self.strides:
            # generate grid cells
            fmp_h, fmp_w = self.img_size // stride, self.img_size // stride
            anchor_x, anchor_y = np.meshgrid(np.arange(fmp_w), np.arange(fmp_h))
            # [H, W, 2]
            anchor_xy = np.stack([anchor_x, anchor_y], axis=-1)
            shape = anchor_xy.shape[:2]
            # [H, W, 2] -> [HW, 2]
            anchor_xy = (anchor_xy.reshape(-1, 2) + 0.5) * stride
            all_anchors.append(anchor_xy)

            # expanded stride
            strides = np.full((*shape, 1), stride)
            all_expand_strides.append(strides.reshape(-1, 1))

        anchors = np.concatenate(all_anchors, axis=0)
        expand_strides = np.concatenate(all_expand_strides, axis=0)

        return anchors, expand_strides


    def decode_boxes(self, anchors, pred_regs):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors[..., :2] + pred_regs[..., :2] * self.expand_strides
        # size of bbox
        pred_box_wh = np.exp(pred_regs[..., 2:]) * self.expand_strides

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

        return pred_box


    def __call__(self, predictions):
        """
        Input:
            predictions: (ndarray) [n_anchors_all, 4+1+C]
        """
        reg_preds = predictions[..., :4]
        obj_preds = predictions[..., 4:5]
        cls_preds = predictions[..., 5:]
        scores = np.sqrt(obj_preds * cls_preds)

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
