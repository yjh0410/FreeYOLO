import torch
import numpy as np
import torch.nn as nn

from ...backbone import build_backbone
from ...neck import build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion


class YOLOX(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLOX, self).__init__()
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk

        # backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg)

        # neck
        self.fpn = build_fpn(cfg=cfg, in_dims=bk_dim)
                                     
        # non-shared heads
        self.non_shared_heads = nn.ModuleList([DecoupledHead(cfg) for _ in range(len(cfg['stride']))])

        # pred
        head_dim = int(cfg['head_dim']*cfg['width'])
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, 1, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, self.num_classes, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, 4, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 

        # scale
        self.scales = nn.ModuleList([Scale() for _ in range(len(self.stride))])

        if trainable:
            # init bias
            self._init_biases()

        # criterion
        if self.trainable:
            self.criterion = Criterion(cfg=cfg,
                                       device=device,
                                       loss_obj_weight=cfg['loss_obj_weight'],
                                       loss_cls_weight=cfg['loss_cls_weight'],
                                       loss_reg_weight=cfg['loss_reg_weight'],
                                       num_classes=num_classes)


    def _init_biases(self):  
        # init obj pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)
        # init cls pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        anchor_xy *= self.stride[level]
        anchors = anchor_xy.to(self.device)

        return anchors
        

    def decode_boxes(self, anchors, pred_deltas):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [M, 4] (l, t, r, b)
        """
        # x1 = x_anchor - l, x2 = x_anchor + r
        # y1 = y_anchor - t, y2 = y_anchor + b
        pred_x1y1 = anchors - pred_deltas[..., :2]
        pred_x2y2 = anchors + pred_deltas[..., 2:]
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    @torch.no_grad()
    def inference_single_image(self, x):
        img_h, img_w = x.shape[2:]
        # backbone
        feats = self.backbone(x)
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]

        # neck
        pyramid_feats = self.fpn(pyramid_feats)

        # shared head
        all_scores = []
        all_labels = []
        all_bboxes = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # decode box
            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [1, C, H, W] -> [H, W, C] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)
            reg_pred = torch.exp(reg_pred) * self.stride[level]

            # scores
            scores, labels = torch.max(obj_pred.sigmoid() * cls_pred.sigmoid(), dim=-1)

            # [M, 4]
            anchors = self.generate_anchors(level, fmp_size)
            # topk
            if scores.shape[0] > self.topk:
                scores, indices = torch.topk(scores, self.topk)
                labels = labels[indices]
                reg_pred = reg_pred[indices]
                anchors = anchors[indices]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors, reg_pred)

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
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
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # normalize bbox
        bboxes /= max(img_h, img_w)
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feats = self.backbone(x)
            pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]

            # neck
            pyramid_feats = self.fpn(pyramid_feats) # [P3, P4, P5, P6, P7]

            # shared head
            all_anchors = []
            all_obj_preds = []
            all_cls_preds = []
            all_reg_preds = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # [B, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                reg_pred = torch.exp(reg_pred) * self.stride[level]

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)

                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                all_anchors.append(anchors)
            
            # output dict
            outputs = {"pred_obj": all_obj_preds,        # List [B, M, 1]
                       "pred_cls": all_cls_preds,        # List [B, M, C]
                       "pred_reg": all_reg_preds,        # List [B, M, 4]
                       'strides': self.stride}           # List [B, M,]

            # loss
            loss_dict = self.criterion(outputs = outputs, 
                                       targets = targets, 
                                       anchors = all_anchors)

            return loss_dict 
    