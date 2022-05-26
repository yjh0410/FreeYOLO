import numpy as np
import math
import torch
import torch.nn as nn
from ...backbone import build_backbone
from ...neck import build_neck
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion


DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
DEFAULT_EXP_CLAMP = math.log(1e8)


class YOLOF(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(YOLOF, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.fmp_size = None
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])
        self.num_anchors = len(cfg['anchor_size'])
        
        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg, trainable=trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=bk_dim[-1], out_dim=cfg['head_dim'])
                                     
        ## head
        self.head = DecoupledHead(cfg) 

        ## pred
        self.obj_pred = nn.Conv2d(cfg['head_dim'], 1 * self.num_anchors, kernel_size=3, padding=1)
        self.cls_pred = nn.Conv2d(cfg['head_dim'], self.num_classes * self.num_anchors, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'], 4 * self.num_anchors, kernel_size=3, padding=1)

        # --------- Network Initialization ----------
        if trainable:
            # init bias
            self._init_pred_layers()

        # --------- Criterion for Training ----------
        if trainable:
            self.criterion = Criterion(
                cfg=cfg,
                device=device,
                alpha=cfg['alpha'],
                gamma=cfg['gamma'],
                loss_cls_weight=cfg['loss_cls_weight'],
                loss_reg_weight=cfg['loss_reg_weight'],
                num_classes=num_classes)


    def _init_pred_layers(self):  
        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)
        # init obj pred
        nn.init.normal_(self.obj_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.obj_pred.bias, 0.0)


    def generate_anchors(self, fmp_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # generate grid cells
        fmp_h, fmp_w = fmp_size
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
        anchor_xy *= self.stride

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4] or [M, 4]
            pred_reg: (List[tensor]) [B, M, 4] or [M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        if self.cfg['ctr_clamp'] is not None:
            pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                        max=self.cfg['ctr_clamp'],
                                        min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, 
                                max=DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
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
        feat = feats['layer4']

        # neck
        feat = self.neck(feat)
        H, W = feat.shape[2:]

        # head
        cls_feats, reg_feats = self.head(feat)

        obj_pred = self.obj_pred(reg_feats)
        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        # implicit objectness
        B, _, H, W = obj_pred.size()
        obj_pred = obj_pred.view(B, -1, 1, H, W)
        cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)
        cls_pred = cls_pred.sigmoid() * obj_pred.sigmoid()
        # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
        cls_pred = cls_pred.permute(0, 3, 4, 1, 2).contiguous()
        cls_pred = cls_pred.view(B, -1, self.num_classes)

        # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        reg_pred =reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        reg_pred = reg_pred.view(B, -1, 4)

        # [1, M, C] -> [M, C]
        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]

        # anchor boxes
        anchor_boxes = self.generate_anchors(fmp_size=[H, W]) # [M, 4]

        # scores
        scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

        # topk
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]

        # decode box: [M, 4]
        bboxes = self.decode_boxes(anchor_boxes, reg_pred)

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
            feat = feats['layer4']

            # neck
            feat = self.neck(feat)
            H, W = feat.shape[2:]

            # head
            cls_feats, reg_feats = self.head(feat)

            obj_pred = self.obj_pred(reg_feats)
            cls_pred = self.cls_pred(cls_feats)
            reg_pred = self.reg_pred(reg_feats)

            # implicit objectness
            B, _, H, W = obj_pred.size()
            obj_pred = obj_pred.view(B, -1, 1, H, W)
            cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)

            normalized_cls_pred = cls_pred + obj_pred - torch.log(
                    1. + 
                    torch.clamp(cls_pred, max=DEFAULT_EXP_CLAMP).exp() + 
                    torch.clamp(obj_pred, max=DEFAULT_EXP_CLAMP).exp())
            # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
            normalized_cls_pred = normalized_cls_pred.permute(0, 3, 4, 1, 2).contiguous()
            normalized_cls_pred = normalized_cls_pred.view(B, -1, self.num_classes)

            # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
            reg_pred =reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
            reg_pred = reg_pred.view(B, -1, 4)

            # decode box
            anchor_boxes = self.generate_anchors(fmp_size=[H, W]) # [M, 4]
            box_pred = self.decode_boxes(anchor_boxes[None], reg_pred) # [B, M, 4]
            
            outputs = {
                'pred_cls': normalized_cls_pred,
                'pred_box': box_pred,
                'strides': self.stride,
                'anchors': anchor_boxes
                }

            # loss
            loss_dict = self.criterion(outputs=outputs, targets=targets)

            return loss_dict 
