import numpy as np
import torch
import torch.nn as nn
from ...backbone import build_backbone
from ...neck import build_neck
from ...head.decoupled_head import DecoupledHead


# Single-level YOLO
class SLYOLO(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(SLYOLO, self).__init__()
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
        self.obj_pred = nn.Conv2d(cfg['head_dim'], 1 * self.num_anchors, kernel_size=1)
        self.cls_pred = nn.Conv2d(cfg['head_dim'], self.num_classes * self.num_anchors, kernel_size=1)
        self.reg_pred = nn.Conv2d(cfg['head_dim'], 4 * self.num_anchors, kernel_size=1)

        # --------- Network Initialization ----------
        if trainable:
            # init bias
            self.init_yolo()


    def init_yolo(self): 
        # Init yolo
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03
                
        # Init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # obj pred
        b = self.obj_pred.bias.view(self.num_anchors, -1)
        b.data.fill_(bias_value.item())
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        b = self.cls_pred.bias.view(self.num_anchors, -1)
        b.data.fill_(bias_value.item())
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def generate_anchors(self, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        return anchor_xy, anchor_wh
        

    def decode_boxes(self, anchor_xy, anchor_wh, pred_reg):
        """
            anchor_xy:  [Tensor] -> [M, 2]
            anchor_wh:  [Tensor] -> [M, 2]
            pred_reg:   [Tensor] -> [M, 4]
        """
        pred_ctr_delta = pred_reg[..., :2].sigmoid()
        pred_ctr = (anchor_xy + pred_ctr_delta) * self.stride
        pred_wh = pred_reg[..., 2:].exp() * anchor_wh
        
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
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


    def post_process(self, obj_pred, cls_pred, reg_pred, anchor_xy, anchor_wh):
        """
        Input:
            obj_pred: (Tensor) -> [H x W x KA, 1]
            cls_pred: (Tensor) -> [H x W x KA, C]
            reg_pred: (Tensor) -> [H x W x KA, 4]
            anchors:  (Tensor) -> [H x W x KA, 2]
        """
    
        # (H x W x KA x C,)
        scores = (torch.sqrt(obj_pred.sigmoid() * cls_pred.sigmoid())).flatten()

        # Keep top k top scoring indices only.
        num_topk = min(self.topk, reg_pred.size(0))

        # torch.sort is actually faster than .topk (at least on GPUs)
        predicted_prob, topk_idxs = scores.sort(descending=True)
        topk_scores = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]

        # filter out the proposals with low confidence score
        keep_idxs = topk_scores > self.conf_thresh
        scores = topk_scores[keep_idxs]
        topk_idxs = topk_idxs[keep_idxs]

        anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
        labels = topk_idxs % self.num_classes

        reg_pred = reg_pred[anchor_idxs]
        anchor_xy = anchor_xy[anchor_idxs]
        anchor_wh = anchor_wh[anchor_idxs]

        # decode box: [M, 4]
        bboxes = self.decode_boxes(anchor_xy, anchor_wh, reg_pred)

        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

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

        return bboxes, scores, labels


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

        # generate anchor boxes
        _, _, H, W = cls_pred.size()
        fmp_size = [H, W]
        anchor_xy, anchor_wh = self.generate_anchors(fmp_size)

        # [1, KC, H, W] -> [H, W, KC] -> [M, C]
        obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
        cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
        reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

        # post process
        bboxes, scores, labels = self.post_process(obj_pred, cls_pred, reg_pred, anchor_xy, anchor_wh)

        # normalize bbox
        bboxes /= max(img_h, img_w)
        bboxes = bboxes.clip(0., 1.)

        return bboxes, scores, labels


    def forward(self, x):
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

            B, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # generate anchor boxes: [M, 4]
            anchor_xy, anchor_wh = self.generate_anchors(fmp_size)
            
            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
            box_pred = self.decode_boxes(anchor_xy.unsqueeze(0), anchor_wh.unsqueeze(0), reg_pred)
            
            # output dict
            outputs = {"pred_obj": obj_pred,        # List [B, M, 1]
                       "pred_cls": cls_pred,        # List [B, M, C]
                       "pred_box": box_pred,        # List [B, M, 4]
                       'fmp_size': fmp_size,        # List
                       'stride': self.stride,       # Int
                       }

            return outputs 
