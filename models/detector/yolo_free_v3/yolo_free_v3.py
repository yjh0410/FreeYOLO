import torch
import numpy as np
import torch.nn as nn

from ...basic.repconv import RepConv
from ...backbone import build_backbone
from ...neck import build_neck, build_fpn
from ...head.decoupled_head import DecoupledHead
from .loss import Criterion


# Anchor-free YOLO
class FreeYOLOv3(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(FreeYOLOv3, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        
        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg, trainable=trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=bk_dim[-1], out_dim=cfg['neck_dim'])
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=cfg['fpn_dim'], out_dim=cfg['head_dim'])

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [DecoupledHead(cfg) 
            for _ in range(len(cfg['stride']))
            ])

        ## pred
        head_dim = cfg['head_dim']
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

        # --------- Network Initialization ----------
        if trainable:
            # init bias
            self.init_yolo()

        # --------- Criterion for Training ----------
        if trainable:
            self.criterion = Criterion(
                cfg=cfg,
                device=device,
                loss_obj_weight=cfg['loss_obj_weight'],
                loss_cls_weight=cfg['loss_cls_weight'],
                loss_reg_weight=cfg['loss_reg_weight'],
                num_classes=num_classes
                )


    def init_yolo(self):  
        # Init head
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # init obj pred
        for obj_pred in self.obj_preds:
            nn.init.constant_(obj_pred.bias, bias_value)
        # init cls pred
        for cls_pred in self.cls_preds:
            nn.init.constant_(cls_pred.bias, bias_value)


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
        

    def decode_boxes(self, anchors, pred_regs, stride):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [M, 4] (l, t, r, b)
        """
        # center of bbox
        pred_ctr_xy = anchors + pred_regs[..., :2] * stride
        # size of bbox
        pred_box_wh = pred_regs[..., 2:].exp() * stride

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
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


    def fuse_repconv(self):
        print('Fusing RepCpnv layers... ')
        for m in self.modules():
            if isinstance(m, RepConv):
                #print(f" fuse_repvgg_block")
                m.fuse_repvgg_block()
        self.info()
        return self


    def post_process(self, obj_preds, cls_preds, reg_preds, anchors):
        """
        Input:
            obj_preds: List(Tensor) [[H x W, 1], ...]
            cls_preds: List(Tensor) [[H x W, C], ...]
            reg_preds: List(Tensor) [[H x W, 4], ...]
            anchors:  List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for level, (obj_pred_i, cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(obj_preds, cls_preds, reg_preds, anchors)):
            # (H x W x C,)
            scores_i = (torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

            # Keep top k top scoring indices only.
            num_topk = min(self.topk, reg_pred_i.size(0))

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores_i.sort(descending=True)
            topk_scores = predicted_prob[:num_topk]
            topk_idxs = topk_idxs[:num_topk]

            # filter out the proposals with low confidence score
            keep_idxs = topk_scores > self.conf_thresh
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            reg_pred_i = reg_pred_i[anchor_idxs]
            anchors_i = anchors_i[anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchors, reg_pred, self.stride[level])

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

        # neck
        feats['layer4'] = self.neck(feats['layer4'])

        # fpn
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # shared head
        all_obj_preds = []
        all_cls_preds = []
        all_reg_preds = []
        all_anchors = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # [M, 4]
            anchors = self.generate_anchors(level, fmp_size)

            # [1, C, H, W] -> [H, W, C] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)

        # post process
        bboxes, scores, labels = self.post_process(all_obj_preds, all_cls_preds, all_reg_preds, all_anchors)

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

            # neck
            feats['layer4'] = self.neck(feats['layer4'])

            # fpn
            pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
            pyramid_feats = self.fpn(pyramid_feats)

            # shared head
            all_anchors = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # [1, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 4]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

                # decode box: [M, 4]
                box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
            
            # output dict
            outputs = {"pred_obj": all_obj_preds,        # List(Tensor) [B, M, 1]
                       "pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [B, M, 2]
                       'strides': self.stride}           # List(Int) [8, 16, 32]

            # loss
            loss_dict = self.criterion(outputs = outputs, targets = targets)

            return loss_dict 
    