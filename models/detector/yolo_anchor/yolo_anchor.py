import torch
import numpy as np
import torch.nn as nn

from ...backbone import build_backbone
from ...neck import build_neck, build_fpn
from ...head.decoupled_head import DecoupledHead

from utils.misc import nms


# Anchor-based YOLO
class AnchorYOLO(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000):
        super(AnchorYOLO, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        ## anchor config
        self.num_levels = len(cfg['stride'])
        self.num_anchors = cfg['num_anchors']
        self.anchor_size = torch.as_tensor(
            cfg['anchor_size']
            ).view(self.num_levels, self.num_anchors, 2) # [S, KA, 2]
        
        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg, trainable=trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=bk_dim[-1], out_dim=bk_dim[-1])
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=bk_dim, out_dim=cfg['head_dim'])

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [DecoupledHead(cfg) 
            for _ in range(len(cfg['stride']))
            ])

        ## pred
        head_dim = cfg['head_dim']
        self.obj_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim, self.num_anchors * 1, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim,  self.num_anchors * self.num_classes, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(head_dim,  self.num_anchors * 4, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ])                 

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
        for obj_pred in self.obj_preds:
            b = obj_pred.bias.view(self.num_anchors, -1)
            b.data.fill_(bias_value.item())
            obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(self.num_anchors, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        fmp_h, fmp_w = fmp_size
        # [KA, 2]
        anchor_size = self.anchor_size[level]
        stride = self.stride[level]

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.num_anchors, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device) * stride

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        return anchor_xy, anchor_wh
        

    def decode_boxes(self, anchor_xy, anchor_wh, pred_reg, stride):
        """
            anchor_xy:  (List[Tensor]) [M, 2]
            anchor_wh:  (List[Tensor]) [M, 2]
            pred_reg:   (List[Tensor]) [M, 4]
        """
        pred_ctr_delta = pred_reg[..., :2].sigmoid() * 3.0 - 1.5
        pred_ctr = anchor_xy + pred_ctr_delta * stride
        pred_wh = pred_reg[..., 2:].exp() * anchor_wh
        
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process(self, obj_preds, cls_preds, reg_preds, anchor_xy, anchor_wh):
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
        
        for level, (obj_pred_i, cls_pred_i, reg_pred_i, anchor_xy_i, anchor_wh_i) \
                in enumerate(zip(obj_preds, cls_preds, reg_preds, anchor_xy, anchor_wh)):
            # (H x W x KA x C,)
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
            anchor_xy_i = anchor_xy_i[anchor_idxs]
            anchor_wh_i = anchor_wh_i[anchor_idxs]

            # decode box: [M, 4]
            bboxes = self.decode_boxes(anchor_xy_i, anchor_wh_i, reg_pred_i, self.stride[level])

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
            c_keep = nms(c_bboxes, c_scores, self.nms_thresh)
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
        all_anchor_xy = []
        all_anchor_wh = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # [1, C, H, W]
            obj_pred = self.obj_preds[level](reg_feat)
            cls_pred = self.cls_preds[level](cls_feat)
            reg_pred = self.reg_preds[level](reg_feat)

            # decode box
            _, _, H, W = cls_pred.size()
            fmp_size = [H, W]
            # anchors: [M, 2]
            anchor_xy, anchor_wh = self.generate_anchors(level, fmp_size)

            # [1, KC, H, W] -> [H, W, KC] -> [M, C]
            obj_pred = obj_pred[0].permute(1, 2, 0).contiguous().view(-1, 1)
            cls_pred = cls_pred[0].permute(1, 2, 0).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred[0].permute(1, 2, 0).contiguous().view(-1, 4)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchor_xy.append(anchor_xy)
            all_anchor_wh.append(anchor_wh)

        # post process
        bboxes, scores, labels = self.post_process(
            all_obj_preds, all_cls_preds, all_reg_preds, all_anchor_xy, all_anchor_wh)

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

            # neck
            feats['layer4'] = self.neck(feats['layer4'])

            # fpn
            pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
            pyramid_feats = self.fpn(pyramid_feats)

            # head
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            all_fmp_sizes = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # [1, C, H, W]
                obj_pred = self.obj_preds[level](reg_feat)
                cls_pred = self.cls_preds[level](cls_feat)
                reg_pred = self.reg_preds[level](reg_feat)

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 4]
                anchor_xy, anchor_wh = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                box_pred = self.decode_boxes(
                    anchor_xy.unsqueeze(0),
                    anchor_wh.unsqueeze(0),
                    reg_pred,
                    self.stride[level]
                    )

                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_fmp_sizes.append(fmp_size)

            # output dict
            outputs = {"pred_obj": all_obj_preds,        # List [B, M, 1]
                       "pred_cls": all_cls_preds,        # List [B, M, C]
                       "pred_box": all_box_preds,        # List [B, M, 4]
                       'fmp_sizes': all_fmp_sizes,       # List
                       'strides': self.stride,           # List
                       }

            return outputs 
    