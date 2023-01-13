import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone
from ..neck import build_neck, build_fpn
from ..head import build_head

from utils.nms import multiclass_nms


# Anchor-free YOLO
class FreeYOLO(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 trainable = False, 
                 topk = 1000,
                 no_decode = False):
        super(FreeYOLO, self).__init__()
        # --------- Basic Parameters ----------
        self.cfg = cfg
        self.device = device
        self.stride = cfg['stride']
        self.reg_max = cfg['reg_max']
        self.use_dfl = cfg['reg_max'] > 0
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.no_decode = no_decode
        
        # --------- Network Parameters ----------
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, kernel_size=1, bias=False)

        ## backbone
        self.backbone, bk_dim = build_backbone(cfg=cfg, trainable=trainable)

        ## neck
        self.neck = build_neck(cfg=cfg, in_dim=bk_dim[-1], out_dim=cfg['neck_dim'])
        
        ## fpn
        self.fpn = build_fpn(cfg=cfg, in_dims=cfg['fpn_dim'], out_dim=cfg['head_dim'])

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg) 
            for _ in range(len(cfg['stride']))
            ])

        ## pred
        self.cls_preds = nn.ModuleList(
                            [nn.Conv2d(cfg['head_dim'], self.num_classes, kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ]) 
        self.reg_preds = nn.ModuleList(
                            [nn.Conv2d(cfg['head_dim'], 4*(cfg['reg_max'] + 1), kernel_size=1) 
                              for _ in range(len(cfg['stride']))
                              ])                 

        # --------- Network Initialization ----------
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
        # cls pred
        for cls_pred in self.cls_preds:
            b = cls_pred.bias.view(1, -1)
            b.data.fill_(bias_value.item())
            cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for reg_pred in self.reg_preds:
            b = reg_pred.bias.view(-1, )
            b.data.fill_(1.0)
            reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = reg_pred.weight
            w.data.fill_(0.)
            reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)


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
        Input:
            anchors:  (List[Tensor]) [1, M, 2]
            pred_reg: (List[Tensor]) [B, M, 4*(reg_max + 1)]
        Output:
            pred_box: (Tensor) [B, M, 4]
        """
        if self.use_dfl:
            B, M = pred_regs.shape[:2]
            # [B, M, 4*(reg_max + 1)] -> [B, M, 4, reg_max + 1] -> [B, 4, M, reg_max + 1]
            pred_regs = pred_regs.reshape([B, M, 4, self.reg_max + 1])
            # [B, M, 4, reg_max + 1] -> [B, reg_max + 1, 4, M]
            pred_regs = pred_regs.permute(0, 3, 2, 1).contiguous()
            # [B, reg_max + 1, 4, M] -> [B, 1, 4, M]
            pred_regs = self.proj_conv(F.softmax(pred_regs, dim=1))
            # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
            pred_regs = pred_regs.view(B, 4, M).permute(0, 2, 1).contiguous()

        # tlbr -> xyxy
        pred_x1y1 = anchors - pred_regs[..., :2] * stride
        pred_x2y2 = anchors + pred_regs[..., 2:] * stride
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def post_process(self, cls_preds, reg_preds, anchors):
        """
        Input:
            cls_preds: List(Tensor) [[B, H x W, C], ...]
            reg_preds: List(Tensor) [[B, H x W, 4*(reg_max + 1)], ...]
            anchors:   List(Tensor) [[H x W, 2], ...]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []
        
        for level, (cls_pred_i, reg_pred_i, anchors_i) in enumerate(zip(cls_preds, reg_preds, anchors)):
            # [B, M, C] -> [M, C]
            cls_pred_i = cls_pred_i[0]
            reg_pred_i = reg_pred_i[0]
            # [MC,]
            scores_i = cls_pred_i.sigmoid().flatten()

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
            box_pred_i = self.decode_boxes(
                anchors_i[None], reg_pred_i[None], self.stride[level])
            bboxes = box_pred_i[0]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

        # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.num_classes, False)

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

        # non-shared heads
        all_cls_preds = []
        all_reg_preds = []
        all_anchors = []
        for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
            cls_feat, reg_feat = head(feat)

            # pred
            cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
            reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max + 1), H, W]

            if self.no_decode:
                anchors = None
            else:
                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # [M, 4]
                anchors = self.generate_anchors(level, fmp_size)

            # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*(self.reg_max + 1))

            all_cls_preds.append(cls_pred)
            all_reg_preds.append(reg_pred)
            all_anchors.append(anchors)

        if self.no_decode:
            B, M = cls_pred.shape[:2]
            # no post process
            cls_preds = torch.cat(all_cls_preds, dim=1)  # [B, M, C]
            reg_preds = torch.cat(all_reg_preds, dim=1)  # [B, M, 4*(reg_max + 1)]

            if self.use_dfl:
                # [B, M, 4*(reg_max + 1)] -> [B, M, 4, reg_max + 1] -> [B, 4, M, reg_max + 1]
                reg_preds = reg_preds.reshape([B, M, 4, self.reg_max + 1])
                # [B, M, 4, reg_max + 1] -> [B, reg_max + 1, 4, M]
                reg_preds = reg_preds.permute(0, 3, 2, 1).contiguous()
                # [B, reg_max + 1, 4, M] -> [B, 1, 4, M]
                reg_preds = self.proj_conv(F.softmax(reg_preds, dim=1))
                # [B, 1, 4, M] -> [B, 4, M] -> [B, M, 4]
                reg_preds = reg_preds.view(B, 4, M).permute(0, 2, 1).contiguous()

            # [B, M, 4 + C]
            preds = torch.cat([reg_preds, cls_preds.sigmoid()], dim=-1)
            # [M, 4 + C]
            outputs = preds[0]

            return outputs

        else:
            # post process
            bboxes, scores, labels = self.post_process(
                all_cls_preds, all_reg_preds, all_anchors)
            
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

            # non-shared heads
            all_anchors = []
            all_cls_preds = []
            all_reg_preds = []
            all_box_preds = []
            all_strides = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.non_shared_heads)):
                cls_feat, reg_feat = head(feat)

                # pred
                cls_pred = self.cls_preds[level](cls_feat)  # [B, C, H, W]
                reg_pred = self.reg_preds[level](reg_feat)  # [B, 4*(reg_max + 1), H, W]

                B, _, H, W = cls_pred.size()
                fmp_size = [H, W]
                # generate anchor boxes: [M, 2]
                anchors = self.generate_anchors(level, fmp_size)
                
                # [B, C, H, W] -> [B, H, W, C] -> [B, M, C]
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4*(self.reg_max + 1))

                # decode box: [B, M, 4]
                box_pred = self.decode_boxes(anchors, reg_pred, self.stride[level])

                # stride tensor: [M, 1]
                stride_tensor = torch.ones_like(anchors[..., :1]) * self.stride[level]

                all_cls_preds.append(cls_pred)
                all_reg_preds.append(reg_pred)
                all_box_preds.append(box_pred)
                all_anchors.append(anchors)
                all_strides.append(stride_tensor)
            
            # output dict
            outputs = {"pred_cls": all_cls_preds,        # List(Tensor) [B, M, C]
                       "pred_reg": all_reg_preds,        # List(Tensor) [B, M, 4*(reg_max + 1)]
                       "pred_box": all_box_preds,        # List(Tensor) [B, M, 4]
                       "anchors": all_anchors,           # List(Tensor) [M, 2]
                       "strides": self.stride,           # List(Int) = [8, 16, 32]
                       "stride_tensor": all_strides      # List(Tensor) [M, 1]
                       }           
            return outputs 
    