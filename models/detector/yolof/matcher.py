import numpy as np
import torch
from torch import nn
from utils.box_ops import *


class UniformMatcher(object):
    """
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        topk_candidates(int): Number of positive anchors for each gt box.
    """

    def __init__(self, topk_candidates: int = 4):
        self.topk_candidates = topk_candidates

    @torch.no_grad()
    def __call__(self, pred_boxes, anchor_boxes, targets):
        """
            pred_boxes: (Tensor)   [B, num_queries, 4]
            anchor_boxes: (Tensor) [num_queries, 4]
            targets: (Dict) dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}
        """

        bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [B, num_queries, 4] -> [M, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        # [num_queries, 4] -> [1, num_queries, 4] -> [B, num_queries, 4] -> [M, 4]
        anchor_boxes = anchor_boxes[None].repeat(bs, 1, 1)
        anchor_boxes = anchor_boxes.flatten(0, 1)

        # Also concat the target boxes
        tgt_bbox = torch.cat([v['boxes'] for v in targets])

        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(box_xyxy_to_cxcywh(out_bbox), 
                                box_xyxy_to_cxcywh(tgt_bbox), 
                                p=1)
        cost_bbox_anchors = torch.cdist(anchor_boxes, 
                                        box_xyxy_to_cxcywh(tgt_bbox), 
                                        p=1)

        # Final cost matrix: [B, M, N], M=num_queries, N=num_tgt
        C = cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_queries, -1).cpu()

        sizes = [len(v['boxes']) for v in targets]  # the number of object instances in each image
        all_indices_list = [[] for _ in range(bs)]
        # positive indices when matching predict boxes and gt boxes
        # len(indices) = batch size
        # len(tupe) = topk
        indices = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.topk_candidates,
                    dim=0,
                    largest=False)[1].numpy().tolist()
            )
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # positive indices when matching anchor boxes and gt boxes
        indices1 = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.topk_candidates,
                    dim=0,
                    largest=False)[1].numpy().tolist())
            for i, c in enumerate(C1.split(sizes, -1))]

        # concat the indices according to image ids
        # img_id = batch_id
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_idx_i = [
                np.array(idx_ + idx1_)
                for (idx_, idx1_) in zip(idx, idx1)
            ] # 'i' is the index of queris
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ] # 'j' is the index of tgt
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))


        return [(torch.as_tensor(i, dtype=torch.int64), 
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in all_indices]
