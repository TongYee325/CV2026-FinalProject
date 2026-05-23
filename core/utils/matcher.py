"""
Hungarian Matcher for Grounding DINO
=====================================
Performs bipartite matching between predictions and ground truths.

Matching cost includes:
  - Classification cost (contrastive loss)
  - Box L1 cost
  - Box GIoU cost

As described in Sec. 3.5 of the paper.
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for assigning predictions to ground truth targets.
    
    Args:
        cost_class: Weight for classification cost
        cost_bbox: Weight for box L1 cost
        cost_giou: Weight for box GIoU cost
    """
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            outputs: Dict with:
                - "pred_logits": [B, num_queries, num_text_tokens]
                - "pred_boxes": [B, num_queries, 4]
            targets: List of dicts per sample, each with:
                - "labels": [num_gt] class indices (or text token indices)
                - "boxes": [num_gt, 4] normalized boxes
                
        Returns:
            List of (index_i, index_j) tuples for each batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute cost matrix
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*N_q, N_t]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N_q, 4]
        
        indices = []
        
        for i in range(bs):
            tgt_ids = targets[i]["labels"]  # [num_gt]
            tgt_bbox = targets[i]["boxes"]  # [num_gt, 4]
            
            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            
            # Classification cost: negative log prob of target text tokens
            cost_class = -out_prob[i * num_queries:(i + 1) * num_queries, tgt_ids]
            
            # L1 cost
            cost_bbox = torch.cdist(out_bbox[i * num_queries:(i + 1) * num_queries], tgt_bbox, p=1)
            
            # GIoU cost (placeholder - need actual giou implementation)
            cost_giou = -self._generalized_box_iou(
                out_bbox[i * num_queries:(i + 1) * num_queries],
                tgt_bbox,
            )
            
            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.cpu()
            
            indices_i, indices_j = linear_sum_assignment(C)
            indices.append((torch.as_tensor(indices_i, dtype=torch.long), torch.as_tensor(indices_j, dtype=torch.long)))
        
        return indices
    
    def _generalized_box_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """
        Compute GIoU between two sets of boxes in (cx, cy, w, h) format.
        
        Args:
            boxes1: [N, 4]
            boxes2: [M, 4]
            
        Returns:
            giou: [N, M]
        """
        # TODO: Implement actual GIoU or import from torchvision.ops
        # For now, return zeros as placeholder
        return torch.zeros(boxes1.shape[0], boxes2.shape[0], device=boxes1.device)
