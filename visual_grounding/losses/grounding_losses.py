"""
Visual Grounding Loss Functions
================================
Team 2: Implements losses for Referring Expression Comprehension.

In REC, each text refers to exactly ONE object.
Loss components:
  - Contrastive loss (same as OVOD but usually binary: target vs background)
  - L1 box regression loss
  - GIoU loss
  
Can be simplified compared to OVOD since there's only one target.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, "../..")
from core.utils.matcher import HungarianMatcher
from shared_utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy


class GroundingLoss(nn.Module):
    """
    Loss for Visual Grounding / REC.
    
    Args:
        weight_class: Weight for classification loss
        weight_bbox: Weight for L1 bbox loss
        weight_giou: Weight for GIoU loss
    """
    
    def __init__(
        self,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ):
        super().__init__()
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
        )
    
    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Classification loss for REC.
        
        In REC, each query is matched to either the target (text-described)
        or background.
        """
        B, N_q, N_t = pred_logits.shape
        target = torch.zeros_like(pred_logits)
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue
            gt_labels = targets[i]["labels"][tgt_idx]
            target[i, src_idx, gt_labels] = 1.0
        
        # Binary cross entropy with logits
        return F.binary_cross_entropy_with_logits(pred_logits, target, reduction="mean")
    
    def loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Box regression losses."""
        l1_loss = 0.0
        giou_loss = 0.0
        num_boxes = 0
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue
            src_boxes = pred_boxes[i, src_idx]
            tgt_boxes = targets[i]["boxes"][tgt_idx]
            
            src_xyxy = box_cxcywh_to_xyxy(src_boxes)
            tgt_xyxy = box_cxcywh_to_xyxy(tgt_boxes)
            
            l1_loss += F.l1_loss(src_boxes, tgt_boxes, reduction="sum")
            giou_loss += (1 - torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))).sum()
            num_boxes += len(tgt_idx)
        
        if num_boxes > 0:
            l1_loss /= num_boxes
            giou_loss /= num_boxes
        
        return l1_loss, giou_loss
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Compute total loss."""
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        
        if pred_boxes.dim() == 4:
            pred_boxes = pred_boxes[-1]
        
        indices = self.matcher(
            {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
            targets,
        )
        
        loss_class = self.loss_labels(pred_logits, targets, indices)
        loss_l1, loss_giou = self.loss_boxes(pred_boxes, targets, indices)
        
        total_loss = (
            self.weight_class * loss_class +
            self.weight_bbox * loss_l1 +
            self.weight_giou * loss_giou
        )
        
        return {
            "loss": total_loss,
            "loss_class": loss_class,
            "loss_bbox": loss_l1,
            "loss_giou": loss_giou,
        }
