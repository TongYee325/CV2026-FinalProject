"""
OVOD Loss Functions
===================
Team 1: Implements losses for Open-Vocabulary Object Detection.

Components:
  - Contrastive loss (focal loss on query-text similarity)
  - L1 box regression loss
  - GIoU loss
  - Auxiliary losses for intermediate decoder layers

As described in Sec. 3.5 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, "../..")
from core.utils.matcher import HungarianMatcher
from shared_utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy


class OVODLoss(nn.Module):
    """
    Combined loss for OVOD.
    
    Args:
        weight_class: Weight for classification (contrastive) loss
        weight_bbox: Weight for L1 bbox loss
        weight_giou: Weight for GIoU loss
        focal_alpha: Focal loss alpha
        focal_gamma: Focal loss gamma
    """
    
    def __init__(
        self,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
        )
    
    def focal_loss(self, pred_logits: torch.Tensor, target_logits: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for contrastive classification.
        
        Args:
            pred_logits: [B, N_q, N_t]
            target_logits: [B, N_q, N_t] binary targets
        """
        pred_prob = pred_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(pred_logits, target_logits, reduction="none")
        p_t = pred_prob * target_logits + (1 - pred_prob) * (1 - target_logits)
        loss = ce_loss * ((1 - p_t) ** self.focal_gamma)
        if self.focal_alpha >= 0:
            alpha_t = self.focal_alpha * target_logits + (1 - self.focal_alpha) * (1 - target_logits)
            loss = alpha_t * loss
        return loss.mean()
    
    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Contrastive classification loss.
        
        Args:
            pred_logits: [B, N_q, N_t]
            targets: Ground truth list
            indices: Matching indices from Hungarian matcher
        """
        B, N_q, N_t = pred_logits.shape
        
        # Build target: [B, N_q, N_t] binary tensor
        target = torch.zeros_like(pred_logits)
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue
            # Map matched queries to their target text token indices
            # targets[i]["labels"] contains text token indices for each GT
            gt_labels = targets[i]["labels"][tgt_idx]
            target[i, src_idx, gt_labels] = 1.0
        
        return self.focal_loss(pred_logits, target)
    
    def loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Box regression losses (L1 + GIoU).
        
        Args:
            pred_boxes: [B, N_q, 4] in cxcywh format
        """
        l1_loss = 0.0
        giou_loss = 0.0
        num_boxes = 0
        
        for i, (src_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) == 0:
                continue
            src_boxes = pred_boxes[i, src_idx]  # [num_gt, 4]
            tgt_boxes = targets[i]["boxes"][tgt_idx]  # [num_gt, 4]
            
            # Convert to xyxy for GIoU
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
        """
        Compute total loss.
        
        Args:
            outputs: Model outputs with "pred_logits" and "pred_boxes"
            targets: List of ground truth dicts
            
        Returns:
            Dict of losses
        """
        # Last layer outputs
        pred_logits = outputs["pred_logits"]  # [B, N_q, N_t]
        pred_boxes = outputs["pred_boxes"]
        
        # Use last decoder layer for box prediction in loss
        if pred_boxes.dim() == 4:
            pred_boxes = pred_boxes[-1]  # [B, N_q, 4]
        
        # Hungarian matching
        indices = self.matcher(
            {"pred_logits": pred_logits, "pred_boxes": pred_boxes},
            targets,
        )
        
        # Compute losses
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
