"""
OVOD Model Wrapper
==================
Team 1: Wraps core GroundingDINO with OVOD-specific heads/config.

Key differences from base model:
  - Sub-sentence level text representation (Sec 3.4)
  - Category concatenation with separators
  - Post-processing: map predictions to category indices
"""

import torch
import torch.nn as nn
from typing import Dict, List

import sys
sys.path.insert(0, "../..")
from grounding_dino import GroundingDINO
from configs.ovod_config import OVODConfig


class OVODGroundingDINO(nn.Module):
    """
    Grounding DINO configured for Open-Vocabulary Object Detection.
    
    Args:
        config: OVODConfig instance
    """
    
    def __init__(self, config: OVODConfig = None):
        super().__init__()
        self.config = config or OVODConfig()
        
        # Core model (shared - do not modify here!)
        self.model = GroundingDINO(
            image_backbone=self.config.image_backbone,
            text_backbone=self.config.text_backbone,
            d_model=self.config.d_model,
            num_queries=self.config.num_queries,
            num_feature_levels=self.config.num_feature_levels,
            num_feature_enhancer_layers=self.config.num_feature_enhancer_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
        )
        
    def prepare_text_prompt(self, categories: List[str]) -> str:
        """
        Prepare text prompt from category names.
        
        As per Sec 3.4, use sub-sentence level representation:
        Concatenate category names with '.' separator.
        Example: ["cat", "dog", "person"] -> "cat . dog . person ."
        
        Args:
            categories: List of category name strings
            
        Returns:
            Formatted text prompt string
        """
        sep = self.config.category_separator
        return f" {sep} ".join(categories) + f" {sep}"
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through core model.
        
        Returns:
            Same outputs as GroundingDINO
        """
        return self.model(images, text_input_ids, text_attention_mask)
    
    def post_process(
        self,
        outputs: Dict[str, torch.Tensor],
        category_mapping: Dict[int, str],
        threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Post-process model outputs to OVOD predictions.
        
        Args:
            outputs: Model output dict
            category_mapping: Maps token indices to category names
            threshold: Confidence threshold
            
        Returns:
            List of prediction dicts per sample
        """
        pred_logits = outputs["pred_logits"]  # [B, N_q, N_t]
        pred_boxes = outputs["pred_boxes"]      # [num_layers, B, N_q, 4]
        
        # Use last decoder layer for final predictions
        final_boxes = pred_boxes[-1] if pred_boxes.dim() == 4 else pred_boxes
        
        # Get max logit and corresponding text token per query
        scores, token_ids = pred_logits.max(dim=-1)  # [B, N_q]
        
        # Apply sigmoid to scores if needed (already logits)
        scores = scores.sigmoid()
        
        results = []
        for b in range(scores.shape[0]):
            keep = scores[b] > threshold
            results.append({
                "scores": scores[b][keep],
                "labels": token_ids[b][keep],
                "boxes": final_boxes[b][keep],
            })
        
        return results
