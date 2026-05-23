"""
Visual Grounding Model Wrapper
==============================
Team 2: Wraps core GroundingDINO with REC-specific heads/config.

Key differences from OVOD:
  - Input is a referring expression (sentence), not category list
  - Sentence-level text representation (no sub-sentence mask needed)
  - Output: single box with highest score (REC has exactly one target)
"""

import torch
import torch.nn as nn
from typing import Dict, List

import sys
sys.path.insert(0, "../..")
from grounding_dino import GroundingDINO
from configs.grounding_config import GroundingConfig


class VisualGroundingDINO(nn.Module):
    """
    Grounding DINO configured for Visual Grounding / REC.
    
    Args:
        config: GroundingConfig instance
    """
    
    def __init__(self, config: GroundingConfig = None):
        super().__init__()
        self.config = config or GroundingConfig()
        
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
    
    def post_process_rec(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> List[Dict]:
        """
        Post-process for Referring Expression Comprehension.
        
        In REC, the text describes exactly ONE object.
        We select the query with the highest confidence score.
        
        Args:
            outputs: Model output dict
            
        Returns:
            List of prediction dicts per sample, each with single box
        """
        pred_logits = outputs["pred_logits"]  # [B, N_q, N_t]
        pred_boxes = outputs["pred_boxes"]      # [num_layers, B, N_q, 4]
        
        # Use last decoder layer
        final_boxes = pred_boxes[-1] if pred_boxes.dim() == 4 else pred_boxes  # [B, N_q, 4]
        
        # Max score per query (max over text tokens)
        scores, _ = pred_logits.max(dim=-1)  # [B, N_q]
        scores = scores.sigmoid()
        
        results = []
        for b in range(scores.shape[0]):
            # Select query with highest score
            best_idx = scores[b].argmax()
            results.append({
                "score": scores[b, best_idx],
                "box": final_boxes[b, best_idx],  # [4]
            })
        
        return results
