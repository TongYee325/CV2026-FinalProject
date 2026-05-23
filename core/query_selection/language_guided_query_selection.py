"""
Language-Guided Query Selection Module (Phase B Fusion)
========================================================
Selects cross-modality queries from image features based on text relevance.

Given image features X_I [B, N_i, D] and text features X_T [B, N_t, D],
select top N_q indices using:
    I_{N_q} = Top_{N_q}(Max^{(-1)}(X_I @ X_T^T))

Then initialize decoder queries with:
  - Positional part: dynamic anchor boxes initialized from encoder outputs
  - Content part: learnable embeddings

As shown in Fig. 3 block 1 and Algorithm 1 of the paper.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LanguageGuidedQuerySelection(nn.Module):
    """
    Language-Guided Query Selection.
    
    Args:
        d_model: Feature dimension
        num_queries: Number of queries to select (default 900)
        num_feature_levels: Number of image feature scales
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 900,
        num_feature_levels: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        
        # Projections to align features if needed
        self.img_proj = nn.Linear(d_model, d_model)
        self.txt_proj = nn.Linear(d_model, d_model)
        
        # Learnable content queries (the "content part" in mixed query selection)
        self.content_queries = nn.Embedding(num_queries, d_model)
        
        # For generating reference boxes from selected features
        self.reference_head = nn.Linear(d_model, 4)
        
    def forward(
        self,
        image_features: torch.Tensor,  # [B, N_i, D] - flattened multi-scale features
        text_features: torch.Tensor,   # [B, N_t, D]
        text_padding_mask: Optional[torch.Tensor] = None,  # [B, N_t]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: Flattened image features [B, N_i, D]
            text_features: Text features [B, N_t, D]
            text_padding_mask: Padding mask for text tokens [B, N_t]
            
        Returns:
            content_queries: Learnable content queries [B, N_q, D]
            positional_queries: Positional queries from selected features [B, N_q, D]
            reference_boxes: Initial reference boxes [B, N_q, 4] in (cx, cy, w, h) format
        """
        B = image_features.shape[0]
        N_i = image_features.shape[1]
        
        # Project features
        img_feats = self.img_proj(image_features)  # [B, N_i, D]
        txt_feats = self.txt_proj(text_features)   # [B, N_t, D]
        
        # Compute similarity: [B, N_i, N_t]
        # X_I @ X_T^T
        similarity = torch.bmm(img_feats, txt_feats.transpose(1, 2))
        
        # Mask out padded text tokens
        if text_padding_mask is not None:
            # text_padding_mask: True for padded tokens
            similarity = similarity.masked_fill(
                text_padding_mask.unsqueeze(1).expand(-1, N_i, -1),
                float('-inf'),
            )
        
        # Max over text dimension: [B, N_i]
        max_similarity, _ = similarity.max(dim=-1)
        
        # Top-N_q selection: [B, N_q]
        topk_values, topk_indices = torch.topk(
            max_similarity, k=min(self.num_queries, N_i), dim=1
        )
        
        # Handle case where N_i < num_queries
        actual_nq = topk_indices.shape[1]
        
        # Gather selected image features as positional queries
        # [B, N_q, D]
        positional_queries = torch.gather(
            image_features, 1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        
        # Content queries (learnable)
        content_queries = self.content_queries.weight[:actual_nq].unsqueeze(0).expand(B, -1, -1)
        
        # Pad if needed
        if actual_nq < self.num_queries:
            pad_size = self.num_queries - actual_nq
            content_queries = torch.cat([
                content_queries,
                self.content_queries.weight[actual_nq:self.num_queries].unsqueeze(0).expand(B, -1, -1)
            ], dim=1)
            positional_queries = torch.cat([
                positional_queries,
                positional_queries[:, -1:].expand(-1, pad_size, -1)
            ], dim=1)
        
        # Predict reference boxes from positional queries
        reference_boxes = self.reference_head(positional_queries).sigmoid()
        # reference_boxes: [B, N_q, 4] as (cx, cy, w, h)
        
        return content_queries, positional_queries, reference_boxes
