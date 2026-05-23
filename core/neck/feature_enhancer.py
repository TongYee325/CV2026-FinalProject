"""
Feature Enhancer Module (Phase A Fusion)
=========================================
The neck module that performs cross-modality feature fusion.

Each FeatureEnhancerLayer stacks:
  1. Self-Attention (vanilla for text, deformable for image)
  2. Image-to-Text Cross-Attention
  3. Text-to-Image Cross-Attention
  4. FFN for both branches

As shown in Fig. 3 block 2 of the paper.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class FeatureEnhancerLayer(nn.Module):
    """
    Single Feature Enhancer Layer.
    
    Processes image and text features in parallel with cross-attention
    between modalities.
    
    Args:
        d_model: Feature dimension (default 256)
        n_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # ---- Image branch ----
        # Deformable Self-Attention for image features (placeholder)
        # In full implementation, use MultiScaleDeformableAttention
        self.img_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.img_cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.img_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.img_norm1 = nn.LayerNorm(d_model)
        self.img_norm2 = nn.LayerNorm(d_model)
        self.img_norm3 = nn.LayerNorm(d_model)
        
        # ---- Text branch ----
        # Vanilla Self-Attention for text features
        self.txt_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.txt_cross_attn_img = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.txt_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.txt_norm1 = nn.LayerNorm(d_model)
        self.txt_norm2 = nn.LayerNorm(d_model)
        self.txt_norm3 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        image_features: torch.Tensor,  # [B, N_i, D]
        text_features: torch.Tensor,   # [B, N_t, D]
        image_padding_mask: Optional[torch.Tensor] = None,  # [B, N_i]
        text_padding_mask: Optional[torch.Tensor] = None,   # [B, N_t]
    ):
        """
        Args:
            image_features: [B, num_image_tokens, d_model]
            text_features: [B, num_text_tokens, d_model]
            image_padding_mask: [B, num_image_tokens]
            text_padding_mask: [B, num_text_tokens]
            
        Returns:
            updated_image_features: [B, num_image_tokens, d_model]
            updated_text_features: [B, num_text_tokens, d_model]
        """
        # ---- Image Self-Attention ----
        # TODO: Replace with Deformable Self-Attention for actual implementation
        img_q = self.img_norm1(image_features)
        img_self, _ = self.img_self_attn(img_q, img_q, img_q, key_padding_mask=image_padding_mask)
        image_features = image_features + img_self
        
        # ---- Image -> Text Cross-Attention (Text queries, Image keys/values) ----
        txt_q = self.txt_norm1(text_features)
        txt_cross, _ = self.txt_cross_attn_img(
            txt_q, image_features, image_features,
            key_padding_mask=image_padding_mask,
        )
        text_features = text_features + txt_cross
        
        # ---- Text Self-Attention ----
        txt_q2 = self.txt_norm2(text_features)
        txt_self, _ = self.txt_self_attn(txt_q2, txt_q2, txt_q2, key_padding_mask=text_padding_mask)
        text_features = text_features + txt_self
        
        # ---- Text -> Image Cross-Attention (Image queries, Text keys/values) ----
        img_q2 = self.img_norm2(image_features)
        img_cross, _ = self.img_cross_attn_text(
            img_q2, text_features, text_features,
            key_padding_mask=text_padding_mask,
        )
        image_features = image_features + img_cross
        
        # ---- FFN ----
        image_features = image_features + self.img_ffn(self.img_norm3(image_features))
        text_features = text_features + self.txt_ffn(self.txt_norm3(text_features))
        
        return image_features, text_features


class FeatureEnhancer(nn.Module):
    """
    Feature Enhancer Module stacking multiple FeatureEnhancerLayers.
    
    Paper uses 6 feature enhancer layers by default.
    
    Args:
        d_model: Feature dimension
        n_heads: Number of attention heads
        num_layers: Number of stacked enhancer layers (default 6)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FeatureEnhancerLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args/Returns same as FeatureEnhancerLayer but processes through all layers.
        """
        for layer in self.layers:
            image_features, text_features = layer(
                image_features, text_features,
                image_padding_mask=image_padding_mask,
                text_padding_mask=text_padding_mask,
            )
        return image_features, text_features
