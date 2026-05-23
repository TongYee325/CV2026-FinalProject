"""
Cross-Modality Decoder Module (Phase C Fusion)
===============================================
The detection head that refines queries using both image and text features.

Each decoder layer contains:
  1. Self-Attention on queries
  2. Image Cross-Attention (deformable)
  3. Text Cross-Attention
  4. FFN

Compared to DINO decoder, each layer has an extra text cross-attention layer.
Paper uses 6 decoder layers by default.

As shown in Fig. 3 block 3 of the paper.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalityDecoderLayer(nn.Module):
    """
    Single Cross-Modality Decoder Layer.
    
    Args:
        d_model: Feature dimension
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
        
        # 1. Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        # 2. Image Cross-Attention (Deformable in full implementation)
        self.cross_attn_image = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 3. Text Cross-Attention
        self.cross_attn_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        
        # 4. FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm4 = nn.LayerNorm(d_model)
        
    def forward(
        self,
        queries: torch.Tensor,         # [B, N_q, D]
        image_features: torch.Tensor,  # [B, N_i, D]
        text_features: torch.Tensor,   # [B, N_t, D]
        image_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            queries: [B, num_queries, d_model]
            image_features: [B, num_image_tokens, d_model]
            text_features: [B, num_text_tokens, d_model]
            image_padding_mask: [B, num_image_tokens]
            text_padding_mask: [B, num_text_tokens]
            
        Returns:
            updated_queries: [B, num_queries, d_model]
        """
        # 1. Self-Attention
        q = self.norm1(queries)
        attn_out, _ = self.self_attn(q, q, q)
        queries = queries + attn_out
        
        # 2. Image Cross-Attention
        # TODO: Replace with Deformable Cross-Attention for actual implementation
        q = self.norm2(queries)
        img_attn, _ = self.cross_attn_image(
            q, image_features, image_features,
            key_padding_mask=image_padding_mask,
        )
        queries = queries + img_attn
        
        # 3. Text Cross-Attention
        q = self.norm3(queries)
        txt_attn, _ = self.cross_attn_text(
            q, text_features, text_features,
            key_padding_mask=text_padding_mask,
        )
        queries = queries + txt_attn
        
        # 4. FFN
        queries = queries + self.ffn(self.norm4(queries))
        
        return queries


class CrossModalityDecoder(nn.Module):
    """
    Cross-Modality Decoder stacking multiple decoder layers.
    
    Args:
        d_model: Feature dimension
        n_heads: Number of attention heads
        num_layers: Number of decoder layers (default 6)
        dropout: Dropout rate
        return_intermediate: Whether to return outputs from all layers
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        return_intermediate: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalityDecoderLayer(d_model, n_heads, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
        # Output norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        queries: torch.Tensor,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        image_padding_mask: Optional[torch.Tensor] = None,
        text_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            queries: [B, num_queries, d_model]
            image_features: [B, num_image_tokens, d_model]
            text_features: [B, num_text_tokens, d_model]
            image_padding_mask: [B, num_image_tokens]
            text_padding_mask: [B, num_text_tokens]
            
        Returns:
            If return_intermediate:
                List of [B, num_queries, d_model] for each layer + final
            Else:
                [B, num_queries, d_model]
        """
        intermediate = []
        output = queries
        
        for layer in self.layers:
            output = layer(
                output,
                image_features,
                text_features,
                image_padding_mask=image_padding_mask,
                text_padding_mask=text_padding_mask,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.return_intermediate:
            return intermediate  # List of [B, N_q, D]
        
        return self.norm(output)
