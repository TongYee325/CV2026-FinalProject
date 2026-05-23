"""
Grounding DINO - Main Model
============================
Full implementation of the Grounding DINO architecture.

Architecture Flow:
  1. Image Backbone -> Vanilla Image Features
  2. Text Backbone -> Vanilla Text Features
  3. Feature Enhancer -> Enhanced Image/Text Features
  4. Language-Guided Query Selection -> Decoder Queries
  5. Cross-Modality Decoder -> Refined Queries
  6. Prediction Heads -> Boxes + Text Logits

Reference: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from core import (
    ImageBackbone,
    TextBackbone,
    FeatureEnhancer,
    LanguageGuidedQuerySelection,
    CrossModalityDecoder,
)


class GroundingDINO(nn.Module):
    """
    Grounding DINO Model.
    
    Args:
        image_backbone: Image backbone config or instance
        text_backbone: Text backbone config or instance
        d_model: Feature dimension (default 256)
        num_queries: Number of decoder queries (default 900)
        num_feature_levels: Number of image feature scales
        num_feature_enhancer_layers: Number of feature enhancer layers (default 6)
        num_decoder_layers: Number of decoder layers (default 6)
        n_heads: Number of attention heads (default 8)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        image_backbone: str = "swin_t",
        text_backbone: str = "bert-base-uncased",
        d_model: int = 256,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        num_feature_enhancer_layers: int = 6,
        num_decoder_layers: int = 6,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries
        
        # 1. Backbones
        self.image_backbone = ImageBackbone(model_name=image_backbone)
        self.text_backbone = TextBackbone(model_name=text_backbone, max_tokens=256)
        
        # Input projections (backbone dims may differ from d_model)
        # For Swin-T: typically 192/384/768; for BERT: 768
        self.img_input_proj = nn.ModuleList([
            nn.Conv2d(self.image_backbone.hidden_dims[i], d_model, kernel_size=1)
            for i in range(self.image_backbone.num_scales)
        ])
        self.txt_input_proj = nn.Linear(self.text_backbone.hidden_dim, d_model)
        
        # 2. Feature Enhancer (Neck)
        self.feature_enhancer = FeatureEnhancer(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_feature_enhancer_layers,
            dropout=dropout,
        )
        
        # 3. Language-Guided Query Selection
        self.query_selection = LanguageGuidedQuerySelection(
            d_model=d_model,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
        )
        
        # 4. Cross-Modality Decoder
        self.decoder = CrossModalityDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_decoder_layers,
            dropout=dropout,
            return_intermediate=True,
        )
        
        # 5. Prediction Heads
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        # Contrastive classification head is just a dot product with text features
        
    def _flatten_multi_scale_features(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Flatten multi-scale image features to a single sequence.
        
        Args:
            features: List of [B, D, H_i, W_i] for each scale
            
        Returns:
            [B, sum(H_i*W_i), D]
        """
        B = features[0].shape[0]
        flattened = []
        for feat in features:
            # [B, D, H, W] -> [B, H*W, D]
            feat = feat.flatten(2).permute(0, 2, 1)
            flattened.append(feat)
        return torch.cat(flattened, dim=1)
    
    def forward(
        self,
        images: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: [B, 3, H, W]
            text_input_ids: [B, max_tokens]
            text_attention_mask: [B, max_tokens]
            
        Returns:
            dict with:
                - "pred_logits": [B, num_queries, num_text_tokens]
                - "pred_boxes": [num_decoder_layers+1, B, num_queries, 4] or [B, num_queries, 4]
                - "reference_boxes": [B, num_queries, 4]
        """
        B = images.shape[0]
        
        # 1. Backbone feature extraction
        multi_scale_img_feats = self.image_backbone(images)  # List of [B, D, H, W]
        text_features = self.text_backbone(text_input_ids, text_attention_mask)  # [B, N_t, D_txt]
        
        # Project to common dimension
        img_feats_projected = []
        for i, feat in enumerate(multi_scale_img_feats):
            feat = self.img_input_proj[i](feat)  # [B, d_model, H, W]
            img_feats_projected.append(feat)
        
        text_features = self.txt_input_proj(text_features)  # [B, N_t, d_model]
        
        # Flatten image features for transformer processing
        # [B, N_i, d_model] where N_i = sum(H_i*W_i)
        img_features_flat = self._flatten_multi_scale_features(img_feats_projected)
        
        # Create image padding mask (all False since we use full feature maps)
        img_padding_mask = None
        text_padding_mask = ~text_attention_mask.bool() if text_attention_mask is not None else None
        
        # 2. Feature Enhancer
        enhanced_img, enhanced_txt = self.feature_enhancer(
            img_features_flat,
            text_features,
            image_padding_mask=img_padding_mask,
            text_padding_mask=text_padding_mask,
        )
        
        # 3. Language-Guided Query Selection
        content_queries, positional_queries, reference_boxes = self.query_selection(
            enhanced_img,
            enhanced_txt,
            text_padding_mask=text_padding_mask,
        )
        
        # Combine content and positional for decoder input
        # Paper uses dynamic anchor boxes as positional part
        decoder_queries = content_queries + positional_queries  # [B, N_q, D]
        
        # 4. Cross-Modality Decoder
        decoder_outputs = self.decoder(
            decoder_queries,
            enhanced_img,
            enhanced_txt,
            image_padding_mask=img_padding_mask,
            text_padding_mask=text_padding_mask,
        )  # List of [B, N_q, D] for each layer
        
        # 5. Prediction Heads
        # Stack all decoder layer outputs: [num_layers, B, N_q, D]
        decoder_outputs_stacked = torch.stack(decoder_outputs, dim=0)
        
        # Predict boxes from each decoder layer output
        pred_boxes = self.bbox_head(decoder_outputs_stacked).sigmoid()  # [num_layers, B, N_q, 4]
        
        # Contrastive classification: dot product query with text features
        # Use last layer output for logits
        last_layer_queries = decoder_outputs[-1]  # [B, N_q, D]
        
        # [B, N_q, D] @ [B, D, N_t] -> [B, N_q, N_t]
        pred_logits = torch.bmm(last_layer_queries, enhanced_txt.transpose(1, 2))
        
        return {
            "pred_logits": pred_logits,           # [B, N_q, N_t]
            "pred_boxes": pred_boxes,             # [num_layers, B, N_q, 4] or [B, N_q, 4]
            "reference_boxes": reference_boxes,   # [B, N_q, 4]
        }
