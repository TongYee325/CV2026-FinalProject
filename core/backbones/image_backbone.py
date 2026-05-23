"""
Image Backbone Module
=====================
Wraps a vision transformer (e.g. Swin Transformer) to extract multi-scale image features.
Grounding DINO extracts features at scales from 8x to 32x (or 4x to 32x for Swin-L).
"""

import torch
import torch.nn as nn
from typing import List, Dict


class ImageBackbone(nn.Module):
    """
    Image backbone wrapper.
    
    In the paper, authors use Swin Transformer (Tiny or Large).
    Multi-scale features are extracted from outputs of different blocks.
    
    Args:
        model_name: Name of the backbone model to load (e.g. 'swin_t', 'swin_l')
        return_layers: Which layer indices to return as feature scales
        pretrained: Whether to load ImageNet-pretrained weights
    """
    
    def __init__(
        self,
        model_name: str = "swin_t",
        return_layers: List[int] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.return_layers = return_layers or [1, 2, 3]  # Default 3 scales
        self.num_scales = len(self.return_layers)
        
        # TODO: Integrate with timm or torchvision Swin Transformer
        # For now, this is a placeholder that should be replaced with actual backbone
        self._build_backbone(pretrained)
        
        # Channel dimensions for Swin Transformer stages
        self.hidden_dims = self._get_hidden_dims(model_name)
        
    def _build_backbone(self, pretrained: bool):
        """Build the actual backbone network."""
        # PLACEHOLDER: Replace with actual Swin Transformer initialization
        # Example using timm:
        # import timm
        # self.backbone = timm.create_model(self.model_name, pretrained=pretrained, features_only=True)
        self.backbone = None
        
    def _get_hidden_dims(self, model_name: str) -> List[int]:
        """Return output channel dims for each scale."""
        if "tiny" in model_name.lower() or "t" in model_name.lower():
            return [192, 384, 768]  # Swin-T stages 2,3,4
        elif "large" in model_name.lower() or "l" in model_name.lower():
            return [192, 384, 768, 1536]  # Swin-L stages 1,2,3,4
        else:
            return [192, 384, 768]
    
    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            List of feature maps at different scales.
            For Swin-T: [8x, 16x, 32x] resolutions
        """
        # TODO: Implement actual forward pass
        # features = self.backbone(images)
        # return [features[i] for i in self.return_layers]
        
        # Stub for compilation
        B = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        feats = []
        for i, layer_idx in enumerate(self.return_layers):
            scale = 2 ** (layer_idx + 2)  # 8x, 16x, 32x
            h, w = H // scale, W // scale
            dim = self.hidden_dims[i]
            feats.append(torch.zeros(B, dim, h, w, device=images.device))
        return feats
