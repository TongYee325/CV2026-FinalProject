"""
Image Backbone Module
=====================
Real Swin Transformer via timm for multi-scale feature extraction.
"""

import torch
import torch.nn as nn
from typing import List


class ImageBackbone(nn.Module):
    """
    Image backbone wrapper using timm.

    For Grounding DINO, we use Swin Transformer and extract multi-scale features
    from different stages (typically 8x, 16x, 32x resolutions).

    Args:
        model_name: timm model name, e.g. 'swin_tiny_patch4_window7_224'
        pretrained: Whether to load ImageNet pretrained weights
        return_layers: Which stage indices to return. For Swin-T: [1, 2, 3]
        out_indices: Alternative to return_layers; passed to timm
    """

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        out_indices: List[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        # Default to stages 1,2,3 (1-based in timm FeatureListNet -> corresponds to 8x, 16x, 32x)
        self.out_indices = out_indices or [1, 2, 3]

        try:
            import timm
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=self.out_indices,
            )
            # Get actual channel dims from timm feature_info
            self.hidden_dims = [info['num_chs'] for info in self.backbone.feature_info]
            self.num_scales = len(self.hidden_dims)
        except ImportError:
            raise ImportError(
                "timm is required for the image backbone. "
                "Install it: pip install timm"
            )

    def forward(self, images: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            List of feature maps [B, C_i, H_i, W_i] at different scales.
        """
        # timm returns list of tensors directly
        features = self.backbone(images)
        return features
