"""
Image Backbone Module
=====================
Uses official GroundingDINO Swin Transformer + position encoding.
Copied from official repo to match checkpoint exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .position_encoding import PositionEmbeddingSineHW
from .swin_transformer import build_swin_transformer


class ImageBackbone(nn.Module):
    """
    Image backbone wrapper using official GroundingDINO implementation.

    For Swin-B, outputs 3 feature levels with channels [256, 512, 1024].
    Position embeddings are computed dynamically via sine encoding.

    Args:
        model_name: One of 'swin_B_224_22k', 'swin_B_384_22k', 'swin_T_224_1k'
        pretrained: Not used (we load from GroundingDINO checkpoint)
        out_indices: Which stage indices to return (default [1, 2, 3])
    """

    def __init__(
        self,
        model_name: str = "swin_B_384_22k",
        pretrained: bool = False,
        out_indices: List[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.out_indices = out_indices or [1, 2, 3]

        # Parse model name to get pretrain_img_size
        parts = model_name.split("_")
        pretrain_img_size = int(parts[2])  # e.g. "swin_B_384_22k" -> 384

        # Build official Swin Transformer
        self.backbone = build_swin_transformer(
            model_name,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(self.out_indices),
            dilation=False,
            use_checkpoint=False,
        )

        # Channel dimensions for each output level
        # For Swin-B (embed_dim=128): num_features = [128, 256, 512, 1024]
        # With out_indices=[1,2,3]: hidden_dims = [256, 512, 1024]
        self.hidden_dims = self.backbone.num_features[
            4 - len(self.out_indices) :
        ]
        self.num_scales = len(self.hidden_dims)

        # Position embedding matching official implementation
        self.position_embedding = PositionEmbeddingSineHW(
            num_pos_feats=128,
            temperatureH=20,
            temperatureW=20,
            normalize=True,
        )

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            features: List of feature maps [B, C_i, H_i, W_i]
            pos_embeds: List of position embeddings [B, 256, H_i, W_i]
        """
        # Run official Swin forward (returns list of feature maps)
        features = self.backbone.forward_raw(images)

        # Generate position embeddings for each scale
        pos_embeds = []
        for feat in features:
            # Create dummy mask (all valid, no padding)
            B, _, H, W = feat.shape
            mask = torch.zeros((B, H, W), dtype=torch.bool, device=feat.device)
            
            # PositionEmbeddingSineHW expects a NestedTensor-like object
            # We pass tensors and mask directly
            pos = self.position_embedding.forward_simple(feat, mask)
            pos_embeds.append(pos)

        return features, pos_embeds


# Monkey-patch PositionEmbeddingSineHW to accept raw tensors
# The official version expects a NestedTensor, but we can add a helper
_original_forward = PositionEmbeddingSineHW.forward


def _forward_simple(self, x: torch.Tensor, mask: torch.Tensor):
    """Forward that accepts raw tensors instead of NestedTensor."""
    assert mask is not None
    not_mask = ~mask
    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

    dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_tx = self.temperatureW ** (2 * (torch.div(dim_tx, 2, rounding_mode='floor')) / self.num_pos_feats)
    pos_x = x_embed[:, :, :, None] / dim_tx

    dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_ty = self.temperatureH ** (2 * (torch.div(dim_ty, 2, rounding_mode='floor')) / self.num_pos_feats)
    pos_y = y_embed[:, :, :, None] / dim_ty

    pos_x = torch.stack(
        (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos_y = torch.stack(
        (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
    ).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    return pos


PositionEmbeddingSineHW.forward_simple = _forward_simple
