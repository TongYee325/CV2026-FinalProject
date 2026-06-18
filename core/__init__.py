"""Core package exports for the local Grounding DINO implementation."""

from .backbones.image_backbone import ImageBackbone
from .backbones.text_backbone import TextBackbone

__all__ = [
    "ImageBackbone",
    "TextBackbone",
]
