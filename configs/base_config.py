"""
Base Configuration for Grounding DINO
======================================
Shared hyperparameters and model settings.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BaseConfig:
    """Base configuration shared across tasks."""
    
    # Model architecture
    d_model: int = 256
    num_queries: int = 900
    n_heads: int = 8
    num_feature_enhancer_layers: int = 6
    num_decoder_layers: int = 6
    num_feature_levels: int = 4
    dropout: float = 0.1
    
    # Backbones
    image_backbone: str = "swin_t"
    text_backbone: str = "bert-base-uncased"
    
    # Training
    batch_size: int = 2
    num_epochs: int = 12
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    clip_max_norm: float = 0.1
    
    # Loss weights
    weight_class: float = 1.0
    weight_bbox: float = 5.0
    weight_giou: float = 2.0
    
    # Matching costs
    cost_class: float = 2.0
    cost_bbox: float = 5.0
    cost_giou: float = 2.0
    
    # Data
    num_workers: int = 4
    image_size: int = 800
    max_text_tokens: int = 256
