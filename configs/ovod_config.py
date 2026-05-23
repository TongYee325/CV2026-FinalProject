"""
Open-Vocabulary Object Detection Configuration
===============================================
Team 1 (OVOD) - Modify freely within your directory.
"""

from dataclasses import dataclass
from typing import List
from .base_config import BaseConfig


@dataclass
class OVODConfig(BaseConfig):
    """Configuration for Open-Vocabulary Object Detection."""
    
    # Task identifier
    task: str = "ovod"
    
    # Text prompt settings
    use_subsentence_mask: bool = True       # Use sub-sentence level text representation (Sec 3.4)
    category_separator: str = "."           # Separator between category names
    
    # Datasets
    train_datasets: List[str] = None
    eval_datasets: List[str] = None
    
    # OVOD-specific training
    # Can use detection + grounding + caption data for pre-training
    use_detection_data: bool = True
    use_grounding_data: bool = True
    use_caption_data: bool = False
    
    # Evaluation
    eval_coco: bool = True
    eval_lvis: bool = True
    eval_odinw: bool = False
    
    def __post_init__(self):
        if self.train_datasets is None:
            self.train_datasets = ["o365", "goldg"]
        if self.eval_datasets is None:
            self.eval_datasets = ["coco", "lvis"]
