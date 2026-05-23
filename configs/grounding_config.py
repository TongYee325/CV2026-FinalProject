"""
Visual Grounding (Referring Expression Comprehension) Configuration
====================================================================
Team 2 (Visual Grounding) - Modify freely within your directory.
"""

from dataclasses import dataclass
from typing import List
from .base_config import BaseConfig


@dataclass
class GroundingConfig(BaseConfig):
    """Configuration for Visual Grounding / Referring Expression Comprehension."""
    
    # Task identifier
    task: str = "grounding"
    
    # Text prompt settings
    use_subsentence_mask: bool = False      # REC uses full sentence representation
    max_text_tokens: int = 256
    
    # Datasets
    train_datasets: List[str] = None
    eval_datasets: List[str] = None
    
    # REC-specific settings
    # In REC, each sentence describes exactly one object
    num_queries: int = 900                  # Still use 900 but usually only 1 target
    
    # Evaluation
    eval_refcoco: bool = True
    eval_refcocop: bool = True
    eval_refcocog: bool = True
    eval_flickr30k: bool = False
    
    def __post_init__(self):
        if self.train_datasets is None:
            self.train_datasets = ["refcoco", "refcocop", "refcocog"]
        if self.eval_datasets is None:
            self.eval_datasets = ["refcoco", "refcocop", "refcocog"]
