"""
LVIS Dataset and Evaluator for OVOD
====================================
Team 1: Implement LVIS loading and evaluation.

LVIS contains 1000+ categories including rare ones.
Key metrics: AP, APr (rare), APc (common), APf (frequent)
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List


class LVISDataset(Dataset):
    """LVIS Dataset for zero-shot open-vocabulary detection."""
    
    def __init__(self, img_dir: str, ann_file: str, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms
        self.img_ids = []
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        image = torch.zeros(3, 800, 800)
        target = {
            "boxes": torch.zeros(0, 4),
            "labels": [],
            "image_id": -1,
        }
        return image, target
    
    def get_text_prompt(self) -> str:
        """Concatenate all LVIS category names."""
        return ""


class LIVSEvaluator:
    """LVIS AP evaluator with rare/common/frequent breakdown."""
    
    def __init__(self, lvis_gt):
        self.lvis_gt = lvis_gt
        
    def update(self, predictions):
        pass
    
    def evaluate(self):
        return {"AP": 0.0, "APr": 0.0, "APc": 0.0, "APf": 0.0}
