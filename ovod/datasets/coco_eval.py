"""
COCO Dataset and Evaluator for OVOD
====================================
Team 1: Implement COCO loading and COCO-style AP evaluation.

For zero-shot evaluation:
  - Map COCO categories to text prompts (e.g. "person . car . dog .")
  - Run model inference
  - Compute AP@0.5:0.95 using pycocotools
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class COCODataset(Dataset):
    """
    COCO Dataset for Open-Vocabulary Object Detection.
    
    Args:
        img_dir: Path to COCO images
        ann_file: Path to COCO annotations
        transforms: Image transforms
    """
    
    def __init__(self, img_dir: str, ann_file: str, transforms=None):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transforms = transforms
        
        # TODO: Load COCO annotations via pycocotools
        # from pycocotools.coco import COCO
        # self.coco = COCO(ann_file)
        # self.img_ids = list(self.coco.imgs.keys())
        self.img_ids = []
        
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W]
            target: Dict with:
                - "boxes": [num_gt, 4] in xyxy format
                - "labels": [num_gt] text descriptions or class indices
                - "image_id": int
        """
        # TODO: Implement actual loading
        image = torch.zeros(3, 800, 800)
        target = {
            "boxes": torch.zeros(0, 4),
            "labels": [],
            "image_id": self.img_ids[idx] if idx < len(self.img_ids) else -1,
        }
        return image, target
    
    def get_text_prompt(self) -> str:
        """
        Get the text prompt for all COCO categories.
        For zero-shot: concatenate category names separated by '.'
        """
        # TODO: Return category names concatenated
        return "person . bicycle . car ."


class COCOEvaluator:
    """
    COCO-style AP evaluator.
    
    Uses pycocotools to compute AP@0.5:0.95, AP50, AP75, etc.
    """
    
    def __init__(self, coco_gt):
        self.coco_gt = coco_gt
        
    def update(self, predictions):
        """Add batch predictions."""
        pass
    
    def evaluate(self):
        """Compute final metrics."""
        # TODO: Use pycocotools COCOeval
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
