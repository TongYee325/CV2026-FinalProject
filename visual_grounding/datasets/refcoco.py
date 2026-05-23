"""
RefCOCO / RefCOCO+ / RefCOCOg Dataset
======================================
Team 2: Implement dataset loading for referring expression comprehension.

Each sample contains:
  - Image
  - Referring expression (text describing ONE target object)
  - Ground truth bounding box for the referred object

Evaluation metric: Accuracy (Pred box IoU with GT > 0.5)
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List
import json
import os


class RefCOCODataset(Dataset):
    """
    RefCOCO Dataset for Referring Expression Comprehension.
    
    Args:
        data_root: Path to dataset directory
        split: Dataset split ("train", "val", "testA", "testB")
        dataset_name: One of "refcoco", "refcocop", "refcocog"
        transforms: Image transforms
    """
    
    def __init__(self, data_root: str, split: str, dataset_name: str = "refcoco", transforms=None):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.dataset_name = dataset_name
        self.transforms = transforms
        
        # TODO: Load RefCOCO data (usually from .pkl or json files)
        # Format: list of dicts with "file_name", "bbox", "text", "ref_id"
        self.samples = []
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W]
            target: Dict with:
                - "boxes": [1, 4] single ground truth box (xyxy format)
                - "text": str referring expression
                - "ref_id": int reference id
                - "image_id": int image id
        """
        # TODO: Load actual image and annotations
        sample = self.samples[idx] if idx < len(self.samples) else {}
        
        image = torch.zeros(3, 800, 800)
        target = {
            "boxes": torch.zeros(1, 4),
            "text": sample.get("text", "the object"),
            "ref_id": sample.get("ref_id", -1),
            "image_id": sample.get("image_id", -1),
        }
        return image, target


class RECAccuracyEvaluator:
    """
    Referring Expression Comprehension evaluator.
    
    Metric: Accuracy = percentage of predictions with IoU > 0.5 with GT
    """
    
    def __init__(self):
        self.total = 0
        self.correct = 0
        
    def update(self, pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
        """
        Args:
            pred_boxes: [B, 4] predicted boxes
            gt_boxes: [B, 4] ground truth boxes
        """
        from shared_utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
        
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        gt_xyxy = box_cxcywh_to_xyxy(gt_boxes)
        
        giou = generalized_box_iou(pred_xyxy, gt_xyxy)  # [B, B]
        iou = torch.diag(giou)
        
        self.total += len(iou)
        self.correct += (iou > 0.5).sum().item()
    
    def evaluate(self):
        if self.total == 0:
            return {"accuracy": 0.0}
        return {"accuracy": self.correct / self.total}
