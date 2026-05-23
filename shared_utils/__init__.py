"""
Shared Utilities
================
Utility functions used by both sub-teams.
Feel free to add general helpers here.
"""

from .box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, generalized_box_iou
from .text_utils import prepare_text_inputs, create_subsentence_mask

__all__ = [
    "box_cxcywh_to_xyxy",
    "box_xyxy_to_cxcywh", 
    "generalized_box_iou",
    "prepare_text_inputs",
    "create_subsentence_mask",
]
