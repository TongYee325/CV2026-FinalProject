"""
CORE MODULES - SHARED ARCHITECTURE
==================================
DO NOT MODIFY WITHOUT TEAM AGREEMENT.

These modules implement the Grounding DINO architecture as described in:
"Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"

Sub-teams should import from here but make task-specific changes in their own directories.
"""

from .backbones.image_backbone import ImageBackbone
from .backbones.text_backbone import TextBackbone
from .neck.feature_enhancer import FeatureEnhancer, FeatureEnhancerLayer
from .decoder.cross_modality_decoder import CrossModalityDecoder, CrossModalityDecoderLayer
from .query_selection.language_guided_query_selection import LanguageGuidedQuerySelection

__all__ = [
    "ImageBackbone",
    "TextBackbone", 
    "FeatureEnhancer",
    "FeatureEnhancerLayer",
    "CrossModalityDecoder",
    "CrossModalityDecoderLayer",
    "LanguageGuidedQuerySelection",
]
