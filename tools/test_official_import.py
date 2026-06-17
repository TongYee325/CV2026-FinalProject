"""Test if we can import from the official GroundingDINO repo."""

import sys
import os
# Try to find GroundingDINO relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
gdino_path = os.path.join(project_root, "GroundingDINO")
if os.path.exists(gdino_path):
    sys.path.insert(0, gdino_path)
    print(f"Added {gdino_path} to PYTHONPATH")
else:
    print(f"Warning: {gdino_path} not found. Trying ../GroundingDINO")
    sys.path.insert(0, "../GroundingDINO")

# Test basic imports
try:
    from groundingdino.models.GroundingDINO.ms_deform_attn import MultiScaleDeformableAttention
    print("✅ MSDeformAttn imported successfully")
except Exception as e:
    print(f"❌ MSDeformAttn import failed: {e}")

try:
    from groundingdino.models.GroundingDINO.fuse_modules import BiAttentionBlock
    print("✅ BiAttentionBlock imported successfully")
except Exception as e:
    print(f"❌ BiAttentionBlock import failed: {e}")

try:
    from groundingdino.models.GroundingDINO.transformer import Transformer
    print("✅ Transformer imported successfully")
except Exception as e:
    print(f"❌ Transformer import failed: {e}")

try:
    from groundingdino.models.GroundingDINO.bertwarper import (
        generate_masks_with_special_tokens_and_transfer_map
    )
    print("✅ generate_masks_with_special_tokens imported successfully")
except Exception as e:
    print(f"❌ generate_masks import failed: {e}")

print("\nIf all imports succeed, we can use the official code directly!")
