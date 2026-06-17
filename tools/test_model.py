"""
Test script for GroundingDINO model.
============================
Run this on the server to verify model instantiation and checkpoint loading.

Usage:
    cd ~/project2/cv/project/CV2026-FinalProject
    python tools/test_model.py
"""

import sys
sys.path.insert(0, ".")

import torch
from grounding_dino import GroundingDINO
from tools.load_checkpoint import load_checkpoint


def test_model():
    print("=" * 60)
    print("Testing GroundingDINO Model")
    print("=" * 60)

    # Create model
    print("\n1. Creating model...")
    model = GroundingDINO()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {total_params:,} parameters")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    model = model.to(device)

    # Load checkpoint
    print("\n2. Loading checkpoint...")
    checkpoint_path = "pretrained_weights/groundingdino_swinb_cogcoor.pth"
    import os
    if not os.path.exists(checkpoint_path):
        # Fallback: try project root
        checkpoint_path = "../pretrained_weights/groundingdino_swinb_cogcoor.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at pretrained_weights/groundingdino_swinb_cogcoor.pth. "
            f"Please ensure the checkpoint is in the pretrained_weights folder."
        )
    stats, missing, unused = load_checkpoint(model, checkpoint_path, verbose=True)

    print(f"\n   Checkpoint loading summary:")
    print(f"   - Matched: {stats['matched']} keys")
    print(f"   - Missing: {stats['missing']} keys")
    print(f"   - Unused:  {stats['unused']} keys")

    # Test forward pass
    print("\n3. Testing forward pass...")
    batch_size = 1
    img_size = 640  # Reduced to fit 12GB VRAM
    max_text_len = 256

    dummy_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, max_text_len), device=device)
    dummy_attention_mask = torch.ones(batch_size, max_text_len, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"   Output shapes:")
    print(f"   - pred_boxes: {outputs['pred_boxes'].shape}")
    print(f"   - pred_logits: {outputs['pred_logits'].shape}")

    # Test predict method
    print("\n4. Testing predict method...")
    captions = ["a cat sitting on a chair"]
    with torch.no_grad():
        results = model.predict(dummy_images, captions, confidence_threshold=0.3)
    for i, result in enumerate(results):
        print(f"   Image {i}: {len(result['boxes'])} boxes detected")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
