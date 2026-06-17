"""
Test script for GroundingDINO V2 (official architecture).
"""

import sys
sys.path.insert(0, ".")

import torch
from grounding_dino_v2 import GroundingDINOV2
from tools.load_checkpoint_v2 import load_checkpoint_v2


def test_v2():
    print("=" * 60)
    print("Testing GroundingDINO V2")
    print("=" * 60)

    print("\n1. Creating V2 model...")
    model = GroundingDINOV2()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model created with {total_params:,} parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    model = model.to(device)

    print("\n2. Loading checkpoint...")
    checkpoint_path = "pretrained_weights/groundingdino_swinb_cogcoor.pth"
    stats, missing, unused = load_checkpoint_v2(model, checkpoint_path, verbose=True)
    print(f"\n   Matched: {stats['matched']} / {len(model.state_dict())}")
    print(f"   Missing: {stats['missing']}")
    print(f"   Unused: {stats['unused']}")

    print("\n3. Testing forward pass...")
    dummy_images = torch.randn(1, 3, 640, 640, device=device)
    dummy_input_ids = torch.zeros(1, 256, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, 256, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images, dummy_input_ids, dummy_attention_mask)

    print(f"   pred_logits: {outputs['pred_logits'].shape}")
    print(f"   pred_boxes: {outputs['pred_boxes'].shape}")

    print("\n4. Testing predict...")
    captions = ["a cat sitting on a chair"]
    with torch.no_grad():
        results = model.predict(dummy_images, captions, confidence_threshold=0.3)
    print(f"   Detected {len(results[0]['boxes'])} boxes")

    print("\n" + "=" * 60)
    print("V2 test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_v2()
