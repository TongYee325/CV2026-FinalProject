"""
Quick Inference Demo
====================
Run this FIRST after setting up the model to verify everything works.

This loads pretrained weights and runs inference on a dummy image + text prompt.
If this runs without errors, your model setup is correct.

Usage:
    # Step 1: Inspect checkpoint keys (optional, for debugging)
    python tools/load_checkpoint.py --checkpoint ./pretrained_weights/groundingdino_swint_ogc.pth

    # Step 2: Run inference demo
    CUDA_VISIBLE_DEVICES=1 python demo_inference.py \
        --checkpoint ./pretrained_weights/groundingdino_swint_ogc.pth
"""

import torch
import argparse

from grounding_dino import GroundingDINO
from core.backbones.text_backbone import TextBackbone
from configs.base_config import BaseConfig
from tools.load_checkpoint import load_grounding_dino_checkpoint


def load_model(checkpoint_path: str, device: str):
    """Load model and pretrained weights."""
    config = BaseConfig()
    model = GroundingDINO(
        image_backbone="swin_tiny_patch4_window7_224",
        text_backbone="bert-base-uncased",
        d_model=config.d_model,
        num_queries=config.num_queries,
        num_feature_levels=3,  # Swin-T has 3 useful feature scales
    ).to(device)

    # Initialize tokenizer from text backbone
    tokenizer = TextBackbone(model_name="bert-base-uncased", max_tokens=256).tokenizer

    if checkpoint_path:
        load_grounding_dino_checkpoint(model, checkpoint_path, strict=False, verbose=True)
    else:
        print("WARNING: No checkpoint provided — using random weights!")

    model.eval()
    return model, tokenizer


@torch.no_grad()
def run_inference(model, tokenizer, image: torch.Tensor, text_prompt: str, device: str):
    """Run a single forward pass with a text prompt."""
    # Tokenize text
    encoding = tokenizer(
        [text_prompt],
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward
    outputs = model(image, input_ids, attention_mask)
    return outputs


def main(args):
    device = torch.device(args.device)

    # Load model + tokenizer
    model, tokenizer = load_model(args.checkpoint, device)

    # Dummy image (replace with real image loading for actual use)
    dummy_image = torch.randn(1, 3, 800, 800).to(device)
    text_prompt = "cat . dog . person ."

    print(f"\nRunning inference with prompt: '{text_prompt}'")
    outputs = run_inference(model, tokenizer, dummy_image, text_prompt, device)

    print(f"  pred_logits shape: {outputs['pred_logits'].shape}")
    print(f"  pred_boxes shape:  {outputs['pred_boxes'].shape}")
    print(f"  reference_boxes shape: {outputs['reference_boxes'].shape}")

    # Print top predictions
    logits = outputs["pred_logits"][0]  # [N_q, N_t]
    boxes = outputs["pred_boxes"][-1, 0]  # [N_q, 4] (last decoder layer)
    scores, token_ids = logits.max(dim=-1)
    scores = scores.sigmoid()

    top_k = 5
    top_indices = scores.topk(top_k).indices
    print(f"\nTop {top_k} predictions (from random weights = meaningless boxes):")
    for idx in top_indices:
        print(f"  Query {idx.item()}: score={scores[idx].item():.3f}, box={boxes[idx].tolist()}")

    if args.checkpoint is None:
        print("\nNOTE: You used random weights. Results are meaningless.")
        print("Download pretrained weights and rerun with --checkpoint <path>")
    else:
        print("\n✓ Inference pipeline works! Now implement dataset loaders and evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to pretrained .pth file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)
