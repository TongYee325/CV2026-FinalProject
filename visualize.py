"""
Qualitative Visualization Script
=================================
Run inference on a single image and draw predicted boxes.
Useful for debugging and for report figures.

Usage:
    CUDA_VISIBLE_DEVICES=1 python visualize.py \
        --checkpoint ./pretrained_weights/groundingdino_swint_ogc.pth \
        --image ./data/coco/val2017/000000000139.jpg \
        --text "person . car . dog ." \
        --output ./output_vis.jpg
"""

import torch
import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from grounding_dino import GroundingDINO
from core.backbones.text_backbone import TextBackbone
from configs.base_config import BaseConfig
from tools.load_checkpoint import load_grounding_dino_checkpoint


def load_image(image_path: str, size: int = 800):
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    # Simple resize maintaining aspect ratio
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = img.resize((new_w, new_h))

    # Convert to tensor and normalize (ImageNet stats)
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(img)  # [3, H, W]
    return tensor, (w, h), img


def draw_boxes(image_pil, boxes, labels, scores, threshold=0.3, output_path="output.jpg"):
    """Draw bounding boxes on image and save."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        # box is [cx, cy, w, h] in [0, 1]
        cx, cy, bw, bh = box
        x = (cx - bw / 2) * image_pil.width
        y = (cy - bh / 2) * image_pil.height
        w = bw * image_pil.width
        h = bh * image_pil.height

        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"{label}: {score:.2f}", color='yellow', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {output_path}")
    plt.close()


def main(args):
    device = torch.device(args.device)
    config = BaseConfig()

    # Build model
    model = GroundingDINO(
        image_backbone="swin_tiny_patch4_window7_224",
        text_backbone="bert-base-uncased",
        d_model=config.d_model,
        num_queries=config.num_queries,
        num_feature_levels=3,
    ).to(device)

    tokenizer = TextBackbone(model_name="bert-base-uncased", max_tokens=256).tokenizer

    # Load checkpoint
    if args.checkpoint:
        load_grounding_dino_checkpoint(model, args.checkpoint, strict=False)
    else:
        print("WARNING: Using random weights!")

    model.eval()

    # Load image
    image_tensor, orig_size, image_pil = load_image(args.image, size=config.image_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    # Tokenize text
    encoding = tokenizer(
        [args.text],
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor, input_ids, attention_mask)

    # Post-process: take last decoder layer, threshold
    logits = outputs["pred_logits"][0]  # [N_q, N_t]
    boxes = outputs["pred_boxes"][-1, 0]  # [N_q, 4]
    scores, token_ids = logits.max(dim=-1)
    scores = scores.sigmoid()

    keep = scores > args.threshold
    pred_boxes = boxes[keep].cpu()
    pred_scores = scores[keep].cpu()
    pred_tokens = token_ids[keep].cpu()

    # For labels, we'd need to decode token_ids back to words
    # For now, just show token IDs
    pred_labels = [f"tok-{t.item()}" for t in pred_tokens]

    print(f"Found {len(pred_boxes)} predictions above threshold {args.threshold}")
    for lbl, scr in zip(pred_labels, pred_scores):
        print(f"  {lbl}: {scr.item():.3f}")

    # Visualize
    if len(pred_boxes) > 0:
        draw_boxes(image_pil, pred_boxes, pred_labels, pred_scores,
                   threshold=0, output_path=args.output)
    else:
        print("No predictions above threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to pretrained .pth")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text", required=True, help="Text prompt (e.g. 'cat . dog .')")
    parser.add_argument("--output", default="output_vis.jpg", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    main(args)
