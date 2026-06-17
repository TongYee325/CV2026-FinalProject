"""
Qualitative Visualization Script
=================================
Run inference on a single image and draw predicted boxes.
Useful for debugging and for report figures.

Usage:
    CUDA_VISIBLE_DEVICES=1 python visualize.py \
        --checkpoint ./pretrained_weights/groundingdino_swinb_cogcoor.pth \
        --image ./data/coco/val2017/000000000139.jpg \
        --text "person . car . dog ." \
        --output ./output_vis.jpg \
        --threshold 0.3
"""

import torch
import argparse
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

from grounding_dino import GroundingDINO
from tools.load_checkpoint import load_checkpoint


# GroundingDINO uses ImageNet normalization
IMAGENET_MEAN = [0.485, 0.406, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_image(image_path: str, size: int = 800):
    """Load and preprocess image."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    # Resize so longest side = size (maintaining aspect ratio)
    scale = size / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_resized = img.resize((new_w, new_h))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(img_resized)  # [3, H, W]
    return tensor, (orig_w, orig_h), img


def draw_boxes(image_pil, boxes, labels, scores, threshold=0.3, output_path="output.jpg"):
    """Draw bounding boxes on image and save."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)

    colors = plt.cm.rainbow(torch.linspace(0, 1, len(set(labels))).numpy())
    label_to_color = {lbl: colors[i] for i, lbl in enumerate(sorted(set(labels)))}

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        # box is [cx, cy, w, h] in [0, 1]
        cx, cy, bw, bh = box
        x = (cx - bw / 2) * image_pil.width
        y = (cy - bh / 2) * image_pil.height
        w = bw * image_pil.width
        h = bh * image_pil.height

        color = label_to_color[label]
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                  edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 5, f"{label}: {score:.2f}", color='white', fontsize=10,
                bbox=dict(facecolor=color, alpha=0.7, edgecolor='none'))

    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print(f"Saved visualization to {output_path}")
    plt.close()


def tokenize_captions(tokenizer, captions, device, max_length=256):
    """Tokenize a list of caption strings."""
    encoding = tokenizer(
        captions,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return encoding["input_ids"].to(device), encoding["attention_mask"].to(device)


def parse_caption(text: str):
    """
    Parse a caption string into individual class names.
    
    Examples:
        "person . car . dog ."  -> ["person", "car", "dog"]
        "a cat sitting on a chair" -> ["a cat sitting on a chair"]
    """
    text = text.strip()
    if "." in text:
        # Period-separated classes (COCO-style)
        parts = [p.strip() for p in text.split(".") if p.strip()]
        return parts
    return [text]


def main(args):
    device = torch.device(args.device)

    # Build model
    print("Loading model...")
    model = GroundingDINO().to(device)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, strict=False, verbose=False)
    else:
        print("WARNING: Using random weights!")

    model.eval()

    # Tokenizer from text backbone
    tokenizer = model.text_backbone.tokenizer

    # Load image
    print(f"Loading image: {args.image}")
    image_tensor, orig_size, image_pil = load_image(args.image, size=args.image_size)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    # Parse text prompt
    class_names = parse_caption(args.text)
    print(f"Class names: {class_names}")

    # Tokenize each class separately, then run inference
    all_boxes = []
    all_scores = []
    all_labels = []

    for class_name in class_names:
        input_ids, attention_mask = tokenize_captions(
            tokenizer, [class_name], device, max_length=256
        )

        with torch.no_grad():
            outputs = model(image_tensor, input_ids, attention_mask)

        scores = outputs["pred_logits"][0].sigmoid()  # [900]
        boxes = outputs["pred_boxes"][0]              # [900, 4]

        keep = scores > args.threshold
        if keep.sum() > 0:
            pred_boxes = boxes[keep].cpu()
            pred_scores = scores[keep].cpu()
            pred_labels = [class_name] * len(pred_boxes)

            all_boxes.append(pred_boxes)
            all_scores.append(pred_scores)
            all_labels.extend(pred_labels)

    if len(all_boxes) > 0:
        all_boxes = torch.cat(all_boxes, dim=0).numpy()
        all_scores = torch.cat(all_scores, dim=0).numpy()

        print(f"Found {len(all_boxes)} predictions above threshold {args.threshold}")
        for lbl, scr in zip(all_labels[:10], all_scores[:10]):
            print(f"  {lbl}: {scr:.3f}")

        draw_boxes(image_pil, all_boxes, all_labels, all_scores,
                   threshold=0, output_path=args.output)
    else:
        print("No predictions above threshold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pretrained_weights/groundingdino_swinb_cogcoor.pth",
                        help="Path to pretrained .pth")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--text", required=True,
                        help="Text prompt (e.g. 'person . car . dog .')")
    parser.add_argument("--output", default="output_vis.jpg", help="Output image path")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--image_size", type=int, default=800, help="Resize longest side to this")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    main(args)
