"""
COCO val2017 Evaluation for GroundingDINO V2
=============================================
Evaluates AP / AP50 / AP75 using pycocotools.
"""
import sys
sys.path.insert(0, ".")

import os
import json
import argparse
import pickle
from tqdm import tqdm
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from grounding_dino_v2 import GroundingDINOV2
from tools.load_checkpoint_v2 import load_checkpoint_v2


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

# COCO 2017 category IDs for the 80 classes above (not contiguous 1-80).
COCO_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def build_caption_and_positive_map(tokenizer, classes):
    """Build caption string and token->class mapping."""
    caption = " . ".join(classes)
    # Tokenize
    tokens = tokenizer(caption, return_tensors="pt")
    input_ids = tokens["input_ids"][0].tolist()

    # Build positive map: for each class, which token positions belong to it
    # Strategy: tokenize each class individually and match subsequences
    positive_map = {}  # class_idx -> list of token positions

    # Tokenize each class to find its token sequence
    class_token_ids = []
    for cls_name in classes:
        cls_tokens = tokenizer(cls_name, add_special_tokens=False)["input_ids"]
        class_token_ids.append(cls_tokens)

    # Match each class's token sequence in the full caption
    # The caption is: cls0 [1012] cls1 [1012] ... [1012] cls79
    # where 1012 is the period token id
    # We'll use a simple sliding window approach
    for cls_idx, cls_toks in enumerate(class_token_ids):
        positions = []
        for i in range(len(input_ids) - len(cls_toks) + 1):
            if input_ids[i:i + len(cls_toks)] == cls_toks:
                # Check boundaries: not inside another word
                # Simple heuristic: accept all matches
                positions.extend(range(i, i + len(cls_toks)))
        # Remove duplicates while preserving order
        seen = set()
        unique_positions = []
        for p in positions:
            if p not in seen:
                seen.add(p)
                unique_positions.append(p)
        positive_map[cls_idx] = unique_positions

    return caption, positive_map


def token_idx_to_class(token_idx, positive_map):
    """Map a token index to its COCO class index (0-based)."""
    for cls_idx, positions in positive_map.items():
        if token_idx in positions:
            return cls_idx
    return -1


def bbox_cxcywh_to_xywh(boxes):
    """Convert [cx, cy, w, h] to [x, y, w, h]."""
    x = boxes[:, 0] - boxes[:, 2] / 2
    y = boxes[:, 1] - boxes[:, 3] / 2
    w = boxes[:, 2]
    h = boxes[:, 3]
    return torch.stack([x, y, w, h], dim=-1)


@torch.no_grad()
def evaluate_coco(model, tokenizer, coco_gt, image_dir, caption, positive_map,
                    device, confidence_threshold=0.05, max_images=None):
    """Run inference and return COCO-format results."""
    model.eval()

    img_ids = sorted(coco_gt.getImgIds())
    if max_images:
        img_ids = img_ids[:max_images]

    results = []

    for img_id in tqdm(img_ids, desc="COCO eval"):
        img_info = coco_gt.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        img_path = os.path.join(image_dir, file_name)

        # Load image
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Resize to standard size: shorter side = 800, longer side <= 1333
        scale = 800 / min(orig_w, orig_h)
        if max(orig_w, orig_h) * scale > 1333:
            scale = 1333 / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Official GroundingDINO uses ImageNet normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Run inference
        outputs = model.predict(img_tensor, [caption], confidence_threshold=confidence_threshold)

        boxes = outputs[0]["boxes"]     # [nq, 4] in cxcywh normalized
        scores = outputs[0]["scores"]   # [nq]
        labels = outputs[0]["labels"]   # [nq] token indices

        # Convert to COCO format
        # COCO expects [x, y, w, h] in absolute pixels
        if len(boxes) > 0:
            boxes_abs = bbox_cxcywh_to_xywh(boxes)
            boxes_abs[:, [0, 2]] *= orig_w
            boxes_abs[:, [1, 3]] *= orig_h

            for j in range(len(boxes)):
                token_idx = int(labels[j].item())
                cls_idx = token_idx_to_class(token_idx, positive_map)
                if cls_idx < 0:
                    continue

                # Map class index to actual COCO 2017 category ID
                category_id = COCO_CATEGORY_IDS[cls_idx]

                x, y, w, h = boxes_abs[j].cpu().tolist()
                results.append({
                    "image_id": img_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "score": float(scores[j].item()),
                })

    return results, img_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pretrained_weights/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--coco-dir", default="data/coco")
    parser.add_argument("--ann-file", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image-dir", default="data/coco/val2017")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--output", default="results/coco_predictions.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load model
    print("Loading model...")
    model = GroundingDINOV2().to(device)
    load_checkpoint_v2(model, args.checkpoint, verbose=True)
    model.eval()

    # 2. Build caption and positive map
    caption, positive_map = build_caption_and_positive_map(model.tokenizer, COCO_CLASSES)
    print(f"Caption length: {len(caption)} chars")
    print(f"Positive map: {len(positive_map)} classes mapped")

    # Save positive map for debugging
    with open("tools/coco_positive_map.pkl", "wb") as f:
        pickle.dump({"caption": caption, "positive_map": positive_map}, f)

    # 3. Load COCO ground truth
    print("Loading COCO annotations...")
    coco_gt = COCO(args.ann_file)

    # 4. Run evaluation
    print(f"Running inference on COCO val2017 (threshold={args.threshold})...")
    results, eval_img_ids = evaluate_coco(
        model, model.tokenizer, coco_gt, args.image_dir,
        caption, positive_map, device,
        confidence_threshold=args.threshold,
        max_images=args.max_images,
    )

    # 5. Save results
    print(f"Saving {len(results)} predictions to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f)

    # 6. Evaluate with pycocotools
    print("Evaluating with pycocotools...")
    coco_dt = coco_gt.loadRes(args.output)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.params.imgIds = eval_img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print key metrics
    print("\n=== COCO Results ===")
    print(f"  AP     : {coco_eval.stats[0]:.3f}")
    print(f"  AP50   : {coco_eval.stats[1]:.3f}")
    print(f"  AP75   : {coco_eval.stats[2]:.3f}")
    print(f"  APs    : {coco_eval.stats[3]:.3f}")
    print(f"  APm    : {coco_eval.stats[4]:.3f}")
    print(f"  APl    : {coco_eval.stats[5]:.3f}")


if __name__ == "__main__":
    main()
