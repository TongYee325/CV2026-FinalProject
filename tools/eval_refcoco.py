"""
RefCOCO / RefCOCO+ / RefCOCOg Evaluation for GroundingDINO V2
===============================================================
Metric: Accuracy@0.5 (top-1 predicted box IoU > 0.5 with GT)
"""
import sys
sys.path.insert(0, ".")

import os
import json
import pickle
import argparse
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from grounding_dino_v2 import GroundingDINOV2
from tools.load_checkpoint_v2 import load_checkpoint_v2


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union == 0:
        return 0.0
    return inter / union


def cxcywh_to_xyxy(boxes, img_w, img_h):
    """Convert normalized [cx, cy, w, h] to absolute [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def load_refer_data(refer_root, dataset, split_by):
    """Load referring expression data."""
    refs_path = os.path.join(refer_root, dataset, f"refs({split_by}).p")
    instances_path = os.path.join(refer_root, dataset, "instances.json")

    with open(refs_path, "rb") as f:
        refs_data = pickle.load(f)

    with open(instances_path, "r") as f:
        instances = json.load(f)

    # Build annotation lookup
    anns = {ann["id"]: ann for ann in instances["annotations"]}
    imgs = {img["id"]: img for img in instances["images"]}

    return refs_data, anns, imgs


@torch.no_grad()
def evaluate_refcoco(model, device, refer_root, dataset, split_by, split,
                     confidence_threshold=0.0, top_k=1):
    """
    Evaluate on a RefCOCO split.
    confidence_threshold: minimum score to keep a box (0 = keep all)
    top_k: use top-k boxes by score for matching (usually 1)
    """
    model.eval()

    refs_data, anns, imgs = load_refer_data(refer_root, dataset, split_by)

    # Filter to desired split
    refs = [r for r in refs_data if r["split"] == split]
    print(f"{dataset} ({split_by}) {split}: {len(refs)} expressions")

    transform = transforms.Compose([transforms.ToTensor()])

    correct = 0
    total = 0

    for ref in tqdm(refs, desc=f"{dataset}/{split}"):
        ann_id = ref["ann_id"]
        if ann_id not in anns:
            continue

        ann = anns[ann_id]
        img_id = ann["image_id"]
        if img_id not in imgs:
            continue

        img_info = imgs[img_id]
        file_name = img_info["file_name"]
        img_w, img_h = img_info["width"], img_info["height"]

        # Image path varies by dataset
        if dataset == "refcocog":
            img_dir = os.path.join(refer_root, dataset, "images")
        else:
            img_dir = os.path.join(refer_root, dataset, "images", f"{dataset}_{split_by}")

        img_path = os.path.join(img_dir, file_name)
        if not os.path.exists(img_path):
            # Try alternate path structure
            img_path = os.path.join(refer_root, dataset, "images", file_name)
        if not os.path.exists(img_path):
            continue

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)

        # Use first sentence
        sentence = ref["sentences"][0]["sent"]

        # Run inference
        outputs = model.predict(img_tensor, [sentence], confidence_threshold=confidence_threshold)

        boxes = outputs[0]["boxes"]     # [nq, 4] normalized cxcywh
        scores = outputs[0]["scores"]   # [nq]

        if len(boxes) == 0:
            total += 1
            continue

        # Convert to absolute xyxy
        boxes_xyxy = cxcywh_to_xyxy(boxes, img_w, img_h).cpu().numpy()

        # GT box
        gt_box = ann["bbox"]  # [x, y, w, h] COCO format
        gt_xyxy = [
            gt_box[0],
            gt_box[1],
            gt_box[0] + gt_box[2],
            gt_box[1] + gt_box[3],
        ]

        # Match: use top-k boxes by score, check if any IoU > 0.5
        if top_k > 0 and len(boxes_xyxy) > top_k:
            top_idx = torch.topk(scores, min(top_k, len(scores)))[1]
            boxes_xyxy = boxes_xyxy[top_idx.cpu().numpy()]

        max_iou = 0.0
        for pred_box in boxes_xyxy:
            iou = compute_iou(pred_box, gt_xyxy)
            max_iou = max(max_iou, iou)

        total += 1
        if max_iou >= 0.5:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    print(f"\n{dataset} ({split_by}) {split} Results:")
    print(f"  Correct: {correct} / {total}")
    print(f"  Accuracy@0.5: {acc * 100:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pretrained_weights/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--refer-root", default="data/refer")
    parser.add_argument("--dataset", default="all", choices=["all", "refcoco", "refcoco+", "refcocog"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = GroundingDINOV2().to(device)
    load_checkpoint_v2(model, args.checkpoint, verbose=True)
    model.eval()

    # Evaluation configs: (dataset, split_by, split)
    configs = []
    if args.dataset in ("all", "refcoco"):
        configs.extend([
            ("refcoco", "unc", "val"),
            ("refcoco", "unc", "testA"),
            ("refcoco", "unc", "testB"),
        ])
    if args.dataset in ("all", "refcoco+"):
        configs.extend([
            ("refcoco+", "unc", "val"),
            ("refcoco+", "unc", "testA"),
            ("refcoco+", "unc", "testB"),
        ])
    if args.dataset in ("all", "refcocog"):
        configs.extend([
            ("refcocog", "google", "val"),
            ("refcocog", "umd", "val"),
            ("refcocog", "umd", "test"),
        ])

    results = {}
    for dataset, split_by, split in configs:
        print(f"\n{'='*60}")
        acc = evaluate_refcoco(
            model, device, args.refer_root, dataset, split_by, split,
            confidence_threshold=args.threshold
        )
        results[f"{dataset}_{split_by}_{split}"] = acc

    print(f"\n{'='*60}")
    print("Summary:")
    for key, acc in results.items():
        print(f"  {key}: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
