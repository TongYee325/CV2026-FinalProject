"""Compare model predictions with COCO GT for a single image."""
import sys
sys.path.insert(0, ".")

import os
import argparse
import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO

from grounding_dino_v2 import GroundingDINOV2
from tools.load_checkpoint_v2 import load_checkpoint_v2
from tools.eval_coco import build_caption_and_positive_map, COCO_CLASSES, COCO_CATEGORY_IDS, token_idx_to_class


def load_image(path, device):
    image = Image.open(path).convert("RGB")
    orig_w, orig_h = image.size
    scale = 800 / min(orig_w, orig_h)
    if max(orig_w, orig_h) * scale > 1333:
        scale = 1333 / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device), (orig_w, orig_h)


def cxcywh_to_xyxy_abs(boxes, orig_w, orig_h):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    return torch.stack([x1, y1, x2, y2], dim=-1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", type=int, default=139)
    parser.add_argument("--checkpoint", default="pretrained_weights/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--ann-file", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--image-dir", default="data/coco/val2017")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = GroundingDINOV2().to(device)
    load_checkpoint_v2(model, args.checkpoint, verbose=False)
    model.eval()

    coco_gt = COCO(args.ann_file)
    img_info = coco_gt.loadImgs(args.image_id)[0]
    file_name = img_info["file_name"]
    orig_w, orig_h = img_info["width"], img_info["height"]
    img_path = os.path.join(args.image_dir, file_name)

    img_tensor, _ = load_image(img_path, device)

    # Print GT
    print(f"\nImage {args.image_id}: {file_name} ({orig_w}x{orig_h})")
    ann_ids = coco_gt.getAnnIds(imgIds=args.image_id)
    anns = coco_gt.loadAnns(ann_ids)
    print(f"\nGround truth ({len(anns)} objects):")
    id_to_name = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    for ann in anns:
        x, y, w, h = ann["bbox"]
        print(f"  {id_to_name[ann['category_id']]:15s} [{x:.1f}, {y:.1f}, {x+w:.1f}, {y+h:.1f}]")

    # Simple prompt predictions
    print(f"\n--- Simple prompts (threshold={args.threshold}) ---")
    for prompt in ["person", "bicycle", "car", "chair", "tv", "dog", "cat"]:
        results = model.predict(img_tensor, [prompt], confidence_threshold=args.threshold)
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        if len(boxes) == 0:
            print(f"  {prompt}: no detections")
            continue
        boxes_xyxy = cxcywh_to_xyxy_abs(boxes, orig_w, orig_h).cpu()
        print(f"  {prompt}: top-{min(3, len(boxes))} boxes")
        for j in range(min(3, len(boxes))):
            x1, y1, x2, y2 = boxes_xyxy[j].tolist()
            print(f"    [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score={scores[j].item():.3f}")

    # Full COCO caption predictions
    print(f"\n--- Full COCO caption (threshold={args.threshold}) ---")
    caption, positive_map = build_caption_and_positive_map(model.tokenizer, COCO_CLASSES)
    results = model.predict(img_tensor, [caption], confidence_threshold=args.threshold)
    boxes = results[0]["boxes"]
    scores = results[0]["scores"]
    labels = results[0]["labels"]
    if len(boxes) > 0:
        boxes_xyxy = cxcywh_to_xyxy_abs(boxes, orig_w, orig_h).cpu()
        print(f"Total detections: {len(boxes)}")
        for j in range(min(20, len(boxes))):
            token_idx = int(labels[j].item())
            cls_idx = token_idx_to_class(token_idx, positive_map)
            cls_name = COCO_CLASSES[cls_idx] if cls_idx >= 0 else "<unmapped>"
            x1, y1, x2, y2 = boxes_xyxy[j].tolist()
            print(f"  {cls_name:15s} [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] score={scores[j].item():.3f}")


if __name__ == "__main__":
    main()
