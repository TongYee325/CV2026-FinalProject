"""Shared utilities for RefCOCO-family visual grounding experiments."""

from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DATASET_CONFIGS = {
    "refcoco": {"split_by": "unc", "splits": ["val", "testA", "testB"]},
    "refcoco+": {"split_by": "unc", "splits": ["val", "testA", "testB"]},
    "refcocog": {"split_by": "umd", "splits": ["val", "test"]},
}


def normalize_expression(expression: str, append_period: bool = True) -> str:
    """Normalize whitespace and optionally add Grounding DINO's phrase delimiter."""
    expression = re.sub(r"\s+", " ", expression.strip())
    if append_period:
        expression = expression.rstrip(" .") + " ."
    return expression


def canonical_image_name(image_id: int) -> str:
    return f"COCO_train2014_{int(image_id):012d}.jpg"


def resolve_image_path(image_dir: Path, image_id: int) -> Path:
    return image_dir / canonical_image_name(image_id)


def resize_dimensions(width: int, height: int, short_side: int = 800,
                      max_side: int = 1333) -> Tuple[int, int]:
    scale = short_side / min(width, height)
    if max(width, height) * scale > max_side:
        scale = max_side / max(width, height)
    return int(width * scale), int(height * scale)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Apply the same resize and normalization used by the COCO evaluator."""
    width, height = image.size
    new_width, new_height = resize_dimensions(width, height)
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image)


def xywh_to_xyxy(box: Sequence[float]) -> List[float]:
    x, y, width, height = box
    return [float(x), float(y), float(x + width), float(y + height)]


def cxcywh_normalized_to_xyxy(
    boxes: torch.Tensor, image_width: int, image_height: int
) -> torch.Tensor:
    cx, cy, width, height = boxes.unbind(-1)
    return torch.stack([
        (cx - width / 2) * image_width,
        (cy - height / 2) * image_height,
        (cx + width / 2) * image_width,
        (cy + height / 2) * image_height,
    ], dim=-1)


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(
        0.0, float(box1[3]) - float(box1[1])
    )
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(
        0.0, float(box2[3]) - float(box2[1])
    )
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def load_dataset(refer_root: Path, dataset: str) -> Tuple[List[dict], Dict[int, dict]]:
    config = DATASET_CONFIGS[dataset]
    dataset_root = refer_root / dataset
    with (dataset_root / f"refs({config['split_by']}).p").open("rb") as handle:
        refs = pickle.load(handle)
    with (dataset_root / "instances.json").open() as handle:
        instances = json.load(handle)
    annotations = {int(annotation["id"]): annotation for annotation in instances["annotations"]}
    return refs, annotations


def flatten_sentences(refs: Iterable[dict], annotations: Dict[int, dict],
                      split: str) -> List[dict]:
    """Expand refs so each referring sentence is one evaluation sample."""
    samples = []
    for ref in refs:
        if ref["split"] != split:
            continue
        annotation = annotations.get(int(ref["ann_id"]))
        for sentence in ref.get("sentences", []):
            samples.append({
                "ref_id": int(ref["ref_id"]),
                "sent_id": int(sentence["sent_id"]),
                "ann_id": int(ref["ann_id"]),
                "image_id": int(ref["image_id"]),
                "category_id": int(ref["category_id"]),
                "expression": sentence["sent"],
                "gt_box_xywh": [float(value) for value in annotation["bbox"]]
                if annotation else None,
            })
    return samples


def deterministic_sample(samples: Sequence[dict], sample_count: int | None,
                         seed: int, key: str) -> List[dict]:
    """Return a stable sample independent of dataset iteration order."""
    ordered = sorted(samples, key=lambda item: (item["sent_id"], item["ref_id"]))
    if not sample_count or sample_count >= len(ordered):
        return ordered
    key_seed = seed + sum((index + 1) * ord(char) for index, char in enumerate(key))
    indices = sorted(random.Random(key_seed).sample(range(len(ordered)), sample_count))
    return [ordered[index] for index in indices]


def expression_tags(expression: str, gt_box_xywh: Sequence[float],
                    image_width: int, image_height: int) -> List[str]:
    text = expression.lower()
    tags = []
    if any(word in text for word in ("left", "right", "top", "bottom", "middle", "center")):
        tags.append("position")
    if any(word in text for word in (
        "red", "blue", "green", "yellow", "white", "black", "brown", "orange",
        "pink", "purple", "gray", "grey", "striped", "small", "large", "young", "old",
    )):
        tags.append("attribute")
    if any(word in text for word in (
        "holding", "wearing", "next to", "behind", "in front", "under", "above",
        "with", "beside", "near", "between", "on top",
    )):
        tags.append("relation")
    if any(word in text for word in (
        "hidden", "blocked", "occluded", "partially", "partly", "only visible",
        "cut off", "behind",
    )):
        tags.append("occlusion")
    if any(word in text for word in ("person", "man", "woman", "boy", "girl", "lady", "guy")):
        tags.append("person")
    area_ratio = gt_box_xywh[2] * gt_box_xywh[3] / (image_width * image_height)
    if area_ratio < 0.02:
        tags.append("small_target")
    return tags or ["other"]
