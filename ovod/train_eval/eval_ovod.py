"""
OVOD Evaluation Script
======================
Team 1: Evaluate pretrained model on COCO, LVIS.

Supports zero-shot transfer using official pretrained weights.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from ovod.models.ovod_model import OVODGroundingDINO
from ovod.datasets.coco_eval import COCODataset, COCOEvaluator
from configs.ovod_config import OVODConfig
from tools.load_checkpoint import load_grounding_dino_checkpoint


@torch.no_grad()
def evaluate_coco(model, dataset, device, config):
    """Evaluate on COCO dataset (zero-shot)."""
    model.eval()

    # Get text prompt for all COCO categories
    text_prompt = dataset.get_text_prompt()
    print(f"Text prompt (first 100 chars): {text_prompt[:100]}...")

    # Tokenize once (same categories for all images in zero-shot)
    encoding = model.model.text_backbone.tokenize([text_prompt], device=str(device))
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    evaluator = COCOEvaluator(None)

    for i, (image, target) in enumerate(dataset):
        image = image.unsqueeze(0).to(device)

        outputs = model.model(image, input_ids, attention_mask)

        # TODO: Post-process outputs to COCO format
        # predictions = model.post_process(outputs, category_mapping)
        # evaluator.update(predictions)

        if i % 100 == 0:
            print(f"Processed {i}/{len(dataset)} images")

    results = evaluator.evaluate()
    print("\nCOCO Zero-Shot Evaluation Results:")
    print(f"  AP:   {results.get('AP', 0.0):.3f}")
    print(f"  AP50: {results.get('AP50', 0.0):.3f}")
    print(f"  AP75: {results.get('AP75', 0.0):.3f}")
    return results


def main(args):
    config = OVODConfig()
    device = torch.device(args.device)

    # Build model
    model = OVODGroundingDINO(config).to(device)

    # Load pretrained weights
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_grounding_dino_checkpoint(model.model, args.checkpoint, strict=False)
    else:
        print("WARNING: No checkpoint provided — using random weights!")

    # TODO: Load datasets
    # coco_dataset = COCODataset(
    #     img_dir=f"{args.data_root}/coco/val2017",
    #     ann_file=f"{args.data_root}/coco/annotations/instances_val2017.json",
    # )

    if config.eval_coco:
        print("\nEvaluating on COCO...")
        # evaluate_coco(model, coco_dataset, device, config)
        print("(COCO evaluation not yet implemented — finish dataset loader first)")

    if config.eval_lvis:
        print("\nEvaluating on LVIS...")
        print("(LVIS evaluation not yet implemented)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OVOD model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    main(args)
