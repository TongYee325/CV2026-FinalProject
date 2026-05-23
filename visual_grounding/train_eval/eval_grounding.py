"""
Visual Grounding Evaluation Script
===================================
Team 2: Evaluate pretrained model on RefCOCO/+/g.

Run zero-shot or fine-tuned evaluation using official pretrained weights.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from visual_grounding.models.grounding_model import VisualGroundingDINO
from visual_grounding.datasets.refcoco import RefCOCODataset, RECAccuracyEvaluator
from configs.grounding_config import GroundingConfig
from tools.load_checkpoint import load_grounding_dino_checkpoint


@torch.no_grad()
def evaluate_refcoco(model, dataset, device, config, dataset_name="refcoco"):
    """Evaluate on RefCOCO-style dataset."""
    model.eval()
    evaluator = RECAccuracyEvaluator()

    for i, (image, target) in enumerate(dataset):
        image = image.unsqueeze(0).to(device)
        text = target["text"]
        gt_box = target["boxes"].to(device)  # [1, 4]

        # Tokenize referring expression
        encoding = model.model.text_backbone.tokenize([text], device=str(device))
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        outputs = model.model(image, input_ids, attention_mask)
        preds = model.post_process_rec(outputs)
        pred_box = preds[0]["box"].unsqueeze(0).to(device)  # [1, 4]

        evaluator.update(pred_box, gt_box)

        if i % 100 == 0:
            print(f"[{dataset_name}] Processed {i}/{len(dataset)} samples")

    results = evaluator.evaluate()
    print(f"\n{dataset_name} Accuracy: {results['accuracy']:.4f}")
    return results


def main(args):
    config = GroundingConfig()
    device = torch.device(args.device)

    # Build model
    model = VisualGroundingDINO(config).to(device)

    # Load pretrained weights
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        load_grounding_dino_checkpoint(model.model, args.checkpoint, strict=False)
    else:
        print("WARNING: No checkpoint provided — using random weights!")

    if config.eval_refcoco:
        print("\nEvaluating on RefCOCO...")
        # dataset = RefCOCODataset(args.data_root, "test", "refcoco")
        # evaluate_refcoco(model, dataset, device, config, "RefCOCO")
        print("(RefCOCO evaluation not yet implemented — finish dataset loader first)")

    if config.eval_refcocop:
        print("\nEvaluating on RefCOCO+...")
        # evaluate_refcoco(model, dataset, device, config, "RefCOCO+")

    if config.eval_refcocog:
        print("\nEvaluating on RefCOCOg...")
        # evaluate_refcoco(model, dataset, device, config, "RefCOCOg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Visual Grounding model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    main(args)
