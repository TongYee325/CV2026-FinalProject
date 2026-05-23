"""
OVOD Evaluation Script
======================
Team 1: Evaluate trained model on COCO, LVIS, ODinW.

Supports both zero-shot transfer and fine-tuned evaluation.
Run independently after training.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import sys

sys.path.insert(0, "../..")
from ovod.models.ovod_model import OVODGroundingDINO
from ovod.datasets.coco_eval import COCODataset, COCOEvaluator
from configs.ovod_config import OVODConfig


@torch.no_grad()
def evaluate_coco(model, dataset, device, config):
    """Evaluate on COCO dataset (zero-shot or fine-tuned)."""
    model.eval()
    
    # Get text prompt for all COCO categories
    text_prompt = dataset.get_text_prompt()
    print(f"Text prompt: {text_prompt}")
    
    # TODO: Tokenize text prompt
    # TODO: Run inference on all images
    # TODO: Collect predictions
    
    evaluator = COCOEvaluator(None)
    results = evaluator.evaluate()
    
    print("COCO Evaluation Results:")
    print(f"  AP: {results['AP']:.3f}")
    print(f"  AP50: {results['AP50']:.3f}")
    print(f"  AP75: {results['AP75']:.3f}")
    return results


@torch.no_grad()
def evaluate_lvis(model, dataset, device, config):
    """Evaluate on LVIS dataset."""
    model.eval()
    print("LVIS evaluation not yet implemented.")
    return {}


def main(args):
    config = OVODConfig()
    device = torch.device(args.device)
    
    # TODO: Load model checkpoint
    model = OVODGroundingDINO(config).to(device)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # TODO: Load datasets
    # coco_dataset = COCODataset(...)
    
    if config.eval_coco:
        print("Evaluating on COCO...")
        # evaluate_coco(model, coco_dataset, device, config)
    
    if config.eval_lvis:
        print("Evaluating on LVIS...")
        # evaluate_lvis(model, lvis_dataset, device, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OVOD model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()
    
    main(args)
