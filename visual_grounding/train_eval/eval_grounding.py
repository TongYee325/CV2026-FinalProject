"""
Visual Grounding Evaluation Script
===================================
Team 2: Evaluate trained model on RefCOCO/+/g and Flickr30K.

Metric: Accuracy (predicted box IoU > 0.5 with GT)
Run independently after training.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import sys

sys.path.insert(0, "../..")
from visual_grounding.models.grounding_model import VisualGroundingDINO
from visual_grounding.datasets.refcoco import RefCOCODataset, RECAccuracyEvaluator
from configs.grounding_config import GroundingConfig


@torch.no_grad()
def evaluate_refcoco(model, dataset, device, config):
    """Evaluate on RefCOCO-style dataset."""
    model.eval()
    evaluator = RECAccuracyEvaluator()
    
    for images, targets in dataset:
        images = images.unsqueeze(0).to(device)
        text = targets["text"]
        gt_box = targets["boxes"].unsqueeze(0).to(device)
        
        # TODO: Tokenize text
        # outputs = model(images, text_input_ids, text_attention_mask)
        # preds = model.post_process_rec(outputs)
        # pred_box = preds[0]["box"].unsqueeze(0)
        
        # evaluator.update(pred_box, gt_box)
        pass
    
    results = evaluator.evaluate()
    print(f"REC Accuracy: {results['accuracy']:.4f}")
    return results


def main(args):
    config = GroundingConfig()
    device = torch.device(args.device)
    
    # TODO: Load model
    model = VisualGroundingDINO(config).to(device)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    if config.eval_refcoco:
        print("Evaluating on RefCOCO...")
        # dataset = RefCOCODataset(args.data_root, "test", "refcoco")
        # evaluate_refcoco(model, dataset, device, config)
    
    if config.eval_refcocop:
        print("Evaluating on RefCOCO+...")
        # dataset = RefCOCODataset(args.data_root, "test", "refcocop")
        # evaluate_refcoco(model, dataset, device, config)
    
    if config.eval_refcocog:
        print("Evaluating on RefCOCOg...")
        # dataset = RefCOCODataset(args.data_root, "test", "refcocog")
        # evaluate_refcoco(model, dataset, device, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Visual Grounding model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()
    
    main(args)
