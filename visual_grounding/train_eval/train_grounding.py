"""
Visual Grounding Training Script
=================================
Team 2: Training loop for Referring Expression Comprehension.

Datasets:
  - RefCOCO / RefCOCO+ / RefCOCOg
  - Flickr30K Entities (optional)

Run this script independently. Do not modify core/ modules from here.
"""

import torch
from torch.utils.data import DataLoader
import argparse
import os
import sys

sys.path.insert(0, "../..")
from visual_grounding.models.grounding_model import VisualGroundingDINO
from visual_grounding.losses.grounding_losses import GroundingLoss
from visual_grounding.datasets.refcoco import RefCOCODataset
from configs.grounding_config import GroundingConfig


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # TODO: Tokenize referring expressions
        # text_inputs = tokenizer([t["text"] for t in targets])
        # outputs = model(images, text_inputs["input_ids"], text_inputs["attention_mask"])
        
        # TODO: Compute loss and backprop
        
        total_loss += 0.0  # Placeholder
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iter {i}, Loss: {0.0:.4f}")
    
    return total_loss / len(data_loader)


def main(args):
    config = GroundingConfig()
    device = torch.device(args.device)
    
    # TODO: Build model
    model = VisualGroundingDINO(config).to(device)
    
    # TODO: Build criterion
    criterion = GroundingLoss()
    
    # TODO: Build datasets
    # dataset = RefCOCODataset(...)
    # data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # TODO: Build optimizer
    
    print("Starting Visual Grounding training...")
    for epoch in range(config.num_epochs):
        # avg_loss = train_one_epoch(model, criterion, data_loader, optimizer, device, epoch)
        print(f"Epoch {epoch} completed.")
        
        # TODO: Save checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Visual Grounding model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default="./grounding_outputs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
