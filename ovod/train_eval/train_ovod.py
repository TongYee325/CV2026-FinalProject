"""
OVOD Training Script
====================
Team 1: Training loop for Open-Vocabulary Object Detection.

Datasets:
  - Pre-training: O365, GoldG, Caption data (optional)
  - Fine-tuning: COCO (if doing fine-tune setting)

Run this script independently. Do not modify core/ modules from here.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import sys

sys.path.insert(0, "../..")
from ovod.models.ovod_model import OVODGroundingDINO
from ovod.losses.ovod_losses import OVODLoss
from ovod.datasets.coco_eval import COCODataset
from configs.ovod_config import OVODConfig


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        
        # TODO: Tokenize text prompts from targets
        # text_inputs = tokenizer([t["text"] for t in targets])
        # outputs = model(images, text_inputs["input_ids"], text_inputs["attention_mask"])
        
        # TODO: Compute loss
        # losses = criterion(outputs, targets)
        
        # TODO: Backprop
        # optimizer.zero_grad()
        # losses["loss"].backward()
        # optimizer.step()
        
        total_loss += 0.0  # Placeholder
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Iter {i}, Loss: {0.0:.4f}")
    
    return total_loss / len(data_loader)


def main(args):
    config = OVODConfig()
    device = torch.device(args.device)
    
    # TODO: Build model
    model = OVODGroundingDINO(config).to(device)
    
    # TODO: Build criterion
    criterion = OVODLoss()
    
    # TODO: Build datasets and dataloaders
    # dataset = COCODataset(...)
    # data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    # TODO: Build optimizer
    # optimizer = torch.optim.AdamW(...)
    
    print("Starting OVOD training...")
    for epoch in range(config.num_epochs):
        # avg_loss = train_one_epoch(model, criterion, data_loader, optimizer, device, epoch)
        print(f"Epoch {epoch} completed.")
        
        # TODO: Save checkpoint
        # torch.save(model.state_dict(), f"ovod_checkpoint_epoch{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OVOD model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--output_dir", default="./ovod_outputs")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
