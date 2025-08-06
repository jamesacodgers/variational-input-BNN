import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from datetime import datetime

from src.datasets.datasets import get_train_test_dataloaders
from src.training.training import train_model
from src.models.models import create_model

# Set random seed for reproducibility
torch.manual_seed(42)


# main.py - Your main script
import hydra
from omegaconf import DictConfig
import torch

@hydra.main(version_base=None, config_path="training_configs", config_name="toy")
def train(cfg: DictConfig) -> None:
    print("config:", cfg)
    print(f"Training {cfg.model.type} with hidden_size={cfg.model.hidden_size}")
    print(f"Learning rate: {cfg.training.lr}, Epochs: {cfg.training.epochs}")
    
    # Your existing training code here
    model = create_model(cfg.model.type, cfg.model.hidden_size)
    train_loader, test_loader = get_train_test_dataloaders(n_samples=256, device=device)

    train_model(model, train_dataloader=train_loader,  lr=cfg.training.lr, epochs=cfg.training.epochs)
    
    print("Training complete!")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    train()
    