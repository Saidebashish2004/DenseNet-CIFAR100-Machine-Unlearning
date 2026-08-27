import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from dataset import get_cifar100_loaders
from model import get_cifar_resnet18
from train_base import train_model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load data loaders - Using only the retain set
    _, retain_loader, _, _ = get_cifar100_loaders(batch_size=128)
    
    # 2. Initialize a fresh CNN model from scratch
    retrained_model = get_cifar_resnet18()
    
    # 3. Train on Retain Set ONLY (20 epochs)
    print("\n--- Training Ground-Truth Retrained Model on Retain Set Only ---")
    retrained_model = train_model(retrained_model, retain_loader, epochs=20, device=device)
    
    # 4. Save model weights
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(retrained_model.state_dict(), 'checkpoints/retrained_model.pth')
    print("\nSuccess! Retrained model saved to checkpoints/retrained_model.pth")