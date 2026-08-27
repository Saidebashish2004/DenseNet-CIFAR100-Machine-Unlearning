import torch
import torch.nn as nn
from dataset import get_cifar100_loaders
from model import get_cifar_resnet18
import os

def evaluate_model(model, dataloader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    return 100. * correct / total

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load Data loaders
    _, retain_loader, forget_loader, test_loader = get_cifar100_loaders(batch_size=128)
    
    # 2. Checkpoint Paths
    paths = {
        "Base Model": 'checkpoints/base_model.pth',
        "Unlearned Model": 'checkpoints/unlearned_model.pth',
        "Retrained Model (Gold Standard)": 'checkpoints/retrained_model.pth'
    }
    
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"Error: Missing checkpoint {path}")
            exit()
            
    print("\n================ COMPREHENSIVE UNLEARNING EVALUATION ================")
    print(f"{'Model Name':<35} | {'Retain Acc (%)':<15} | {'Forget Acc (%)':<15}")
    print("-" * 73)
    
    for name, path in paths.items():
        model = get_cifar_resnet18()
        model.load_state_dict(torch.load(path, map_location=device))
        
        retain_acc = evaluate_model(model, retain_loader, device)
        forget_acc = evaluate_model(model, forget_loader, device)
        
        print(f"{name:<35} | {retain_acc:<15.2f} | {forget_acc:<15.2f}")
        
    print("======================================================================")