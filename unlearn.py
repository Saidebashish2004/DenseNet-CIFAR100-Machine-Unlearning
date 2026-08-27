import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from dataset import get_cifar100_loaders
from model import get_cifar_resnet18

def amnesiac_unlearn(model, retain_loader, epochs=5, device='cpu'):
    """
    Standard Baseline: Fine-tune on the retain set for a few epochs 
    to induce catastrophic forgetting of the absent classes.
    """
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    # Using a slightly higher learning rate to encourage overwriting old weights
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(retain_loader, desc=f"Unlearning Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        print(f"Unlearn Epoch {epoch+1}/{epochs} | Retain Loss: {running_loss/len(retain_loader):.4f} | Retain Acc: {100.*correct/total:.2f}%")
        
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load data
    _, retain_loader, _, _ = get_cifar100_loaders(batch_size=128)
    
    # 2. Initialize model and load the pre-trained base weights
    unlearned_model = get_cifar_resnet18()
    
    base_model_path = 'checkpoints/base_model.pth'
    if os.path.exists(base_model_path):
        unlearned_model.load_state_dict(torch.load(base_model_path, map_location=device))
        print("Successfully loaded base model weights.")
    else:
        print(f"Error: {base_model_path} not found. Please finish training base model first.")
        exit()
        
    # 3. Apply Unlearning
    print("\n--- Starting Amnesiac Unlearning (Fine-Tuning on Retain Set) ---")
    unlearned_model = amnesiac_unlearn(unlearned_model, retain_loader, epochs=5, device=device)
    
    # 4. Save unlearned model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(unlearned_model.state_dict(), 'checkpoints/unlearned_model.pth')
    print("\nSuccess! Unlearned model saved to checkpoints/unlearned_model.pth")