import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

from dataset import get_cifar100_loaders
from model import get_cifar_resnet18

def train_model(model, dataloader, epochs=20, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
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
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Accuracy: {100.*correct/total:.2f}%")
        
    return model

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Load data loaders
    train_loader, _, _, _ = get_cifar100_loaders(batch_size=128)
    
    # 2. Initialize CNN model
    model = get_cifar_resnet18()
    
    # 3. Train Base Model
    print("\n--- Training Base Model on Full CIFAR-100 Dataset ---")
    model = train_model(model, train_loader, epochs=20, device=device)
    
    # 4. Save model weights
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/base_model.pth')
    print("\nSuccess! Base model saved to checkpoints/base_model.pth")  