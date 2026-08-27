import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    forget_classes = list(range(10))
    retain_indices = [idx for idx, (_, label) in enumerate(trainset) if label not in forget_classes]
    retain_set = Subset(trainset, retain_indices)

    # Set num_workers=0 to avoid Windows multiprocessing spawning errors
    retain_loader = DataLoader(retain_set, batch_size=128, shuffle=True, num_workers=0)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    def get_densenet201():
        model = models.densenet201(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 100)
        return model.to(device)

    def train_model(model, dataloader, epochs=3, lr=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")
        return model

    os.makedirs("models", exist_ok=True)

    print("\n--- Training Base DenseNet-201 ---")
    base_model = get_densenet201()
    base_model = train_model(base_model, train_loader, epochs=2)
    torch.save(base_model.state_dict(), "models/base_densenet201.pth")

    print("\n--- Generating Unlearned DenseNet-201 ---")
    unlearned_model = get_densenet201()
    unlearned_model.load_state_dict(torch.load("models/base_densenet201.pth"))
    unlearned_model = train_model(unlearned_model, retain_loader, epochs=1, lr=0.001)
    torch.save(unlearned_model.state_dict(), "models/unlearned_densenet201.pth")

    print("\n--- Training Retrained DenseNet-201 ---")
    retrained_model = get_densenet201()
    retrained_model = train_model(retrained_model, retain_loader, epochs=2)
    torch.save(retrained_model.state_dict(), "models/retrained_densenet201.pth")

    print("\nAll DenseNet-201 models successfully trained and saved!")
