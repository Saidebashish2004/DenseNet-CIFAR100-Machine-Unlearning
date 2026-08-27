import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader

def get_cifar100_loaders(batch_size=128, forget_classes=list(range(10))):
    # Standard training data augmentations for CIFAR-100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Test data transformations (no random cropping/flipping)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    # Download/Load CIFAR-100
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

    # Separate dataset indices into Forget and Retain subsets based on class labels
    forget_indices = [i for i, label in enumerate(trainset.targets) if label in forget_classes]
    retain_indices = [i for i, label in enumerate(trainset.targets) if label not in forget_classes]

    forget_set = Subset(trainset, forget_indices)
    retain_set = Subset(trainset, retain_indices)

    # Wrap into PyTorch DataLoaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    retain_loader = DataLoader(retain_set, batch_size=batch_size, shuffle=True, num_workers=2)
    forget_loader = DataLoader(forget_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, retain_loader, forget_loader, test_loader