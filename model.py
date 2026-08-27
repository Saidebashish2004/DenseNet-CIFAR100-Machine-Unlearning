import torch.nn as nn
from torchvision.models import resnet18

def get_cifar_resnet18(num_classes=100):
    model = resnet18(weights=None)
    # Customize conv1 layer for 32x32 inputs
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() 
    # Set final linear output layer to match CIFAR-100's 100 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model