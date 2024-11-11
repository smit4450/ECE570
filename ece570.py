# -*- coding: utf-8 -*-

# Install torch and torchvision if not already installed (if using Colab)
!pip install torch torchvision
!pip install --upgrade sympy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNet Model
resnet = models.resnet50(pretrained=True)  # Pretrained ResNet
resnet.fc = nn.Linear(resnet.fc.in_features, 10)

# ViT Model
vit = models.vit_b_16(pretrained=True)  # Pretrained ViT model
vit.heads.head = nn.Linear(vit.heads.head.in_features, 10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)
vit.to(device)

def train_model(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Loss function and optimizers
criterion = nn.CrossEntropyLoss()
resnet_optimizer = optim.Adam(resnet.parameters(), lr=0.001)
vit_optimizer = optim.Adam(vit.parameters(), lr=0.001)

# Training loop
num_epochs = 5

# ResNet
print("Training ResNet...")
for epoch in range(num_epochs):
    resnet_loss = train_model(resnet, train_loader, criterion, resnet_optimizer)
    resnet_accuracy = evaluate_model(resnet, test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, ResNet Loss: {resnet_loss:.4f}, Accuracy: {resnet_accuracy:.2f}%")

# ViT
print("Training ViT...")
for epoch in range(num_epochs):
    vit_loss = train_model(vit, train_loader, criterion, vit_optimizer)
    vit_accuracy = evaluate_model(vit, test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, ViT Loss: {vit_loss:.4f}, Accuracy: {vit_accuracy:.2f}%")