import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. Setup and Settings
DATA_DIR = 'Annotated_Images'
BATCH_SIZE = 16  # How many images it looks at before updating its brain
EPOCHS = 10      # How many times it loops through the entire dataset

# Check hardware acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training using device: {device}")

# 2. Prepare the Image with data augmentation
# ResNet18 expects images to be a specific size and format but we apply little chaos to the training images 
# This way it learns to generalize
train_transforms = transforms.Compose([
    transforms.Resize((230, 230)),
    transforms.RandomCrop(224),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the folder twice with different rules to ensure honest testing
base_train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
base_val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

total_size = len(base_train_dataset)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

# Split the dataset
indices = torch.randperm(total_size).tolist()
train_dataset = torch.utils.data.Subset(base_train_dataset, indices[val_size:])
val_dataset = torch.utils.data.Subset(base_val_dataset, indices[:val_size])

# The Conveyor Belts
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"📸 Found {total_size} total images. Training on {train_size}, Testing on {val_size}.")

# 3. Build the Brain (ResNet18)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10) 
model = model.to(device)

# 4. Define How It Learns
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. The Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Test Phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    val_accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Validation Accuracy: {val_accuracy:.2f}%")

# 6. Save the Brain
# torch.save(model.state_dict(), 'water_meter_brain.pth')
# print("✅ Training Complete! Model saved as 'water_meter_brain.pth'")