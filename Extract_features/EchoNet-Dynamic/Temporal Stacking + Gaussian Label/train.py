import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import ssl
import random

# Bypass SSL verification on macOS
ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# 1. Custom Dataset for 2.5D frames
class EchoNet25DDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels_df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = self.labels_df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        # Convert to RGB (The 3 stacked grayscale channels will act as R, G, B)
        image = Image.open(img_path).convert('RGB')
        
        score = self.labels_df.iloc[idx, 1]
        score_tensor = torch.tensor([score], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, score_tensor

# 2. Data Preparation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = EchoNet25DDataset(csv_file='labels_25d.csv', img_dir='frames_25d')

# Data Split (80% Train, 20% Val)
num_samples = len(dataset)
indices = list(range(num_samples))
random.seed(42)
random.shuffle(indices)

train_size = int(0.8 * num_samples)
train_idx, val_idx = indices[:train_size], indices[train_size:]

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

# Apply specific transforms
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

# 3. Model Architecture (Freeze Bottom, Unfreeze Top)
class EchoNet25DModel(nn.Module):
    def __init__(self):
        super(EchoNet25DModel, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Phase 1: Freeze ALL layers
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Phase 2: Unfreeze Layer 3 and Layer 4
        for param in self.base_model.layer3.parameters():
            param.requires_grad = True
        for param in self.base_model.layer4.parameters():
            param.requires_grad = True
            
        # Phase 3: Replace FC layer (Automatically requires_grad = True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base_model(x)

model = EchoNet25DModel().to(device)

# 4. Loss and Optimizer
# Using SmoothL1Loss instead of MSE for better peak detection
criterion = nn.SmoothL1Loss()

# Only optimize parameters that require gradients (Layer 3, Layer 4, and FC)
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=1e-4)

# 5. Training Loop
num_epochs = 20
best_val_loss = float('inf')

print(f"Starting 2.5D Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    train_loss = running_loss / len(train_dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
            
    val_loss = val_loss / len(val_dataset)
    
    print(f"Epoch {epoch+1:02d}/{num_epochs} | Train SmoothL1: {train_loss:.4f} | Val SmoothL1: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_echonet_25d.pth')
        print(f"  -> Best model saved! (Val Loss: {best_val_loss:.4f})")

print(f"Training completed. Absolute Best Val Loss: {best_val_loss:.4f}.")