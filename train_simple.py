"""
Training Script

"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pathlib import Path
import glob

torch.manual_seed(42)
np.random.seed(42)


class CelebrityDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.df = dataframe.copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Filter to only images that exist
        self.df = self.df[self.df['id'].apply(
            lambda x: (self.image_dir / f"{x}.jpg").exists()
        )].reset_index(drop=True)
        
        print(f"Dataset has {len(self.df)} images available")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_dir / f"{row['id']}.jpg"
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(row['label'] / 10.0, dtype=torch.float32)
        return image, label


class ImprovedAttractivenessNet(nn.Module):
    """Simplified architecture - less prone to overfitting"""
    
    def __init__(self):
        super(ImprovedAttractivenessNet, self).__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-20]:
            param.requires_grad = False
        
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.backbone(x)


def main():
    print("="*70)
    print("  SIMPLIFIED ATTRACTIVENESS MODEL TRAINING")
    print("="*70)
    
    # Find CSV
    csv_files = glob.glob('*.csv')
    rescaled = [f for f in csv_files if 'rescaled' in f.lower()]
    csv_file = rescaled[0] if rescaled else csv_files[0]
    
    print(f"\nUsing CSV: {csv_file}")
    
    # Detect delimiter
    with open(csv_file, 'r') as f:
        delimiter = ';' if ';' in f.readline() else ','
    
    df = pd.read_csv(csv_file, sep=delimiter)
    print(f"Loaded {len(df)} celebrities")
    print(f"Score range: {df['label'].min():.1f} - {df['label'].max():.1f}")
    print(f"Score mean: {df['label'].mean():.1f}")
    
    # Check image availability
    image_dir = Path('./celebrity_images')
    available = df['id'].apply(lambda x: (image_dir / f"{x}.jpg").exists()).sum()
    print(f"Images available: {available}/{len(df)}")
    
    if available < 100:
        print("\n⚠️  WARNING: Very few images available!")
        print("   Make sure celebrity_images/ folder has the downloaded images")
        return
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    full_dataset = CelebrityDataset(df, image_dir, transform=train_transform)
    
    # Split
    val_size = int(len(full_dataset) * 0.20)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform
    
    print(f"\nTrain: {train_size}, Val: {val_size}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = ImprovedAttractivenessNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Training
    print("\nTraining...")
    best_val_loss = float('inf')
    patience = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
            images, labels = images.to(device), labels.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.unsqueeze(1).to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        avg_val = val_loss / len(val_loader)
        
        history['train'].append(avg_train)
        history['val'].append(avg_val)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f}, Val={avg_val:.4f}")
        
        scheduler.step(avg_val)
        
        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': best_val_loss,
                'epoch': epoch
            }, './attractiveness_model_simple.pth')
            print(f"  → Saved (val_loss: {best_val_loss:.4f})")
        else:
            patience += 1
            if patience >= 7:
                print("\nEarly stopping")
                break
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./training_simple.png')
    print(f"\n✓ Saved: attractiveness_model_simple.pth")
    print(f"✓ Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
