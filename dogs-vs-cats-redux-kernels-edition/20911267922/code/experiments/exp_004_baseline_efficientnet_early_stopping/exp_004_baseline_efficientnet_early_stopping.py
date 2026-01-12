import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from PIL import Image
import time
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
DATA_DIR = "/home/data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
SUBMISSION_DIR = "/home/submission"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Data loading
print("Loading data...")
train_files = os.listdir(TRAIN_DIR)
train_df = pd.DataFrame({
    'filename': train_files,
    'path': [os.path.join(TRAIN_DIR, f) for f in train_files]
})
train_df['label'] = train_df['filename'].apply(lambda x: 0 if x.startswith('cat') else 1)

print(f"Total training images: {len(train_df)}")
print(f"Cat images: {sum(train_df['label'] == 0)}")
print(f"Dog images: {sum(train_df['label'] == 1)}")
print(f"Class balance: {sum(train_df['label'] == 1) / len(train_df):.3f}")

# Split data
train_df, val_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df['label'], random_state=42
)
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Dataset class
class DogCatDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
train_dataset = DogCatDataset(train_df, transform=train_transform)
val_dataset = DogCatDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Model - EfficientNet-B0
print("\nCreating model...")
model = models.efficientnet_b0(pretrained=True)

# Modify classifier for binary classification
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_features, 1)
)

model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = torch.sigmoid(outputs) > 0.5
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return running_loss / total, correct / total

# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            preds = torch.sigmoid(outputs) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    val_loss = running_loss / total
    val_acc = correct / total
    
    # Clip predictions to avoid log(0) errors
    clipped_preds = np.clip(all_preds, 1e-7, 1-1e-7)
    val_log_loss = log_loss(all_labels, clipped_preds)
    
    return val_log_loss

# Training loop with early stopping
print("\nStarting training with early stopping...")
num_epochs = 50  # Increased max epochs to allow early stopping to trigger
best_val_log_loss = float('inf')
early_stopping_patience = 3
early_stopping_counter = 0
best_epoch = 0

for epoch in range(num_epochs):
    start_time = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_log_loss = validate(model, val_loader, criterion, device)
    
    # Use validation log loss for scheduling
    scheduler.step(val_log_loss)
    
    epoch_time = time.time() - start_time
    
    print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Log Loss: {val_log_loss:.6f}")
    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # Save best model based on log loss
    if val_log_loss < best_val_log_loss:
        best_val_log_loss = val_log_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), os.path.join(SUBMISSION_DIR, 'best_model.pth'))
        print(f"  → Best model saved (Val Log Loss: {val_log_loss:.6f})")
        early_stopping_counter = 0  # Reset counter on improvement
    else:
        early_stopping_counter += 1
        print(f"  → No improvement for {early_stopping_counter} epoch(s)")
    
    # Early stopping check
    if early_stopping_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs!")
        print(f"Best validation log loss: {best_val_log_loss:.6f} at epoch {best_epoch}")
        break

print(f"\nTraining completed! Best validation log loss: {best_val_log_loss:.6f} at epoch {best_epoch}")

# Load best model and predict on test set
print("\nLoading best model and predicting on test set...")
model.load_state_dict(torch.load(os.path.join(SUBMISSION_DIR, 'best_model.pth')))
model.eval()

# Test dataset
test_files = os.listdir(TEST_DIR)
test_df = pd.DataFrame({
    'filename': test_files,
    'path': [os.path.join(TEST_DIR, f) for f in test_files]
})

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.df.iloc[idx]['filename']

test_dataset = TestDataset(test_df, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Predictions
predictions = []
filenames = []

with torch.no_grad():
    for images, batch_filenames in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        predictions.extend(probs)
        filenames.extend(batch_filenames)

# Create submission
submission = pd.DataFrame({
    'id': [f.replace('.jpg', '') for f in filenames],
    'label': predictions
})
submission = submission.sort_values('id')
submission.to_csv(os.path.join(SUBMISSION_DIR, 'submission.csv'), index=False)

print(f"\nSubmission saved to {os.path.join(SUBMISSION_DIR, 'submission.csv')}")
print(f"Total predictions: {len(submission)}")
print(f"Prediction range: [{min(predictions):.4f}, {max(predictions):.4f}]")

# Final validation log loss
final_val_log_loss = validate(model, val_loader, criterion, device)
print(f"\nFinal Validation Log Loss: {final_val_log_loss:.6f}")
