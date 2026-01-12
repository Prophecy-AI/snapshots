import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import functional as TF
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
import os
from PIL import Image
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Enhanced data augmentation with CutMix and MixUp
class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch, targets):
        if self.alpha <= 0:
            return batch, targets
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        y1, y2 = targets, targets[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch.size(), lam)
        
        batch[:, :, bby1:bby2, bbx1:bbx2] = batch[index, :, bby1:bby2, bbx1:bbx2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch.size()[-1] * batch.size()[-2]))
        
        return batch, (y1, y2, lam)
    
    def rand_bbox(self, size, lam):
        W, H = size[-1], size[-2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, batch, targets):
        if self.alpha <= 0:
            return batch, targets
        
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = batch.size(0)
        index = torch.randperm(batch_size)
        
        mixed_batch = lam * batch + (1 - lam) * batch[index]
        y1, y2 = targets, targets[index]
        
        return mixed_batch, (y1, y2, lam)

class EnhancedAugmentation:
    def __init__(self, image_size=300):
        self.image_size = image_size
        
        # Training transforms with RandAugment-like policy
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Validation transforms
        self.val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_train_transform(self):
        return self.train_transform
    
    def get_val_transform(self):
        return self.val_transform

class DogsCatsDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def load_best_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

# Label smoothing loss
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        pred = pred.clamp(1e-7, 1-1e-7)
        
        if isinstance(target, tuple):  # For CutMix/MixUp
            y1, y2, lam = target
            loss = lam * (-y1 * torch.log(pred) - (1-y1) * torch.log(1-pred)).mean() + \
                   (1-lam) * (-y2 * torch.log(pred) - (1-y2) * torch.log(1-pred)).mean()
        else:
            # Standard label smoothing
            smoothed_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            loss = -smoothed_target * torch.log(pred) - (1 - smoothed_target) * torch.log(1 - pred)
            loss = loss.mean()
        
        return loss

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    
    image_paths = []
    labels = []
    
    for filename in os.listdir(train_dir):
        if filename.startswith('cat'):
            labels.append(0)
        elif filename.startswith('dog'):
            labels.append(1)
        else:
            continue
        image_paths.append(os.path.join(train_dir, filename))
    
    return image_paths, labels

def create_model():
    # Load EfficientNet-B3 pretrained on ImageNet
    model = models.efficientnet_b3(pretrained=True)
    
    # Modify the classifier for binary classification
    num_features = model.classifier[1].in_features
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),  # Increased dropout for better regularization
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    return model

def train_epoch(model, train_loader, criterion, optimizer, device, scaler, cutmix, mixup, use_augmentation=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Apply CutMix or MixUp with 50% probability
        if use_augmentation and random.random() < 0.5:
            if random.random() < 0.5:
                inputs, targets = cutmix(inputs, targets)
            else:
                inputs, targets = mixup(inputs, targets)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        with torch.no_grad():
            predicted = (outputs.squeeze() > 0.5).float()
            if isinstance(targets, tuple):
                y1, y2, lam = targets
                targets_for_acc = lam * y1 + (1 - lam) * y2
            else:
                targets_for_acc = targets
            total += targets_for_acc.size(0)
            correct += (predicted == targets_for_acc).sum().item()
        
        pbar.set_postfix({
            'Loss': f'{running_loss/(batch_idx+1):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(train_loader), correct / total

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            
            running_loss += loss.item()
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate log loss
    log_loss_score = log_loss(all_targets, np.clip(all_preds, 1e-7, 1-1e-7))
    
    return running_loss / len(val_loader), log_loss_score

def predict_test(model, test_dir, transform, device):
    model.eval()
    predictions = []
    image_ids = []
    
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')],
                       key=lambda x: int(x.split('.')[0]))
    
    with torch.no_grad():
        for filename in tqdm(test_files, desc='Predicting on test set'):
            img_path = os.path.join(test_dir, filename)
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            output = model(image)
            pred = output.squeeze().item()
            predictions.append(pred)
            image_ids.append(int(filename.split('.')[0]))
    
    return predictions, image_ids

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    data_dir = '/home/data'
    print("\nLoading data...")
    image_paths, labels = load_data(data_dir)
    
    print(f"Total training images: {len(image_paths)}")
    print(f"Cat images: {sum(1 for l in labels if l == 0)}")
    print(f"Dog images: {sum(1 for l in labels if l == 1)}")
    print(f"Class balance: {sum(labels)/len(labels):.3f}")
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Create datasets with enhanced augmentation
    augmentation = EnhancedAugmentation(image_size=300)
    
    train_dataset = DogsCatsDataset(train_paths, train_labels, augmentation.get_train_transform())
    val_dataset = DogsCatsDataset(val_paths, val_labels, augmentation.get_val_transform())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    print("\nCreating EfficientNet-B3 model...")
    model = create_model()
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss, optimizer, and scheduler
    criterion = LabelSmoothingBCELoss(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Augmentation objects
    cutmix = CutMix(alpha=1.0)
    mixup = MixUp(alpha=1.0)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    
    # Training
    print("\nStarting training...")
    scaler = GradScaler()
    best_val_loss = float('inf')
    
    for epoch in range(20):  # Max 20 epochs
        print(f"\nEpoch {epoch+1}/20")
        
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler, cutmix, mixup)
        
        # Validation
        val_loss, val_log_loss = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Log Loss: {val_log_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_log_loss)
        
        # Save best model
        if val_log_loss < best_val_loss:
            best_val_loss = val_log_loss
            torch.save(model.state_dict(), '/home/code/experiments/best_model_b3.pth')
            print(f"→ Best model saved (Val Log Loss: {val_log_loss:.6f})")
        
        # Early stopping
        if early_stopping(val_log_loss, model):
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
    
    # Load best model
    early_stopping.load_best_weights(model)
    
    print(f"\nTraining completed! Best validation log loss: {best_val_loss:.6f}")
    
    # Predict on test set
    print("\nLoading best model and predicting on test set...")
    test_dir = os.path.join(data_dir, 'test')
    test_predictions, test_ids = predict_test(model, test_dir, augmentation.get_val_transform(), device)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_ids,
        'label': test_predictions
    })
    
    submission = submission.sort_values('id')
    submission.to_csv('/home/submission/submission.csv', index=False)
    
    print(f"\nSubmission saved to /home/submission/submission.csv")
    print(f"Total predictions: {len(test_predictions)}")
    print(f"Prediction range: [{min(test_predictions):.4f}, {max(test_predictions):.4f}]")
    
    return best_val_loss

if __name__ == "__main__":
    best_score = main()
    print(f"\nFinal Validation Log Loss: {best_score:.6f}")