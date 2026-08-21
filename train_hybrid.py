import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import datetime

# Import models from models.py
from models import CustomCandlestickCNN, LSTMBaseline, Hybrid_CNN_LSTM

# ==========================================
# CONFIGURATION
# ==========================================
class TrainConfig:
    csv_file = "dataset_market_cv/metadata.csv"
    img_dir = "dataset_market_cv/images"
    tensor_dir = "dataset_market_cv/tensors"
    
    # Model Selection: "cnn", "lstm", or "hybrid"
    model_type = "hybrid"  
    
    batch_size = 32
    epochs = 10
    learning_rate = 0.0001
    train_split_pct = 0.8
    num_classes = 2
    sampler_type = "weighted"
    loss_type = "focal"
    focal_alpha = 0.75
    focal_gamma = 2.0
    early_stopping_patience = 4
    
    # Financial params
    transaction_cost = 0.05
    
    experiment_log_path = "experiment_log_hybrid.csv"
    save_model_path = f"best_model_{model_type}.pth"
    save_plot_path = f"training_curves_{model_type}.png"

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = -alpha_t * ((1.0 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean': return torch.mean(focal_loss)
        elif self.reduction == 'sum': return torch.sum(focal_loss)
        else: return focal_loss

# ==========================================
# CUSTOM HYBRID DATASET
# ==========================================
class HybridCandlestickDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tensor_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tensor_dir = tensor_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Image
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")

        # 2. Load Tensor
        tensor_name = row.get("tensor_filename", img_name.replace(".png", ".npy"))
        tensor_path = os.path.join(self.tensor_dir, tensor_name)
        try:
            tabular_data = np.load(tensor_path)
            tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32)
        except Exception as e:
            raise FileNotFoundError(f"Error loading tensor {tensor_path}: {e}")

        label = int(row["target_label"])
        pct_change = float(row["pct_change"])

        return image, tabular_tensor, label, pct_change

# ==========================================
# TRAINING PIPELINE
# ==========================================
def train_model(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training Architecture: {config.model_type.upper()}")

    df = pd.read_csv(config.csv_file)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    split_idx = int(len(df) * config.train_split_pct)
    if split_idx == 0: split_idx = 1
        
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = HybridCandlestickDataset(train_df, config.img_dir, config.tensor_dir, transform=data_transforms)
    val_dataset = HybridCandlestickDataset(val_df, config.img_dir, config.tensor_dir, transform=data_transforms)

    if config.sampler_type == "weighted":
        train_labels = train_df['target_label'].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = torch.DoubleTensor(class_weights[train_labels])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model Selection
    if config.model_type == "cnn":
        model = CustomCandlestickCNN(num_classes=config.num_classes)
    elif config.model_type == "lstm":
        # Check input dim from the first tensor
        sample_tensor = train_dataset[0][1]
        input_dim = sample_tensor.shape[1]
        model = LSTMBaseline(input_dim=input_dim, num_classes=config.num_classes)
    elif config.model_type == "hybrid":
        sample_tensor = train_dataset[0][1]
        input_dim = sample_tensor.shape[1]
        model = Hybrid_CNN_LSTM(lstm_input_dim=input_dim, num_classes=config.num_classes)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
        
    model = model.to(device)

    if config.loss_type == "focal":
        criterion_none = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, reduction='none')
    else:
        criterion_none = nn.CrossEntropyLoss(reduction='none')
        
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_return = -9999.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, config.epochs + 1):
        # --- TRAIN ---
        model.train()
        train_loss, train_corrects, total_train = 0.0, 0, 0

        for images, tabular, labels, pct_changes in train_loader:
            images, tabular, labels, pct_changes = images.to(device), tabular.to(device), labels.to(device), pct_changes.to(device).float()
            optimizer.zero_grad()

            if config.model_type == "cnn":
                outputs = model(images)
            elif config.model_type == "lstm":
                outputs = model(tabular)
            elif config.model_type == "hybrid":
                outputs = model(images, tabular)
                
            raw_loss = criterion_none(outputs, labels)
            loss_weights = 1.0 + 5.0 * torch.abs(pct_changes / 100.0) # Downscale pct change
            loss = torch.mean(raw_loss * loss_weights)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            total_train += images.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_corrects / total_train

        # --- VAL ---
        model.eval()
        val_loss, total_val = 0.0, 0
        all_probs, all_labels, all_pct_changes = [], [], []

        with torch.no_grad():
            for images, tabular, labels, pct_changes in val_loader:
                images, tabular, labels, pct_changes = images.to(device), tabular.to(device), labels.to(device), pct_changes.to(device).float()

                if config.model_type == "cnn":
                    outputs = model(images)
                elif config.model_type == "lstm":
                    outputs = model(tabular)
                elif config.model_type == "hybrid":
                    outputs = model(images, tabular)
                    
                raw_loss = criterion_none(outputs, labels)
                loss_weights = 1.0 + 5.0 * torch.abs(pct_changes / 100.0)
                loss = torch.mean(raw_loss * loss_weights)

                val_loss += loss.item() * images.size(0)
                total_val += images.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_pct_changes.extend(pct_changes.cpu().tolist())

        epoch_val_loss = val_loss / total_val
        
        # Threshold Optimization
        best_epoch_val_return = -9999.0
        best_epoch_val_acc = 0.0
        best_epoch_val_threshold = 0.5
        cost = config.transaction_cost
        
        for th in np.arange(0.1, 1.0, 0.1):
            th_preds = [1 if p >= th else 0 for p in all_probs]
            tp = sum((p == 1 and l == 1) for p, l in zip(th_preds, all_labels))
            tn = sum((p == 0 and l == 0) for p, l in zip(th_preds, all_labels))
            
            th_acc = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
            th_return = sum((change - cost) for pred, change in zip(th_preds, all_pct_changes) if pred == 1)
            
            if th_return > best_epoch_val_return:
                best_epoch_val_return = th_return
                best_epoch_val_acc = th_acc
                best_epoch_val_threshold = th
                
        if best_epoch_val_return <= -9999.0:
            best_epoch_val_threshold = 0.5
            best_epoch_val_return = 0.0

        print(f"Epoch {epoch}/{config.epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
              f"Val Net Return: {best_epoch_val_return:+.2f}% | Best TH: {best_epoch_val_threshold:.1f}")

        history.append({"epoch": epoch, "train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "val_net_return": best_epoch_val_return})

        if best_epoch_val_return > best_val_return:
            best_val_return = best_epoch_val_return
            best_epoch = epoch
            torch.save(model.state_dict(), config.save_model_path)
            print(f"  [+] Saved new best model -> {best_val_return:+.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"\n[-] Early stopping triggered.")
            break

if __name__ == "__main__":
    train_model(TrainConfig())
