import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
class TrainConfig:
    csv_file = "dataset_market_cv/metadata.csv"
    img_dir = "dataset_market_cv/images"
    batch_size = 32
    epochs = 10
    learning_rate = 0.0001
    train_split_pct = 0.8
    num_classes = 2
    sampler_type = "weighted"  # "weighted" or "standard"
    model_backbone = "custom_cnn"
    experiment_log_path = "experiment_log.csv"
    save_model_path = "best_model.pth"
    save_history_path = "history.csv"
    save_plot_path = "training_curves.png"

class CustomCandlestickCNN(nn.Module):
    """
    Custom regularized 4-layer CNN designed specifically for geometric financial charts.
    Uses BatchNorm and Dropout to force generalization and prevent pixel memorization.
    """
    def __init__(self, num_classes=2, dropout_prob=0.3):
        super(CustomCandlestickCNN, self).__init__()
        
        # Conv block 1: Input (3, 224, 224) -> Output (16, 112, 112)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2: Input (16, 112, 112) -> Output (32, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3: Input (32, 56, 56) -> Output (64, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 4: Input (64, 28, 28) -> Output (128, 14, 14)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # 128 channels * 14 * 14 feature map dimension = 25088 input features
        self.fc1 = nn.Linear(128 * 14 * 14, 64)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# CUSTOM DATASET
# ==========================================
class CandlestickDataset(Dataset):
    """
    Custom PyTorch Dataset that loads candlestick images based on the metadata CSV.
    """
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Load image and convert to RGB
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")

        label = int(row["target_label"])
        pct_change = float(row["pct_change"])

        if self.transform:
            image = self.transform(image)

        return image, label, pct_change

# ==========================================
# TRAINING AND VALIDATION PIPELINE
# ==========================================
def train_model(config: TrainConfig):
    # Determine execution device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if dataset exists
    if not os.path.exists(config.csv_file) or not os.path.exists(config.img_dir):
        print(f"Error: Dataset not found. Please run generate_mock_data.py or import.py first.")
        return

    # Load metadata and parse dates for chronological sorting
    df = pd.read_csv(config.csv_file)
    print(f"Loaded metadata.csv containing {len(df)} records.")
    
    # Ensure end_date is parsed as datetime
    df["end_date"] = pd.to_datetime(df["end_date"])
    
    # Sort chronologically by end_date to avoid look-ahead bias
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    # Calculate chronological train/val split index
    split_idx = int(len(df) * config.train_split_pct)
    
    # Safeguard if split results in an empty set
    if split_idx == 0:
        split_idx = 1
        
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Chronological split applied:")
    print(f"  - Training samples: {len(train_df)} (from {train_df['end_date'].min()} to {train_df['end_date'].max()})")
    print(f"  - Validation samples: {len(val_df)} (from {val_df['end_date'].min()} to {val_df['end_date'].max()})")

    # Define standard PyTorch transformations
    # Resizing to 224x224 and applying standard ImageNet normalization (since we use a pre-trained ResNet-18)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset instances
    train_dataset = CandlestickDataset(train_df, config.img_dir, transform=data_transforms)
    val_dataset = CandlestickDataset(val_df, config.img_dir, transform=data_transforms)

    # Dataloaders
    if hasattr(config, "sampler_type") and config.sampler_type == "weighted":
        # Calculate class weights for sampler
        # 0 = No-Buy, 1 = Buy
        train_labels = train_df['target_label'].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        
        # Assign a weight to each sample based on its class
        sample_weights = class_weights[train_labels]
        sample_weights = torch.DoubleTensor(sample_weights)
        
        # Instantiate WeightedRandomSampler
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Pass sampler to DataLoader (shuffle must be False when using a sampler)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, drop_last=False)
        print(f"Using WeightedRandomSampler to balance batches (Class counts: {class_counts})")
    else:
        # Standard Loader
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
        print("Using standard shuffled DataLoader (no batch balancing)")

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # Initialize Custom Regularized CNN Model
    print("Initializing CustomCandlestickCNN backbone from scratch...")
    model = CustomCandlestickCNN(num_classes=config.num_classes, dropout_prob=0.3)
    model = model.to(device)

    # Loss function and Optimizer
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_val_f1 = -1.0
    best_epoch = 0
    best_metrics = {
        "val_loss": 0.0,
        "val_acc": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0
    }
    history = []

    print(f"Starting training loop ({config.epochs} epochs)...")
    for epoch in range(1, config.epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_train_samples = 0

        for inputs, labels, pct_changes in train_loader:
            inputs, labels, pct_changes = inputs.to(device), labels.to(device), pct_changes.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            
            # Custom Return-Magnitude Weighted Loss
            raw_loss = criterion_none(outputs, labels)
            loss_weights = 1.0 + 5.0 * torch.abs(pct_changes)
            loss = torch.mean(raw_loss * loss_weights)
            
            # Backward pass & Optimize
            loss.backward()
            optimizer.step()

            # Statistics tracking
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            total_train_samples += inputs.size(0)

        epoch_train_loss = train_loss / total_train_samples if total_train_samples > 0 else 0
        epoch_train_acc = train_corrects / total_train_samples if total_train_samples > 0 else 0

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        total_val_samples = 0
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, pct_changes in val_loader:
                inputs, labels, pct_changes = inputs.to(device), labels.to(device), pct_changes.to(device).float()

                # Forward pass
                outputs = model(inputs)
                
                # Custom Return-Magnitude Weighted Loss
                raw_loss = criterion_none(outputs, labels)
                loss_weights = 1.0 + 5.0 * torch.abs(pct_changes)
                loss = torch.mean(raw_loss * loss_weights)

                # Statistics tracking
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                total_val_samples += inputs.size(0)
                
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0
        epoch_val_acc = val_corrects / total_val_samples if total_val_samples > 0 else 0

        # Calculate Precision, Recall, and F1-Score for validation split
        tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
        tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
        
        epoch_val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        epoch_val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        epoch_val_f1 = (2 * epoch_val_precision * epoch_val_recall) / (epoch_val_precision + epoch_val_recall) if (epoch_val_precision + epoch_val_recall) > 0 else 0.0

        print(f"Epoch {epoch}/{config.epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
              f"Val Precision: {epoch_val_precision*100:.2f}% | Val Recall: {epoch_val_recall*100:.2f}% | "
              f"Val F1: {epoch_val_f1:.4f}")

        # Save performance stats to history list
        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "val_precision": epoch_val_precision,
            "val_recall": epoch_val_recall,
            "val_f1": epoch_val_f1
        })

        # Save the best model if validation F1-Score improves
        if epoch_val_f1 > best_val_f1 and total_val_samples > 0:
            best_val_f1 = epoch_val_f1
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_metrics = {
                "val_loss": epoch_val_loss,
                "val_acc": epoch_val_acc,
                "val_precision": epoch_val_precision,
                "val_recall": epoch_val_recall,
                "val_f1": epoch_val_f1
            }
            torch.save(model.state_dict(), config.save_model_path)
            print(f"  [+] Saved new best model to {config.save_model_path} with Val F1: {best_val_f1:.4f}")

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.save_history_path, index=False)
    print(f"\nSaved training metrics history to {config.save_history_path}")

    # Generate and save loss/accuracy/F1 plot
    plot_training_curves(history_df, config.save_plot_path)
    print(f"Saved loss and accuracy curves to {config.save_plot_path}")

    # Persistent Experiment Logging
    log_path = getattr(config, "experiment_log_path", "experiment_log.csv")
    import datetime
    
    log_exists = os.path.exists(log_path)
    
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": getattr(config, "model_backbone", "custom_cnn"),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "sampler_type": getattr(config, "sampler_type", "standard"),
        "loss_weighted": "yes (magnitude-weighted)",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_metrics["val_loss"], 4),
        "best_val_accuracy": round(best_metrics["val_acc"] * 100.0, 2),
        "best_val_precision": round(best_metrics["val_precision"] * 100.0, 2),
        "best_val_recall": round(best_metrics["val_recall"] * 100.0, 2),
        "best_val_f1": round(best_metrics["val_f1"], 4)
    }
    
    log_df = pd.DataFrame([log_entry])
    if log_exists:
        log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_path, mode='w', header=True, index=False)
    print(f"Persistent experiment settings and best metrics appended to: {os.path.abspath(log_path)}")

def plot_training_curves(history_df: pd.DataFrame, save_path: str):
    """
    Generates training and validation curves for loss, accuracy, and F1-score, then saves them as a PNG.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history_df["epoch"]

    # Plot Loss Curve
    ax1.plot(epochs, history_df["train_loss"], label="Train Loss", color="royalblue", marker="o", linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], label="Val Loss", color="orange", marker="x", linewidth=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Plot Accuracy and F1-Score Curve
    ax2.plot(epochs, history_df["train_acc"] * 100.0, label="Train Acc", color="forestgreen", marker="o", linewidth=2)
    ax2.plot(epochs, history_df["val_acc"] * 100.0, label="Val Acc", color="darkred", marker="x", linewidth=2)
    if "val_f1" in history_df.columns:
        ax2.plot(epochs, history_df["val_f1"] * 100.0, label="Val F1-Score", color="purple", marker="s", linestyle="-.", linewidth=2)
    ax2.set_title("Training/Val Accuracy & F1-Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Percentage (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    app_config = TrainConfig()
    train_model(app_config)
