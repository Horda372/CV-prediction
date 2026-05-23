import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
    save_model_path = "best_model.pth"
    save_history_path = "history.csv"
    save_plot_path = "training_curves.png"

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

        if self.transform:
            image = self.transform(image)

        return image, label

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
    # Shuffle only training set to improve model generalization while keeping split chronological boundary
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    # Initialize Pre-trained ResNet-18 Model
    print("Initializing ResNet-18 pre-trained on ImageNet...")
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    
    # Modify the final classification head for binary classification (labels: 0 or 1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    
    model = model.to(device)

    # Loss function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    history = []

    print(f"Starting training loop ({config.epochs} epochs)...")
    for epoch in range(1, config.epochs + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_train_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
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

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Statistics tracking
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                total_val_samples += inputs.size(0)

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0
        epoch_val_acc = val_corrects / total_val_samples if total_val_samples > 0 else 0

        print(f"Epoch {epoch}/{config.epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

        # Save performance stats to history list
        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc
        })

        # Save the best model if validation loss improves
        if epoch_val_loss < best_val_loss and total_val_samples > 0:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), config.save_model_path)
            print(f"  [+] Saved new best model to {config.save_model_path} with Val Loss: {best_val_loss:.4f}")

    # Save training history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv(config.save_history_path, index=False)
    print(f"\nSaved training metrics history to {config.save_history_path}")

    # Generate and save loss/accuracy plot
    plot_training_curves(history_df, config.save_plot_path)
    print(f"Saved loss and accuracy curves to {config.save_plot_path}")

def plot_training_curves(history_df: pd.DataFrame, save_path: str):
    """
    Generates training and validation curves for loss and accuracy, then saves them as a PNG.
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

    # Plot Accuracy Curve
    ax2.plot(epochs, history_df["train_acc"] * 100.0, label="Train Acc", color="forestgreen", marker="o", linewidth=2)
    ax2.plot(epochs, history_df["val_acc"] * 100.0, label="Val Acc", color="darkred", marker="x", linewidth=2)
    ax2.set_title("Training & Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    app_config = TrainConfig()
    train_model(app_config)
