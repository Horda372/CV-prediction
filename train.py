import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import matplotlib.pyplot as plt

from config import TrainConfig
from models import CustomCandlestickCNN
from utils import CandlestickDataset, FocalLoss

def train_model(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config.metadata_file) or not os.path.exists(config.img_dir):
        print(f"Error: Dataset not found.")
        return

    df = pd.read_csv(config.metadata_file)
    print(f"Loaded metadata.csv containing {len(df)} records.")
    
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    split_idx = int(len(df) * config.train_split_pct)
    if split_idx == 0:
        split_idx = 1
        
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Chronological split applied:")
    print(f"  - Training samples: {len(train_df)}")
    print(f"  - Validation samples: {len(val_df)}")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CandlestickDataset(train_df, config.img_dir, transform=data_transforms)
    val_dataset = CandlestickDataset(val_df, config.img_dir, transform=data_transforms)

    if getattr(config, "sampler_type", "") == "weighted":
        train_labels = train_df['target_label'].values
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        sample_weights = torch.DoubleTensor(class_weights[train_labels])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler, drop_last=False, num_workers=4, pin_memory=True)
        print(f"Using WeightedRandomSampler to balance batches (Class counts: {class_counts})")
    else:
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        print("Using standard shuffled DataLoader")

    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    backbone = getattr(config, "model_backbone", "custom_cnn")
    if backbone == "resnet18":
        print("Initializing Pretrained ResNet18 backbone...")
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, config.num_classes)
        )
    else:
        print("Initializing CustomCandlestickCNN backbone...")
        model = CustomCandlestickCNN(num_classes=config.num_classes, dropout_prob=0.3)
        
    model = model.to(device)

    if getattr(config, "loss_type", "cross_entropy") == "focal":
        print(f"Using Focal Loss (alpha={config.focal_alpha}, gamma={config.focal_gamma})")
        criterion_none = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, reduction='none')
    else:
        print("Using Standard CrossEntropy Loss")
        criterion_none = nn.CrossEntropyLoss(reduction='none')
        
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_val_return = -9999.0
    best_epoch = 0
    best_threshold = 0.5
    epochs_no_improve = 0
    best_metrics = {
        "val_loss": 0.0,
        "val_acc": 0.0,
        "val_precision": 0.0,
        "val_recall": 0.0,
        "val_f1": 0.0,
        "best_threshold": 0.5,
        "strategy_return": -9999.0
    }
    history = []

    print(f"Starting training loop ({config.epochs} epochs)...")
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_corrects = 0
        total_train_samples = 0

        for inputs, labels, pct_changes in train_loader:
            inputs, labels, pct_changes = inputs.to(device), labels.to(device), pct_changes.to(device).float()

            optimizer.zero_grad()

            outputs = model(inputs)
            
            raw_loss = criterion_none(outputs, labels)
            loss_weights = 1.0 + 5.0 * torch.abs(pct_changes / 100.0)
            loss = torch.mean(raw_loss * loss_weights)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            total_train_samples += inputs.size(0)

        epoch_train_loss = train_loss / total_train_samples if total_train_samples > 0 else 0
        epoch_train_acc = train_corrects / total_train_samples if total_train_samples > 0 else 0

        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        
        all_probs = []
        all_labels = []
        all_pct_changes = []

        with torch.no_grad():
            for inputs, labels, pct_changes in val_loader:
                inputs, labels, pct_changes = inputs.to(device), labels.to(device), pct_changes.to(device).float()

                outputs = model(inputs)
                
                raw_loss = criterion_none(outputs, labels)
                loss_weights = 1.0 + 5.0 * torch.abs(pct_changes / 100.0)
                loss = torch.mean(raw_loss * loss_weights)

                val_loss += loss.item() * inputs.size(0)
                total_val_samples += inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_pct_changes.extend(pct_changes.cpu().tolist())

        epoch_val_loss = val_loss / total_val_samples if total_val_samples > 0 else 0

        best_epoch_val_return = -9999.0
        best_epoch_val_f1 = 0.0
        best_epoch_val_precision = 0.0
        best_epoch_val_recall = 0.0
        best_epoch_val_acc = 0.0
        best_epoch_val_threshold = 0.5
        
        cost = getattr(config, "transaction_cost", 0.05)
        
        thresholds_to_test = np.arange(0.1, 1.0, 0.1)
        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels)
        all_pct_changes_np = np.array(all_pct_changes)
        
        for th in thresholds_to_test:
            th_preds = (all_probs_np >= th).astype(int)
            
            tp = np.sum((th_preds == 1) & (all_labels_np == 1))
            fp = np.sum((th_preds == 1) & (all_labels_np == 0))
            tn = np.sum((th_preds == 0) & (all_labels_np == 0))
            fn = np.sum((th_preds == 0) & (all_labels_np == 1))
            
            th_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            th_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            th_f1 = (2 * th_prec * th_rec) / (th_prec + th_rec) if (th_prec + th_rec) > 0 else 0.0
            th_acc = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
            
            th_return = np.sum(np.where(th_preds == 1, all_pct_changes_np - cost, 0))
            
            if th_return > best_epoch_val_return:
                best_epoch_val_return = th_return
                best_epoch_val_f1 = th_f1
                best_epoch_val_precision = th_prec
                best_epoch_val_recall = th_rec
                best_epoch_val_acc = th_acc
                best_epoch_val_threshold = th
                
        if best_epoch_val_return <= -9999.0:
            best_epoch_val_threshold = 0.5
            best_epoch_val_return = 0.0
            th_preds = (all_probs_np >= 0.5).astype(int)
            tp = np.sum((th_preds == 1) & (all_labels_np == 1))
            fp = np.sum((th_preds == 1) & (all_labels_np == 0))
            fn = np.sum((th_preds == 0) & (all_labels_np == 1))
            best_epoch_val_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            best_epoch_val_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            best_epoch_val_f1 = (2 * best_epoch_val_precision * best_epoch_val_recall) / (best_epoch_val_precision + best_epoch_val_recall) if (best_epoch_val_precision + best_epoch_val_recall) > 0 else 0.0
            best_epoch_val_acc = np.sum(th_preds == all_labels_np) / len(all_labels_np) if len(all_labels_np) > 0 else 0.0

        print(f"Epoch {epoch}/{config.epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {best_epoch_val_acc*100:.2f}% | "
              f"Val Precision: {best_epoch_val_precision*100:.2f}% | Val Recall: {best_epoch_val_recall*100:.2f}% | "
              f"Val Net Return: {best_epoch_val_return:+.2f}% | Best Threshold: {best_epoch_val_threshold:.1f}")

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": best_epoch_val_acc,
            "val_precision": best_epoch_val_precision,
            "val_recall": best_epoch_val_recall,
            "val_f1": best_epoch_val_f1,
            "val_net_return": best_epoch_val_return,
            "best_threshold": best_epoch_val_threshold
        })

        if best_epoch_val_return > best_val_return and total_val_samples > 0:
            best_val_return = best_epoch_val_return
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_threshold = best_epoch_val_threshold
            best_metrics = {
                "val_loss": epoch_val_loss,
                "val_acc": best_epoch_val_acc,
                "val_precision": best_epoch_val_precision,
                "val_recall": best_epoch_val_recall,
                "val_f1": best_epoch_val_f1,
                "best_threshold": best_epoch_val_threshold,
                "strategy_return": best_epoch_val_return
            }
            torch.save(model.state_dict(), config.save_model_path)
            print(f"  [+] Saved new best model to {config.save_model_path}")

        if best_epoch == epoch:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        patience = getattr(config, "early_stopping_patience", 3)
        if epochs_no_improve >= patience:
            print(f"\n[-] Early stopping triggered.")
            break

    history_df = pd.DataFrame(history)
    history_df.to_csv(config.save_history_path, index=False)
    plot_training_curves(history_df, config.save_plot_path)

    log_path = getattr(config, "experiment_log_path", "experiment_log.csv")
    import datetime
    
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": getattr(config, "model_backbone", "custom_cnn"),
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "sampler_type": getattr(config, "sampler_type", "standard"),
        "loss_weighted": f"yes (magnitude-weighted + {getattr(config, 'loss_type', 'cross_entropy')})",
        "best_epoch": best_epoch,
        "best_val_loss": round(best_metrics["val_loss"], 4),
        "best_val_accuracy": round(best_metrics["val_acc"] * 100.0, 2),
        "best_val_precision": round(best_metrics["val_precision"] * 100.0, 2),
        "best_val_recall": round(best_metrics["val_recall"] * 100.0, 2),
        "best_val_f1": round(best_metrics["val_f1"], 4),
        "best_threshold": round(best_metrics.get("best_threshold", 0.5), 2),
        "strategy_return": round(best_metrics.get("strategy_return", 0.0), 2)
    }
    
    if os.path.exists(log_path):
        try:
            existing_df = pd.read_csv(log_path)
            if "best_threshold" not in existing_df.columns:
                existing_df["best_threshold"] = 0.5
            if "strategy_return" not in existing_df.columns:
                existing_df["strategy_return"] = 0.0
            
            new_row_df = pd.DataFrame([log_entry])
            combined_df = pd.concat([existing_df, new_row_df], ignore_index=True)
            combined_df.to_csv(log_path, index=False)
        except Exception as e:
            print(f"Warning: Could not append to experiment log via DataFrame: {e}. Falling back.")
            log_df = pd.DataFrame([log_entry])
            log_df.to_csv(log_path, mode='a', header=False, index=False)
    else:
        log_df = pd.DataFrame([log_entry])
        log_df.to_csv(log_path, index=False)

def plot_training_curves(history_df: pd.DataFrame, save_path: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history_df["epoch"]

    ax1.plot(epochs, history_df["train_loss"], label="Train Loss", color="royalblue", marker="o", linewidth=2)
    ax1.plot(epochs, history_df["val_loss"], label="Val Loss", color="orange", marker="x", linewidth=2)
    ax1.set_title("Training & Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

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
    app_config.model_backbone = "resnet18"
    train_model(app_config)
