import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import matplotlib.pyplot as plt

from config import TrainConfig
from models import CustomCandlestickCNN, LSTMBaseline, Hybrid_CNN_LSTM
from utils import FocalLoss, HybridCandlestickDataset

def train_model(config: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv(config.metadata_file)
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

    if config.model_type == "cnn":
        model = CustomCandlestickCNN(num_classes=config.num_classes)
    elif config.model_type == "lstm":
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
            loss_weights = 1.0 + 5.0 * torch.abs(pct_changes / 100.0)
            loss = torch.mean(raw_loss * loss_weights)
            
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * images.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            total_train += images.size(0)

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = train_corrects / total_train

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
        
        best_epoch_val_return = -9999.0
        best_epoch_val_acc = 0.0
        best_epoch_val_threshold = 0.5
        cost = config.transaction_cost
        
        all_probs_np = np.array(all_probs)
        all_labels_np = np.array(all_labels)
        all_pct_changes_np = np.array(all_pct_changes)
        
        for th in np.arange(0.1, 1.0, 0.1):
            th_preds = (all_probs_np >= th).astype(int)
            tp = np.sum((th_preds == 1) & (all_labels_np == 1))
            tn = np.sum((th_preds == 0) & (all_labels_np == 0))
            
            th_acc = (tp + tn) / len(all_labels_np) if len(all_labels_np) > 0 else 0.0
            th_return = np.sum(np.where(th_preds == 1, all_pct_changes_np - cost, 0))
            
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
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= config.early_stopping_patience:
            print(f"\n[-] Early stopping triggered.")
            break

if __name__ == "__main__":
    train_model(TrainConfig())
