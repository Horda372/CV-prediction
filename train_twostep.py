import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from config import TrainConfig
from utils import FocalLoss, HybridCandlestickDataset
from models import Hybrid_CNN_LSTM, CustomCandlestickCNN, LSTMBaseline
from backtester import BacktestConfig, run_backtest

def run_two_step_training():
    config = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv(config.metadata_file)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    split_idx = int(len(df) * config.train_split_pct)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = HybridCandlestickDataset(train_df, config.img_dir, config.tensor_dir, transform=data_transforms)
    val_dataset = HybridCandlestickDataset(val_df, config.img_dir, config.tensor_dir, transform=data_transforms)
    
    sample_tensor = train_dataset[0][1]
    input_dim = sample_tensor.shape[1]
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    model = Hybrid_CNN_LSTM(lstm_input_dim=input_dim, num_classes=config.num_classes)
    
    cnn_path = "best_model_cnn.pth"
    lstm_path = "best_model_lstm.pth"
    
    if os.path.exists(cnn_path):
        cnn_state = torch.load(cnn_path, map_location=device)
        model.cnn.load_state_dict(cnn_state, strict=False)
    else:
        print(f"[-] Missing {cnn_path}. Exiting.")
        return
        
    if os.path.exists(lstm_path):
        lstm_state = torch.load(lstm_path, map_location=device)
        model.lstm.load_state_dict(lstm_state, strict=False)
    else:
        print(f"[-] Missing {lstm_path}. Exiting.")
        return
        
    for param in model.cnn.parameters():
        param.requires_grad = False
    for param in model.lstm.parameters():
        param.requires_grad = False
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (Fusion Layers Only): {trainable_params}")
    
    model = model.to(device)
    
    criterion_none = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma, reduction='none')
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    best_val_return = -9999.0
    epochs_no_improve = 0
    save_path = "best_model_hybrid_twostep.pth"
    
    max_epochs = 30
    
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss, total_train = 0.0, 0
        
        for images, tabular, labels, pct_changes in train_loader:
            images, tabular, labels, pct_changes = images.to(device), tabular.to(device), labels.to(device), pct_changes.to(device).float()
            optimizer.zero_grad()
            
            outputs = model(images, tabular)
            raw_loss = criterion_none(outputs, labels)
            loss_weights = 1.0 + 5.0 * torch.abs(pct_changes / 100.0)
            loss = torch.mean(raw_loss * loss_weights)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)
            
        epoch_train_loss = train_loss / total_train
        
        model.eval()
        all_probs, all_labels, all_pct_changes = [], [], []
        with torch.no_grad():
            for images, tabular, labels, pct_changes in val_loader:
                images, tabular = images.to(device), tabular.to(device)
                outputs = model(images, tabular)
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs[:, 1].cpu().tolist())
                all_labels.extend(labels.tolist())
                all_pct_changes.extend(pct_changes.tolist())
                
        best_epoch_val_return = -9999.0
        best_epoch_val_threshold = 0.5
        cost = config.transaction_cost
        
        all_probs_np = np.array(all_probs)
        all_pct_changes_np = np.array(all_pct_changes)
        
        for th in np.arange(0.1, 1.0, 0.1):
            th_preds = (all_probs_np >= th).astype(int)
            th_return = np.sum(np.where(th_preds == 1, all_pct_changes_np - cost, 0))
            if th_return > best_epoch_val_return:
                best_epoch_val_return = th_return
                best_epoch_val_threshold = th
                
        print(f"Epoch {epoch}/{max_epochs} -> Train Loss: {epoch_train_loss:.4f} | Val Net Return: {best_epoch_val_return:+.2f}% | Best TH: {best_epoch_val_threshold:.1f}")
        
        if best_epoch_val_return > best_val_return:
            best_val_return = best_epoch_val_return
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= 8:
            break
            
    b_config = BacktestConfig()
    b_config.model_type = "hybrid"
    b_config.model_path = save_path
    run_backtest(b_config)

if __name__ == "__main__":
    run_two_step_training()
