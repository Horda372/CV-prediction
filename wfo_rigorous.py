import os
import pandas as pd
import numpy as np
import torch
import json
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import TrainConfig
from train_hybrid import train_model
from utils import HybridCandlestickDataset
from models import CustomCandlestickCNN, LSTMBaseline, Hybrid_CNN_LSTM

def run_event_driven_backtest(test_df, model_path, model_type, start_capital):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load correct model
    if model_type == "cnn":
        model = CustomCandlestickCNN(num_classes=2)
    elif model_type == "lstm":
        model = LSTMBaseline(input_dim=11, num_classes=2)
    elif model_type == "hybrid":
        model = Hybrid_CNN_LSTM(lstm_input_dim=11, num_classes=2)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = HybridCandlestickDataset(test_df, "dataset_market_cv/images", "dataset_market_cv/tensors", transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    all_probs = []
    with torch.no_grad():
        for images, tabular, _, _ in val_loader:
            images, tabular = images.to(device), tabular.to(device)
            if model_type == "cnn":
                outputs = model(images)
            elif model_type == "lstm":
                outputs = model(tabular)
            elif model_type == "hybrid":
                outputs = model(images, tabular)
            
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().tolist())
            
    val_df = test_df.copy()
    val_df["pred_prob"] = all_probs
    val_df["end_date"] = pd.to_datetime(val_df["end_date"])
    val_df["trade_exit_date"] = pd.to_datetime(val_df["trade_exit_date"])
    val_df.sort_values(by="end_date", inplace=True)
    
    th = 0.8 # Safe generalized threshold
    realized_capital = start_capital
    unrealized_trades = []
    equity_curve = []
    symbol_cooldown = {}
    
    for idx, row in val_df.iterrows():
        current_date = row["end_date"]
        
        still_open = []
        for trade in unrealized_trades:
            t_exit_date, t_profit, t_margin = trade
            if t_exit_date <= current_date:
                realized_capital += t_profit
                equity_curve.append(realized_capital)
            else:
                still_open.append(trade)
        unrealized_trades = still_open
        
        if row["pred_prob"] >= th:
            symbol = row["symbol"]
            if symbol in symbol_cooldown and current_date < symbol_cooldown[symbol]:
                continue
                
            avg_sl_pct = 0.01
            position_size = (realized_capital * 0.02) / avg_sl_pct
            margin_used = position_size * 0.02 
            
            current_locked_margin = sum([t[2] for t in unrealized_trades])
            if current_locked_margin + margin_used > realized_capital * 0.8:
                continue 
                
            symbol_cooldown[symbol] = current_date + pd.Timedelta(days=20)
            
            transaction_cost = position_size * 0.00007
            trade_profit = position_size * (row["pct_change"] / 100.0) - transaction_cost
            
            unrealized_trades.append((row["trade_exit_date"], trade_profit, margin_used))
            
    for trade in unrealized_trades:
        realized_capital += trade[1]
        equity_curve.append(realized_capital)
        
    return realized_capital, equity_curve

def main():
    print("="*60)
    print("RIGOROUS 3-MODEL WALK-FORWARD OPTIMIZATION (FROM SCRATCH)")
    print("="*60)
    
    df = pd.read_csv("dataset_market_cv/metadata.csv")
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    total_len = len(df)
    folds = [
        {"train_end": int(total_len * 0.60), "test_end": int(total_len * 0.73)},
        {"train_end": int(total_len * 0.73), "test_end": int(total_len * 0.86)},
        {"train_end": int(total_len * 0.86), "test_end": total_len}
    ]
    
    models = ["lstm", "cnn", "hybrid"]
    results = {m: [] for m in models}
    global_curves = {m: [10000.0] for m in models}
    capitals = {m: 10000.0 for m in models}
    
    for i, fold in enumerate(folds):
        print(f"\n{'='*40}\nSTARTING FOLD {i+1}/3\n{'='*40}")
        
        train_df = df.iloc[0:fold["train_end"]]
        test_df = df.iloc[fold["train_end"]:fold["test_end"]]
        
        # Save temp metadata for this fold
        train_meta_path = f"dataset_market_cv/meta_fold{i}.csv"
        if not os.path.exists(train_meta_path):
            train_df.to_csv(train_meta_path, index=False)
            
        for m in models:
            print(f"\n--- Model: {m.upper()} | Fold: {i+1} ---")
            model_path = f"wfo_rigorous_{m}_fold{i}.pth"
            
            if not os.path.exists(model_path):
                print(f"[*] Training {m.upper()} from scratch...")
                config = TrainConfig()
                config.metadata_file = train_meta_path
                config.model_type = m
                config.save_model_path = model_path
                config.epochs = 50 if m == "lstm" else 20  # Fast for LSTM, 20 for CNN/Hybrid
                config.early_stopping_patience = 5
                
                train_model(config)
            else:
                print(f"[+] Model weights {model_path} found. Skipping training (Resuming!).")
                
            print(f"[*] Running Event-Driven OOS Backtest for {m.upper()} Fold {i+1}...")
            cap, curve = run_event_driven_backtest(test_df, model_path, m, capitals[m])
            capitals[m] = cap
            if len(curve) > 0:
                global_curves[m].extend(curve)
                
            print(f"[>] {m.upper()} Fold {i+1} Final Capital: ${cap:.2f}")

    print("\n" + "="*60)
    print("FINAL RIGOROUS WFO RESULTS")
    print("="*60)
    for m in models:
        print(f"{m.upper()}: ${capitals[m]:.2f}")
        
    plt.figure(figsize=(12, 6))
    for m in models:
        plt.plot(global_curves[m], label=f"{m.upper()} (Final: ${capitals[m]:.2f})")
    plt.title("Rigorous WFO Comparison (Event-Driven)")
    plt.xlabel("Trade Timeline (All Folds)")
    plt.ylabel("Portfolio Balance ($)")
    plt.grid(True, alpha=0.5)
    plt.legend()
    plt.savefig("wfo_rigorous_comparison.png", dpi=150)
    print("Saved comparison chart to wfo_rigorous_comparison.png")

if __name__ == "__main__":
    main()
