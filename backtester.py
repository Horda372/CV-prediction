import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

from config import BacktestConfig
from utils import HybridCandlestickDataset
from models import CustomCandlestickCNN, LSTMBaseline, Hybrid_CNN_LSTM

def run_backtest(config: BacktestConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Backtest using Architecture: {config.model_type.upper()}")
    
    df = pd.read_csv(config.metadata_file)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    split_idx = int(len(df) * config.train_split_pct)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    
    if len(val_df) == 0:
        print("Error: Validation set is empty.")
        return

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = HybridCandlestickDataset(val_df, config.img_dir, config.tensor_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    
    if config.model_type == "cnn":
        model = CustomCandlestickCNN()
    elif config.model_type == "lstm":
        sample_tensor = val_dataset[0][1]
        model = LSTMBaseline(input_dim=sample_tensor.shape[1])
    elif config.model_type == "hybrid":
        sample_tensor = val_dataset[0][1]
        model = Hybrid_CNN_LSTM(lstm_input_dim=sample_tensor.shape[1])
        
    if os.path.exists(config.model_path):
        model.load_state_dict(torch.load(config.model_path, map_location=device))
    else:
        print(f"Warning: {config.model_path} not found.")
        
    model = model.to(device)
    model.eval()
    
    all_probs = []
    
    with torch.no_grad():
        for images, tabular, labels, pct_changes in val_loader:
            images, tabular = images.to(device), tabular.to(device)
            
            if config.model_type == "cnn":
                outputs = model(images)
            elif config.model_type == "lstm":
                outputs = model(tabular)
            elif config.model_type == "hybrid":
                outputs = model(images, tabular)
                
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().tolist())
            
    val_df["pred_prob"] = all_probs
    
    val_df["end_date"] = pd.to_datetime(val_df["end_date"])
    val_df["trade_exit_date"] = pd.to_datetime(val_df["trade_exit_date"])
    val_df.sort_values(by="end_date", inplace=True)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_th = 0.5
    best_final_capital = 0.0
    best_equity_curve = []
    
    print("\n" + "="*90)
    print("BACKTEST RESULTS (EVENT-DRIVEN W/ MARGIN)")
    print("="*90)
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Max Drawdown':<15} | {'Final Capital':<15}")
    print("-" * 90)
    
    for th in thresholds:
        realized_capital = config.starting_capital
        unrealized_trades = []
        equity_curve = [realized_capital]
        peak_capital = realized_capital
        max_drawdown = 0.0
        
        trades_taken = 0
        winning_trades = 0
        
        symbol_cooldown = {}
        
        for idx, row in val_df.iterrows():
            current_date = row["end_date"]
            
            still_open = []
            for trade in unrealized_trades:
                t_exit_date, t_profit, t_margin = trade
                if t_exit_date <= current_date:
                    realized_capital += t_profit
                    equity_curve.append(realized_capital)
                    if realized_capital > peak_capital:
                        peak_capital = realized_capital
                    drawdown = (peak_capital - realized_capital) / peak_capital
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                else:
                    still_open.append(trade)
            unrealized_trades = still_open
            
            if row["pred_prob"] >= th:
                symbol = row["symbol"]
                
                if symbol in symbol_cooldown and current_date < symbol_cooldown[symbol]:
                    continue
                    
                avg_sl_pct = 0.01
                position_size = (realized_capital * config.risk_per_trade_pct) / avg_sl_pct
                margin_used = position_size * 0.02
                
                current_locked_margin = sum([t[2] for t in unrealized_trades])
                if current_locked_margin + margin_used > realized_capital * 0.8:
                    continue
                    
                trades_taken += 1
                symbol_cooldown[symbol] = current_date + pd.Timedelta(days=config.symbol_cooldown_days)
                
                transaction_cost = position_size * config.transaction_cost_pct
                trade_profit = position_size * (row["pct_change"] / 100.0) - transaction_cost
                
                unrealized_trades.append((row["trade_exit_date"], trade_profit, margin_used))
                if trade_profit > 0:
                    winning_trades += 1
                    
        for trade in unrealized_trades:
            realized_capital += trade[1]
            equity_curve.append(realized_capital)
            if realized_capital > peak_capital:
                peak_capital = realized_capital
            drawdown = (peak_capital - realized_capital) / peak_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        win_rate = (winning_trades / trades_taken * 100) if trades_taken > 0 else 0
        
        print(f"{th:<10.1f} | {trades_taken:<8} | {win_rate:<9.2f}% | {max_drawdown*100:<14.2f}% | ${realized_capital:<14.2f}")
        
        if realized_capital > best_final_capital:
            best_final_capital = realized_capital
            best_th = th
            best_equity_curve = equity_curve

    print("="*90)
    
    if best_final_capital > config.starting_capital:
        plt.figure(figsize=(10, 5))
        plt.plot(best_equity_curve, color='green', linewidth=2)
        plt.title(f"Equity Curve (Init: ${config.starting_capital}, Final: ${best_final_capital:.2f})")
        plt.xlabel("Number of Trades")
        plt.ylabel("Portfolio Balance ($)")
        plt.grid(True, alpha=0.5)
        curve_path = f"equity_curve_{config.model_type}_realistic.png"
        plt.savefig(curve_path, dpi=150)
    else:
        print(f"\nMaximum capital reached: ${best_final_capital:.2f}")

if __name__ == "__main__":
    run_backtest(BacktestConfig())
