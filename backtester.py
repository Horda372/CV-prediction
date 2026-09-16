import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from train_hybrid import HybridCandlestickDataset
from models import CustomCandlestickCNN, LSTMBaseline, Hybrid_CNN_LSTM
import matplotlib.pyplot as plt

class BacktestConfig:
    csv_file = "dataset_market_cv/metadata.csv"
    img_dir = "dataset_market_cv/images"
    tensor_dir = "dataset_market_cv/tensors"
    
    # Model to evaluate
    model_type = "hybrid"  # Change to "cnn" or "lstm" if needed
    model_path = f"best_model_hybrid.pth"
    
    train_split_pct = 0.8
    batch_size = 32
    
    # PROFESSIONAL FINANCIAL SIMULATION
    starting_capital = 10000.0
    risk_per_trade_pct = 0.02  # Risk exactly 2% of current portfolio equity per trade
    reward_to_risk_ratio = 2.0 # TP is 2x ATR, SL is 1x ATR
    transaction_cost_pct = 0.0005 # Used for timeouts
    
    # CONCURRENCY/MARGIN MANAGEMENT
    # Prevent taking a new trade on the same symbol if we already have an active one.
    # Since we don't have exact exit dates, we enforce a strict 20-day cooldown per symbol.
    symbol_cooldown_days = 20

def run_backtest(config: BacktestConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting REALISTIC Backtest on Validation Set using Architecture: {config.model_type.upper()}")
    
    df = pd.read_csv(config.csv_file)
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
    
    # Load Model
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
        print(f"Loaded weights from {config.model_path}")
    else:
        print(f"Warning: {config.model_path} not found. Running with random weights for demonstration.")
        
    model = model.to(device)
    model.eval()
    
    all_probs = []
    
    print("Running inference...")
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
    
    # --- SIMULATE REALISTIC TRADING ---
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_th = 0.5
    best_final_capital = 0.0
    best_equity_curve = []
    
    print("\n" + "="*90)
    print("PROFESSIONAL BACKTEST RESULTS (2% Risk, Concurrency Protection)")
    print("="*90)
    print(f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Max Drawdown':<15} | {'Final Capital':<15}")
    print("-" * 90)
    
    for th in thresholds:
        capital = config.starting_capital
        equity_curve = [capital]
        peak_capital = capital
        max_drawdown = 0.0
        
        trades_taken = 0
        winning_trades = 0
        
        # Track when a symbol is available to trade again
        # Dictionary mapping symbol -> datetime
        symbol_cooldown = {}
        
        for idx, row in val_df.iterrows():
            if row["pred_prob"] >= th:
                symbol = row["symbol"]
                current_date = row["end_date"]
                
                # Check concurrency cooldown
                if symbol in symbol_cooldown:
                    if current_date < symbol_cooldown[symbol]:
                        continue # Symbol is locked in an active trade, skip this signal
                
                # We take the trade
                trades_taken += 1
                
                # Update cooldown (block this symbol for N days)
                symbol_cooldown[symbol] = current_date + pd.Timedelta(days=config.symbol_cooldown_days)
                
                # Financial simulation based on fixed fractional risk
                # If exit_type == 'TP', we win 2x Risk (because TP is 2x ATR, SL is 1x ATR)
                # If exit_type == 'SL', we lose 1x Risk.
                # If timeout, we scale the risk based on pct_change direction (simplified)
                
                exit_type = row["exit_type"]
                trade_profit = 0.0
                
                if exit_type == "TP":
                    trade_profit = capital * config.risk_per_trade_pct * config.reward_to_risk_ratio
                    trade_profit -= capital * config.transaction_cost_pct # deduct commissions
                    winning_trades += 1
                elif exit_type == "SL":
                    trade_profit = - (capital * config.risk_per_trade_pct)
                    trade_profit -= capital * config.transaction_cost_pct
                else: # Timeout
                    # Approximate timeout result (slightly positive or negative)
                    if row["pct_change"] > 0:
                        trade_profit = capital * (config.risk_per_trade_pct * 0.5) # Assume partial win
                        winning_trades += 1
                    else:
                        trade_profit = - (capital * config.risk_per_trade_pct * 0.5) # Assume partial loss
                    trade_profit -= capital * config.transaction_cost_pct
                
                capital += trade_profit
                equity_curve.append(capital)
                
                if capital > peak_capital:
                    peak_capital = capital
                
                drawdown = (peak_capital - capital) / peak_capital
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
                    
        win_rate = (winning_trades / trades_taken * 100) if trades_taken > 0 else 0
        
        print(f"{th:<10.1f} | {trades_taken:<8} | {win_rate:<9.2f}% | {max_drawdown*100:<14.2f}% | ${capital:<14.2f}")
        
        if capital > best_final_capital:
            best_final_capital = capital
            best_th = th
            best_equity_curve = equity_curve

    print("="*90)
    
    if best_final_capital > config.starting_capital:
        print(f"\nHighly Profitable Strategy Found at Threshold {best_th}")
        plt.figure(figsize=(10, 5))
        plt.plot(best_equity_curve, color='green', linewidth=2)
        plt.title(f"REALISTIC Equity Curve (Init: ${config.starting_capital}, Final: ${best_final_capital:.2f})")
        plt.xlabel("Number of Trades")
        plt.ylabel("Portfolio Balance ($)")
        plt.grid(True, alpha=0.5)
        curve_path = f"equity_curve_{config.model_type}_realistic.png"
        plt.savefig(curve_path, dpi=150)
        print(f"Saved realistic equity curve plot to {curve_path}")
    else:
        print(f"\nNo profitable thresholds found. Maximum capital reached: ${best_final_capital:.2f}")

if __name__ == "__main__":
    run_backtest(BacktestConfig())
