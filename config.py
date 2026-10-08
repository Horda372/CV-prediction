import os

class AppConfig:
    csv_dir = "market_data"
    output_dir = "dataset_market_cv"
    img_dir = os.path.join(output_dir, "images")
    tensor_dir = os.path.join(output_dir, "tensors")
    metadata_file = os.path.join(output_dir, "metadata.csv")
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"]
    timeframe = "1d"
    
class CSVConfig(AppConfig):
    window_size = 20
    normalize_data = True
    moving_averages = [10, 20]
    atr_period = 14
    tp_atr_multiplier = 2.0
    sl_atr_multiplier = 1.0
    max_holding_period = 20
    spread_pct = 0.00015 # ~1.5 pips
    daily_swap_pct = 0.00005 # ~0.005% per day for holding overnight

class TrainConfig(AppConfig):
    model_type = "cnn"
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
    transaction_cost = 0.007 # 0.007% commission
    model_backbone = "custom_cnn"
    experiment_log_path = "experiment_log_hybrid.csv"
    save_model_path = "best_model_hybrid.pth"
    save_history_path = "history.csv"
    save_plot_path = "training_curves_hybrid.png"

class BacktestConfig(AppConfig):
    model_type = "cnn"
    model_path = "best_model_cnn.pth"
    train_split_pct = 0.8
    batch_size = 32
    starting_capital = 10000.0
    risk_per_trade_pct = 0.02
    reward_to_risk_ratio = 2.0
    transaction_cost_pct = 0.00007 # 0.007% commission
    symbol_cooldown_days = 20
