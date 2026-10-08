import os
from train_hybrid import train_model
from config import TrainConfig, BacktestConfig
from backtester import run_backtest

def main():
    print("\n" + "="*50)
    print("EXPERIMENT: LONG TRAINING TABULAR ONLY (BiLSTM + Attention)")
    print("="*50)

    t_config_lstm = TrainConfig()
    t_config_lstm.model_type = "lstm"
    t_config_lstm.experiment_log_path = "experiment_log_lstm_long.csv"
    t_config_lstm.save_model_path = "best_model_lstm.pth"
    t_config_lstm.save_plot_path = "training_curves_lstm_long.png"
    
    t_config_lstm.epochs = 150
    t_config_lstm.early_stopping_patience = 20
    t_config_lstm.learning_rate = 0.0001
    
    train_model(t_config_lstm)

    b_config_lstm = BacktestConfig()
    b_config_lstm.model_type = "lstm"
    b_config_lstm.model_path = "best_model_lstm.pth"
    run_backtest(b_config_lstm)
    
    print("\n" + "="*50)
    print("EXPERIMENT: LONG TRAINING HYBRID (CNN + BiLSTM + Attention)")
    print("="*50)
    
    t_config_hybrid = TrainConfig()
    t_config_hybrid.model_type = "hybrid"
    t_config_hybrid.experiment_log_path = "experiment_log_hybrid_long.csv"
    t_config_hybrid.save_model_path = "best_model_hybrid.pth"
    t_config_hybrid.save_plot_path = "training_curves_hybrid_long.png"
    
    t_config_hybrid.epochs = 150
    t_config_hybrid.early_stopping_patience = 20
    t_config_hybrid.learning_rate = 0.0001
    
    train_model(t_config_hybrid)
    
    b_config_hybrid = BacktestConfig()
    b_config_hybrid.model_type = "hybrid"
    b_config_hybrid.model_path = "best_model_hybrid.pth"
    run_backtest(b_config_hybrid)

if __name__ == '__main__':
    main()
