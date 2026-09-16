import os
from train_hybrid import TrainConfig, train_model
from backtester import BacktestConfig, run_backtest

def main():
    # We already ran Hybrid. Let's run CNN.
    print("\n" + "="*50)
    print("EXPERIMENT 1: VISION ONLY (CNN)")
    print("="*50)

    t_config_cnn = TrainConfig()
    t_config_cnn.model_type = "cnn"
    t_config_cnn.experiment_log_path = "experiment_log_cnn.csv"
    t_config_cnn.save_model_path = "best_model_cnn.pth"
    t_config_cnn.save_plot_path = "training_curves_cnn.png"
    t_config_cnn.epochs = 10  # Full 10 epochs for fair comparison
    train_model(t_config_cnn)

    b_config_cnn = BacktestConfig()
    b_config_cnn.model_type = "cnn"
    b_config_cnn.model_path = "best_model_cnn.pth"
    run_backtest(b_config_cnn)


    print("\n" + "="*50)
    print("EXPERIMENT 2: TABULAR ONLY (LSTM)")
    print("="*50)

    t_config_lstm = TrainConfig()
    t_config_lstm.model_type = "lstm"
    t_config_lstm.experiment_log_path = "experiment_log_lstm.csv"
    t_config_lstm.save_model_path = "best_model_lstm.pth"
    t_config_lstm.save_plot_path = "training_curves_lstm.png"
    t_config_lstm.epochs = 10
    train_model(t_config_lstm)

    b_config_lstm = BacktestConfig()
    b_config_lstm.model_type = "lstm"
    b_config_lstm.model_path = "best_model_lstm.pth"
    run_backtest(b_config_lstm)

if __name__ == '__main__':
    main()
