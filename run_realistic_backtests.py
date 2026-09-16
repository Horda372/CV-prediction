import os
from backtester import BacktestConfig, run_backtest

def main():
    print("="*60)
    print("REALISTIC BACKTEST: CNN (VISION ONLY)")
    print("="*60)
    b_cnn = BacktestConfig()
    b_cnn.model_type = "cnn"
    b_cnn.model_path = "best_model_cnn.pth"
    run_backtest(b_cnn)

    print("\n" + "="*60)
    print("REALISTIC BACKTEST: LSTM (TABULAR ONLY)")
    print("="*60)
    b_lstm = BacktestConfig()
    b_lstm.model_type = "lstm"
    b_lstm.model_path = "best_model_lstm.pth"
    run_backtest(b_lstm)

    print("\n" + "="*60)
    print("REALISTIC BACKTEST: HYBRID (CNN + LSTM)")
    print("="*60)
    b_hybrid = BacktestConfig()
    b_hybrid.model_type = "hybrid"
    b_hybrid.model_path = "best_model_hybrid.pth"
    run_backtest(b_hybrid)

if __name__ == '__main__':
    main()
