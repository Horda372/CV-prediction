from train_hybrid import train_model
from config import TrainConfig

if __name__ == "__main__":
    config = TrainConfig()
    config.model_type = "cnn"
    config.save_model_path = "best_model_cnn.pth"
    config.experiment_log_path = "experiment_log_cnn.csv"
    config.save_plot_path = "training_curves_cnn.png"
    config.epochs = 20
    config.early_stopping_patience = 5
    train_model(config)
