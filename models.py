import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ==========================================
# 1. VISION-ONLY MODEL (BASELINE CNN)
# ==========================================
class CustomCandlestickCNN(nn.Module):
    """
    Custom regularized 4-layer CNN designed specifically for geometric financial charts.
    Uses BatchNorm and Dropout to force generalization and prevent pixel memorization.
    """
    def __init__(self, num_classes=2, dropout_prob=0.3, extract_features=False):
        super(CustomCandlestickCNN, self).__init__()
        self.extract_features = extract_features
        
        # Conv block 1: Input (3, 224, 224) -> Output (16, 112, 112)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 2: Input (16, 112, 112) -> Output (32, 56, 56)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 3: Input (32, 56, 56) -> Output (64, 28, 28)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv block 4: Input (64, 28, 28) -> Output (128, 14, 14)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()
        
        # 128 channels * 14 * 14 feature map dimension = 25088 input features
        self.feature_dim = 128 * 14 * 14
        
        if not self.extract_features:
            self.fc1 = nn.Linear(self.feature_dim, 64)
            self.dropout = nn.Dropout(p=dropout_prob)
            self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        if self.extract_features:
            return x
            
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. TABULAR-ONLY MODEL (BASELINE LSTM)
# ==========================================
class LSTMBaseline(nn.Module):
    """
    LSTM model processing the raw 20-day numerical time series.
    Input shape should be (batch_size, sequence_length, input_dim).
    """
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, num_classes=2, dropout_prob=0.3, extract_features=False):
        super(LSTMBaseline, self).__init__()
        self.extract_features = extract_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer expects input of shape (batch, seq, feature) when batch_first=True
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        if not self.extract_features:
            self.dropout = nn.Dropout(p=dropout_prob)
            self.fc = nn.Linear(hidden_dim, num_classes)
            
    def forward(self, x):
        # x shape: (batch_size, seq_len=20, input_dim=10)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # Extract the output of the last time step
        last_step_out = out[:, -1, :]
        
        if self.extract_features:
            return last_step_out
            
        last_step_out = self.dropout(last_step_out)
        out = self.fc(last_step_out)
        return out

# ==========================================
# 3. MULTI-MODAL MODEL (HYBRID CNN + LSTM)
# ==========================================
class Hybrid_CNN_LSTM(nn.Module):
    """
    Combines Vision features (CNN) and Time-Series features (LSTM) using Late Fusion.
    """
    def __init__(self, lstm_input_dim=10, num_classes=2, dropout_prob=0.4):
        super(Hybrid_CNN_LSTM, self).__init__()
        
        # Instantiate sub-models as feature extractors
        self.cnn = CustomCandlestickCNN(extract_features=True)
        self.lstm = LSTMBaseline(input_dim=lstm_input_dim, hidden_dim=64, num_layers=2, extract_features=True)
        
        cnn_feature_dim = self.cnn.feature_dim # 25088
        lstm_feature_dim = self.lstm.hidden_dim # 64
        
        combined_dim = cnn_feature_dim + lstm_feature_dim
        
        # Fusion Head
        self.fc1 = nn.Linear(combined_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, image_x, tabular_x):
        # 1. Vision stream
        cnn_features = self.cnn(image_x)
        
        # 2. Time-series stream
        lstm_features = self.lstm(tabular_x)
        
        # 3. Concatenate (Late Fusion)
        combined = torch.cat((cnn_features, lstm_features), dim=1)
        
        # 4. Final Classification Head
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
