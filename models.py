import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ==========================================
# 1. VISION-ONLY MODEL (BASELINE CNN)
# ==========================================
class CustomCandlestickCNN(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.3, extract_features=False):
        super(CustomCandlestickCNN, self).__init__()
        self.extract_features = extract_features
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.relu = nn.ReLU()
        
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
        
        x = x.view(x.size(0), -1)
        
        if self.extract_features:
            return x
            
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==========================================
# 2. ADVANCED TABULAR-ONLY MODEL (BiLSTM + Attention)
# ==========================================
class LSTMBaseline(nn.Module):
    """
    Advanced Bidirectional LSTM with Attention Mechanism and Layer Normalization.
    """
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=3, num_classes=2, dropout_prob=0.4, extract_features=False):
        super(LSTMBaseline, self).__init__()
        self.extract_features = extract_features
        self.hidden_dim = hidden_dim
        
        # Bidirectional LSTM to understand context from both past and future relative points
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, 
                            dropout=dropout_prob, bidirectional=True)
        
        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # We need this to expose the output dim for the Hybrid model
        self.output_feature_dim = hidden_dim * 2
        
        if not self.extract_features:
            self.dropout = nn.Dropout(p=dropout_prob)
            self.fc1 = nn.Linear(self.output_feature_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, num_classes)
            
    def forward(self, x):
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = self.attention(lstm_out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Context vector (weighted sum of sequence elements)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_dim * 2)
        
        context_vector = self.layer_norm(context_vector)
        
        if self.extract_features:
            return context_vector
            
        out = self.dropout(context_vector)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# ==========================================
# 3. ADVANCED MULTI-MODAL MODEL (HYBRID CNN + ADVANCED LSTM)
# ==========================================
class Hybrid_CNN_LSTM(nn.Module):
    """
    Combines Vision features (CNN) and Advanced Time-Series features (BiLSTM+Attn).
    """
    def __init__(self, lstm_input_dim=11, num_classes=2, dropout_prob=0.4):
        super(Hybrid_CNN_LSTM, self).__init__()
        
        self.cnn = CustomCandlestickCNN(extract_features=True)
        self.lstm = LSTMBaseline(input_dim=lstm_input_dim, extract_features=True)
        
        cnn_feature_dim = self.cnn.feature_dim # 25088
        lstm_feature_dim = self.lstm.output_feature_dim # 256
        
        # FIX: BOTTLENECK LAYER FOR CNN TO PREVENT GRADIENT DROWNING
        self.cnn_bottleneck = nn.Sequential(
            nn.Linear(cnn_feature_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256)
        )
        
        combined_dim = 256 + lstm_feature_dim # 256 + 256 = 512
        
        self.fc1 = nn.Linear(combined_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, image_x, tabular_x):
        cnn_features = self.cnn(image_x)
        cnn_compressed = self.cnn_bottleneck(cnn_features)
        
        lstm_features = self.lstm(tabular_x)
        
        # Equal concatenation (256 vs 256)
        combined = torch.cat((cnn_compressed, lstm_features), dim=1)
        
        x = self.fc1(combined)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out
