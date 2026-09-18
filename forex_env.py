import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import os

from models import Hybrid_CNN_LSTM

class ForexHybridEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, img_dir: str, tensor_dir: str, hybrid_model_path: str):
        super(ForexHybridEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tensor_dir = tensor_dir
        # Action space: 
        # 0 = Hold/Wait
        # 1-10 = Buy (and risk 10%, 20%, 30% ... 100% of capital)
        # 11 = Close Position
        self.action_space = spaces.Discrete(12)
        
        # State space: 256-dimensional vector from Hybrid FC1 + 1 dim for current position status
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(257,), dtype=np.float32)
        
        # Load the frozen Hybrid Model as Feature Extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine input dim from a sample
        sample_tensor_path = os.path.join(self.tensor_dir, self.df.iloc[0]["tensor_filename"])
        input_dim = np.load(sample_tensor_path).shape[1]
        
        self.feature_extractor = Hybrid_CNN_LSTM(lstm_input_dim=input_dim, num_classes=2)
        self.feature_extractor.load_state_dict(torch.load(hybrid_model_path, map_location=self.device))
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval() # Freeze
        
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Trading state
        self.in_position = False
        self.entry_price = 0.0
        self.capital = 10000.0
        self.initial_capital = 10000.0
        self.current_risk_fraction = 0.0

    def _get_obs(self):
        # Extract features for current step
        row = self.df.iloc[self.current_step]
        img_name = row["filename"]
        tensor_name = row["tensor_filename"]
        
        img_path = os.path.join(self.img_dir, img_name)
        tensor_path = os.path.join(self.tensor_dir, tensor_name)
        
        image = Image.open(img_path).convert("RGB")
        image = self.data_transforms(image).unsqueeze(0).to(self.device)
        
        tabular_data = np.load(tensor_path)
        tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Pass through backbones
            cnn_out = self.feature_extractor.cnn(image)
            lstm_out = self.feature_extractor.lstm(tabular_tensor)
            
            # Pass through bottleneck
            cnn_bottleneck = self.feature_extractor.cnn_bottleneck(cnn_out)
            
            # Concat
            combined = torch.cat((cnn_bottleneck, lstm_out), dim=1)
            
            # Pass through FC1 to get 128-dim feature vector
            features = self.feature_extractor.fc1(combined)
            features = self.feature_extractor.bn1(features)
            features = torch.relu(features) # [1, 128]
            
        features_np = features.cpu().numpy()[0]
        
        # Append position status (0 or 1)
        pos_status = np.array([1.0 if self.in_position else 0.0], dtype=np.float32)
        obs = np.concatenate((features_np, pos_status))
        
        return obs

    def step(self, action):
        reward = 0.0
        row = self.df.iloc[self.current_step]
        pct_change = float(row["pct_change"]) / 100.0 # From subsequent days in original dataset
        
        # Execute Action
        if action >= 1 and action <= 10 and not self.in_position: # BUY
            self.in_position = True
            self.current_risk_fraction = float(action) * 0.10 # 10% to 100%
            reward = -0.0005 # Spread penalty
            
        elif action == 11 and self.in_position: # CLOSE
            self.in_position = False
            trade_capital = self.capital * self.current_risk_fraction
            trade_profit = trade_capital * pct_change
            self.capital += trade_profit
            # Reward proportional to risk taken and pct change
            reward = pct_change * self.current_risk_fraction 
            
        elif action == 0 and self.in_position: # HOLD
            reward = -0.0001
            
        elif action == 0 and not self.in_position: # WAIT
            reward = 0.0
        
        # Penalize invalid actions (trying to close when not in position, etc) to speed up learning
        elif action >= 1 and action <= 10 and self.in_position:
            reward = -0.001 
        elif action == 11 and not self.in_position:
            reward = -0.001
            
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        if self.capital <= 0:
            done = True
            reward = -10.0 # Bankruptcy penalty
            
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.in_position = False
        self.capital = self.initial_capital
        self.current_risk_fraction = 0.0
        return self._get_obs(), {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Capital: {self.capital:.2f}, In Position: {self.in_position}")
