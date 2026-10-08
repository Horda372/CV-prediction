import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal_loss = -alpha_t * ((1.0 - pt) ** self.gamma) * logpt
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

class CandlestickDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")

        label = int(row["target_label"])
        pct_change = float(row["pct_change"])

        if self.transform:
            image = self.transform(image)

        return image, label, pct_change

class HybridCandlestickDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tensor_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tensor_dir = tensor_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            raise FileNotFoundError(f"Error loading image {img_path}: {e}")

        tensor_name = row.get("tensor_filename", img_name.replace(".png", ".npy"))
        tensor_path = os.path.join(self.tensor_dir, tensor_name)
        try:
            tabular_data = np.load(tensor_path)
            tabular_tensor = torch.tensor(tabular_data, dtype=torch.float32)
        except Exception as e:
            raise FileNotFoundError(f"Error loading tensor {tensor_path}: {e}")

        label = int(row["target_label"])
        pct_change = float(row["pct_change"])

        return image, tabular_tensor, label, pct_change
