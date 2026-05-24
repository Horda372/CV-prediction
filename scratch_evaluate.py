import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

class TrainConfig:
    csv_file = "dataset_market_cv/metadata.csv"
    img_dir = "dataset_market_cv/images"
    batch_size = 32
    train_split_pct = 0.8
    num_classes = 2
    save_model_path = "best_model.pth"

class CustomCandlestickCNN(nn.Module):
    """
    Custom regularized 4-layer CNN designed specifically for geometric financial charts.
    Uses BatchNorm and Dropout to force generalization and prevent pixel memorization.
    """
    def __init__(self, num_classes=2, dropout_prob=0.3):
        super(CustomCandlestickCNN, self).__init__()
        
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
        
        # Fully connected layers
        # 128 channels * 14 * 14 feature map dimension = 25088 input features
        self.fc1 = nn.Linear(128 * 14 * 14, 64)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

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
        image = Image.open(img_path).convert("RGB")
        label = int(row["target_label"])

        if self.transform:
            image = self.transform(image)

        return image, label

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(TrainConfig.csv_file)
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values(by="end_date").reset_index(drop=True)
    
    split_idx = int(len(df) * TrainConfig.train_split_pct)
    val_df = df.iloc[split_idx:]
    
    print(f"Validation set size: {len(val_df)}")
    print(f"  - Actual Class 0 (No-Buy): {sum(val_df['target_label'] == 0)}")
    print(f"  - Actual Class 1 (Buy):    {sum(val_df['target_label'] == 1)}")
    
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = CandlestickDataset(val_df, TrainConfig.img_dir, transform=data_transforms)
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.batch_size, shuffle=False)
    
    # Load model
    model = CustomCandlestickCNN(num_classes=TrainConfig.num_classes, dropout_prob=0.3)
    
    if os.path.exists(TrainConfig.save_model_path):
        model.load_state_dict(torch.load(TrainConfig.save_model_path, map_location=device))
        print(f"Loaded model weights from {TrainConfig.save_model_path}")
    else:
        print("Error: Saved model weights not found.")
        return
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            
    # Confusion Matrix
    tp = sum((p == 1 and l == 1) for p, l in zip(all_preds, all_labels))
    fp = sum((p == 1 and l == 0) for p, l in zip(all_preds, all_labels))
    tn = sum((p == 0 and l == 0) for p, l in zip(all_preds, all_labels))
    fn = sum((p == 0 and l == 1) for p, l in zip(all_preds, all_labels))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(all_labels)
    
    print("\nEvaluation Results on Validation Split:")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(f"               Predicted Class 0 | Predicted Class 1")
    print(f"Actual Class 0        {tn:<10} |        {fp:<10}")
    print(f"Actual Class 1        {fn:<10} |        {tp:<10}")

if __name__ == "__main__":
    evaluate()
