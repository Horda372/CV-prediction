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
    """
    Custom PyTorch Dataset that loads candlestick images based on the metadata CSV.
    """
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
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Load and auto-detect model architecture from weights
    if os.path.exists(TrainConfig.save_model_path):
        state_dict = torch.load(TrainConfig.save_model_path, map_location=device)
        print(f"Loaded model weights from {TrainConfig.save_model_path}")
        
        # Check if weights contain ResNet keys
        is_resnet = any(k.startswith("fc.") and not k.startswith("fc1.") and not k.startswith("fc2.") for k in state_dict.keys()) or "layer1.0.conv1.weight" in state_dict.keys()
        
        if is_resnet:
            print("Auto-detected Architecture: ResNet-18")
            weights = ResNet18_Weights.DEFAULT
            model = resnet18(weights=weights)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4),
                nn.Linear(num_ftrs, TrainConfig.num_classes)
            )
        else:
            print("Auto-detected Architecture: CustomCandlestickCNN")
            model = CustomCandlestickCNN(num_classes=TrainConfig.num_classes, dropout_prob=0.3)
            
        model.load_state_dict(state_dict)
    else:
        print("Error: Saved model weights not found.")
        return
        
    model = model.to(device)
    model.eval()
    
    all_probs = []
    all_labels = []
    all_pct_changes = []
    
    with torch.no_grad():
        for inputs, labels, pct_changes in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.extend(probs[:, 1].cpu().tolist())
            all_labels.extend(labels.tolist())
            all_pct_changes.extend(pct_changes.tolist())
            
    print("\nDecision Threshold Tuning Results on Validation Split:")
    print("-" * 90)
    print(f"{'Threshold':<12} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Net Return (%)':<15}")
    print("-" * 90)
    
    best_return = -9999.0
    best_th = 0.5
    best_metrics = {}
    cost = 0.05  # Transaction cost 0.05%
    
    for th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        th_preds = [1 if p >= th else 0 for p in all_probs]
        
        tp = sum((p == 1 and l == 1) for p, l in zip(th_preds, all_labels))
        fp = sum((p == 1 and l == 0) for p, l in zip(th_preds, all_labels))
        tn = sum((p == 0 and l == 0) for p, l in zip(th_preds, all_labels))
        fn = sum((p == 0 and l == 1) for p, l in zip(th_preds, all_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(all_labels)
        
        # Calculate Strategy Net Return
        th_return = sum((change - cost) for pred, change in zip(th_preds, all_pct_changes) if pred == 1)
        
        print(f"{th:<12.1f} | {accuracy*100:<9.2f}% | {precision*100:<9.2f}% | {recall*100:<9.2f}% | {f1:<10.4f} | {th_return:<+14.2f}%")
        
        if th_return > best_return:
            best_return = th_return
            best_th = th
            best_metrics = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "return": th_return,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn
            }
            
    print("-" * 90)
    
    if best_return > -9999.0:
        print(f"\nOptimal Decision Threshold (Max Profit): {best_th:.1f}")
        print(f"Accuracy:  {best_metrics['accuracy']*100:.2f}%")
        print(f"Precision: {best_metrics['precision']*100:.2f}% (Baseline: {sum(all_labels)/len(all_labels)*100:.2f}%)")
        print(f"Recall:    {best_metrics['recall']*100:.2f}%")
        print(f"F1-Score:  {best_metrics['f1']:.4f}")
        print(f"Strategy Net Return: {best_metrics['return']:+.2f}%")
        
        print("\nConfusion Matrix for Optimal Threshold:")
        print(f"               Predicted Class 0 | Predicted Class 1")
        print(f"Actual Class 0        {best_metrics['tn']:<10} |        {best_metrics['fp']:<10}")
        print(f"Actual Class 1        {best_metrics['fn']:<10} |        {best_metrics['tp']:<10}")
    else:
        print("\nNo threshold produced valid trading signals.")

if __name__ == "__main__":
    evaluate()
