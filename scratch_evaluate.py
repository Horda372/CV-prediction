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
    model = resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, TrainConfig.num_classes)
    
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
