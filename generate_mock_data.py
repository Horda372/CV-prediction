import os
import csv
import numpy as np
from PIL import Image
from datetime import datetime, timedelta

def create_mock_dataset():
    output_dir = "dataset_market_cv"
    images_dir = os.path.join(output_dir, "images")
    metadata_file = os.path.join(output_dir, "metadata.csv")

    # Create directories if they do not exist
    os.makedirs(images_dir, exist_ok=True)
    print(f"Creating mock dataset in: {os.path.abspath(output_dir)}")

    # Schema columns
    headers = [
        "filename", "symbol", "timeframe", "window_size", 
        "start_date", "end_date", "start_close_price", "end_close_price",
        "target_label", "future_close_price", "pct_change"
    ]

    symbols = ["EURUSD", "GBPUSD", "USDJPY"]
    timeframe = "1h"
    window_size = 20
    num_samples = 15

    base_time = datetime(2026, 5, 1, 0, 0, 0)
    
    metadata_rows = []

    for i in range(num_samples):
        symbol = symbols[i % len(symbols)]
        filename = f"{symbol}_{timeframe}_w{window_size}_{i:05d}.png"
        filepath = os.path.join(images_dir, filename)

        # Create a mock image (e.g., solid color or simple colored shape representing a candle)
        # Background is black (matching mpf style facecolor='black' in import.py)
        img_array = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add simple visual features so it has some lines/shapes
        # Draw a vertical line (candle body/wick)
        wick_x = 112
        img_array[40:180, wick_x, :] = [0, 255, 0] if i % 2 == 0 else [255, 0, 0] # green or red
        # Draw a body rectangle
        img_array[80:140, wick_x-10:wick_x+10, :] = [0, 200, 0] if i % 2 == 0 else [200, 0, 0]

        img = Image.fromarray(img_array)
        img.save(filepath)

        # Dates separated by 1 hour increments
        start_date = (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        end_date = (base_time + timedelta(hours=i + window_size)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Prices
        start_close = 1.0500 + (i * 0.001)
        end_close = start_close + (0.002 if i % 2 == 0 else -0.002)
        future_close = end_close + (0.005 if i % 3 == 0 else -0.001)
        pct_change = round(((future_close - end_close) / end_close) * 100.0, 3)
        
        # 1 = positive growth above some threshold, 0 = weak/negative
        target_label = 1 if pct_change >= 0.2 else 0

        metadata_rows.append([
            filename, symbol, timeframe, window_size,
            start_date, end_date, round(start_close, 5), round(end_close, 5),
            target_label, round(future_close, 5), pct_change
        ])

    # Write metadata.csv
    with open(metadata_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(metadata_rows)

    print(f"Generated {num_samples} mock images and metadata.csv successfully.")

if __name__ == "__main__":
    create_mock_dataset()
