import os
import csv
import pandas as pd
import mplfinance as mpf
from typing import List
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class CSVConfig:
    # Directory where input CSVs are stored
    csv_dir: str = "market_data"
    
    # List of symbols to process (corresponds to {symbol}_1d.csv)
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURGBP", "EURJPY", "GBPJPY"])
    
    # Window size (number of candles in a single generated image)
    window_size: int = 20
    
    # Timeframe suffix (just for metadata logging)
    timeframe: str = "1d"
    
    # Name of the main output directory
    output_dir: str = "dataset_market_cv"
    
    # Whether to normalize data to 0-1 range (Min-Max scaling)
    normalize_data: bool = True
    
    # List of periods for moving averages (will be drawn on images)
    moving_averages: List[int] = field(default_factory=lambda: [10, 20])
    
    # Forecast horizon (how many periods ahead we check to assign a label)
    forecast_horizon: int = 5
    
    # Volatility multiplier (number of standard deviations above standard window volatility)
    target_volatility_multiplier: float = 1.5

# ==========================================
# MAIN CSV PROCESSING LOGIC
# ==========================================
def generate_dataset_from_csv(config: CSVConfig):
    """
    Reads historical price data from local CSVs, generates scaled candlestick chart images,
    and creates a metadata file with future-pointing binary growth labels.
    """
    # Create directory structure
    images_dir = os.path.join(config.output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    metadata_file = os.path.join(config.output_dir, "metadata.csv")
    
    # Initialize metadata file with headers (Overwrites old mock files)
    print(f"Initializing dataset directory at: {os.path.abspath(config.output_dir)}")
    with open(metadata_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "symbol", "timeframe", "window_size", 
            "start_date", "end_date", "start_close_price", "end_close_price",
            "target_label", "future_close_price", "pct_change"
        ])

    # Chart style configuration (clean candles, no axes, identical to import.py)
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='', figcolor='black', facecolor='black')

    # Keep track of class counts
    class_0_count = 0
    class_1_count = 0

    for symbol in config.symbols:
        csv_filename = f"{symbol}_{config.timeframe}.csv"
        csv_filepath = os.path.join(config.csv_dir, csv_filename)
        
        if not os.path.exists(csv_filepath):
            print(f"Warning: CSV file not found at {csv_filepath}. Skipping symbol {symbol}.")
            continue
            
        print(f"\nProcessing: {symbol} from CSV...")
        try:
            # Read CSV data
            df = pd.read_csv(csv_filepath)
            
            # Identify the date column
            date_col = None
            for col in df.columns:
                if col.lower() in ["date", "time", "timestamp"]:
                    date_col = col
                    break
                    
            if date_col is None:
                # If first column has no header but contains dates
                df.rename(columns={df.columns[0]: "Date"}, inplace=True)
                date_col = "Date"
                
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df.sort_index(ascending=True, inplace=True)
            
            # Standardize column casing
            column_mapping = {col: col.capitalize() for col in df.columns if col.lower() in ["open", "high", "low", "close", "volume"]}
            df.rename(columns=column_mapping, inplace=True)
            
            # Ensure necessary columns are present
            required_cols = ["Open", "High", "Low", "Close"]
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Missing OHLC columns in {csv_filename}. Skipping.")
                continue

            # Calculate Technical Indicators
            if config.moving_averages:
                for ma in config.moving_averages:
                    df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()

            # Trim data to eliminate initial rolling NaN rows
            max_ma = max(config.moving_averages) if config.moving_averages else 0
            df = df.iloc[max_ma:].copy()
            
            total_rows = len(df)
            # Available samples considering sliding window and future forecast horizon
            available_samples = total_rows - config.window_size - config.forecast_horizon + 1
            
            if available_samples <= 0:
                print(f"Warning: Insufficient data points in {csv_filename} (Got {total_rows}, needed at least {config.window_size + config.forecast_horizon}). Skipping.")
                continue

            print(f"Generating {available_samples} sliding window candlestick charts...")
            
            generated_count = 0
            
            # Open file in append mode for each symbol
            with open(metadata_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for i in range(available_samples):
                    # Define sliding window indices
                    start_idx = i
                    end_idx = i + config.window_size
                    
                    window_data = df.iloc[start_idx:end_idx].copy()
                    
                    # --- CLASSIFICATION LOGIC (VOLATILITY-ADJUSTED TARGET LABEL) ---
                    current_close = window_data['Close'].iloc[-1]
                    future_idx = end_idx - 1 + config.forecast_horizon
                    future_close = df['Close'].iloc[future_idx]
                    
                    pct_change = ((future_close - current_close) / current_close) * 100.0 if current_close > 0 else 0
                    
                    # Calculate rolling daily standard deviation inside the 20-candle window
                    daily_returns = window_data['Close'].pct_change().dropna()
                    daily_std = daily_returns.std() * 100.0 if len(daily_returns) > 0 else 0.0
                    
                    # Scale standard deviation by sqrt of forecast horizon to get horizon-level volatility
                    import numpy as np
                    horizon_volatility = daily_std * np.sqrt(config.forecast_horizon)
                    
                    # Calculate dynamic threshold based on volatility multiplier
                    dynamic_threshold = config.target_volatility_multiplier * horizon_volatility
                    # Impose a logical minimum threshold of 0.15% to prevent noise from triggering buy signals
                    dynamic_threshold = max(dynamic_threshold, 0.15)
                    
                    # 1 = success (growth >= dynamic_threshold), 0 = no-buy
                    target_label = 1 if pct_change >= dynamic_threshold else 0

                    if target_label == 1:
                        class_1_count += 1
                    else:
                        class_0_count += 1

                    # Extract metadata values
                    start_date = window_data.index[0].strftime("%Y-%m-%d %H:%M:%S")
                    end_date = window_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                    start_close = round(window_data['Close'].iloc[0], 5)
                    end_close = round(current_close, 5)
                    
                    # --- MIN-MAX NORMALIZATION ---
                    if config.normalize_data:
                        min_val = window_data['Low'].min()
                        max_val = window_data['High'].max()
                        if max_val > min_val:
                            for col in ['Open', 'High', 'Low', 'Close']:
                                window_data[col] = (window_data[col] - min_val) / (max_val - min_val)
                            if config.moving_averages:
                                for ma in config.moving_averages:
                                    ma_col = f'SMA_{ma}'
                                    window_data[ma_col] = (window_data[ma_col] - min_val) / (max_val - min_val)

                    # --- TECHNICAL INDICATOR LINES ---
                    apds = []
                    if config.moving_averages:
                        for ma in config.moving_averages:
                            apds.append(mpf.make_addplot(window_data[f'SMA_{ma}'], type='line', width=1.5))
                    
                    # Unique filename
                    filename = f"{symbol}_{config.timeframe}_w{config.window_size}_{i:05d}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Base parameters matching original import.py exactly
                    plot_kwargs = dict(
                        type='candle', 
                        style=style, 
                        axisoff=True, 
                        savefig=dict(fname=filepath, dpi=100, bbox_inches='tight', pad_inches=0)
                    )
                    
                    if apds:
                        plot_kwargs['addplot'] = apds
                        
                    # Save image and close figure to prevent memory leakage
                    mpf.plot(window_data, **plot_kwargs)
                    plt.close('all')
                    
                    # Save metadata record
                    writer.writerow([
                        filename, symbol, config.timeframe, config.window_size, 
                        start_date, end_date, start_close, end_close,
                        target_label, round(future_close, 5), round(pct_change, 3)
                    ])
                    generated_count += 1
                    
                    # Print progress update every 100 images
                    if generated_count % 200 == 0:
                        print(f"  Processed {generated_count}/{available_samples} images...")
                        
            print(f"Successfully generated {generated_count} images for {symbol}.")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            import traceback
            traceback.print_exc()

    total_samples = class_0_count + class_1_count
    print(f"\nDataset generation completed successfully.")
    print(f"Total samples: {total_samples}")
    if total_samples > 0:
        print(f"  - Class 0 (No-Buy): {class_0_count} ({class_0_count/total_samples*100:.2f}%)")
        print(f"  - Class 1 (Buy):    {class_1_count} ({class_1_count/total_samples*100:.2f}%)")

if __name__ == "__main__":
    app_config = CSVConfig()
    generate_dataset_from_csv(app_config)
