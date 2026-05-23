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
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    
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
    
    # Minimum percentage growth required to assign label "1" (success/buy)
    # 1.0% as selected in the /grill-me session
    target_threshold_pct: float = 1.0

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

            # Calculate rolling 14-day Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
            low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
            df['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()

            # Trim data to eliminate initial rolling NaN rows (SMAs and 14-day ATR)
            max_ma = max(config.moving_averages) if config.moving_averages else 0
            trim_start = max(max_ma, 15)  # 15 to account for 14-day rolling window + 1 shift row
            df = df.iloc[trim_start:].copy()
            
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
                    
                    # --- TRIPLE BARRIER CLASSIFICATION LOGIC ---
                    current_close = window_data['Close'].iloc[-1]
                    current_atr = window_data['ATR'].iloc[-1]
                    
                    barrier_upper = current_close + 2.0 * current_atr
                    barrier_lower = current_close - 1.0 * current_atr
                    
                    future_block = df.iloc[end_idx : end_idx + config.forecast_horizon]
                    
                    target_label = 0
                    future_close = current_close
                    pct_change = 0.0
                    barrier_touched = False
                    
                    for idx_future, future_row in future_block.iterrows():
                        f_high = future_row['High']
                        f_low = future_row['Low']
                        f_close = future_row['Close']
                        
                        # 1. Check Stop-Loss first (Conservative Risk Management)
                        if f_low <= barrier_lower:
                            target_label = 0
                            future_close = barrier_lower
                            pct_change = ((barrier_lower - current_close) / current_close) * 100.0 if current_close > 0 else 0
                            barrier_touched = True
                            break
                            
                        # 2. Check Profit-Take
                        if f_high >= barrier_upper:
                            target_label = 1
                            future_close = barrier_upper
                            pct_change = ((barrier_upper - current_close) / current_close) * 100.0 if current_close > 0 else 0
                            barrier_touched = True
                            break
                            
                    if not barrier_touched:
                        # Timed out (Vertical barrier) -> Class 0 (No-Buy)
                        target_label = 0
                        future_close = future_block['Close'].iloc[-1] if len(future_block) > 0 else current_close
                        pct_change = ((future_close - current_close) / current_close) * 100.0 if current_close > 0 else 0

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
