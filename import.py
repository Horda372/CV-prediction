import yfinance as yf
import mplfinance as mpf
import pandas as pd
import os
import csv
from typing import List
from dataclasses import dataclass, field

# ==========================================
# CONFIGURATION 
# ==========================================
@dataclass
class Config:
    # List of stock symbols to fetch
    symbols: List[str] = field(default_factory=lambda: ["AAPL", "TSLA", "BTC-USD"])
    
    # Window size (number of candles in a single generated image)
    window_size: int = 20
    
    # Number of images to generate for each symbol
    num_images_per_symbol: int = 5
    
    # Timeframe (e.g., '1d' for days, '1h' for hours)
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
    target_threshold_pct: float = 5.0

# ==========================================
# MAIN SCRIPT LOGIC
# ==========================================
def generate_dataset_with_metadata(config: Config):
    """
    Fetches market data, generates candlestick chart images, and creates a metadata file
    based on the provided configuration.
    """
    
    # Create directory structure
    images_dir = os.path.join(config.output_dir, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    metadata_file = os.path.join(config.output_dir, "metadata.csv")
    
    # Initialize metadata file with headers (if it doesn't exist)
    file_exists = os.path.isfile(metadata_file)
    with open(metadata_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "filename", "symbol", "timeframe", "window_size", 
                "start_date", "end_date", "start_close_price", "end_close_price",
                "target_label", "future_close_price", "pct_change"
            ])

    # Chart style configuration (clean candles, no axes)
    mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
    style = mpf.make_mpf_style(marketcolors=mc, gridstyle='', figcolor='black', facecolor='black')

    # Calculate required historical data range (including indicators and horizon)
    max_ma = max(config.moving_averages) if config.moving_averages else 0
    required_rows = (config.num_images_per_symbol - 1) + config.window_size + config.forecast_horizon
    fetch_buffer = max_ma + required_rows
    
    # Determine fetch period for yfinance based on timeframe
    fetch_period = "2y" if config.timeframe == "1d" else "60d" 

    for symbol in config.symbols:
        print(f"\nProcessing: {symbol} (Timeframe: {config.timeframe})...")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=fetch_period, interval=config.timeframe)
            
            if df.empty or len(df) < fetch_buffer:
                print(f"Warning: Insufficient data for {symbol}. Skipping.")
                continue
                
            # Calculate technical indicators before trimming data
            if config.moving_averages:
                for ma in config.moving_averages:
                    df[f'SMA_{ma}'] = df['Close'].rolling(window=ma).mean()

            # Isolate only required data considering the future labels dimension
            df = df.tail(required_rows)
            
            generated_count = 0
            
            # Open file in append mode for each symbol
            with open(metadata_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                for i in range(config.num_images_per_symbol):
                    # Define sliding window
                    start_idx = i
                    end_idx = i + config.window_size
                    # Use .copy() so normalization doesn't affect original df
                    window_data = df.iloc[start_idx:end_idx].copy() 
                    
                    # Safeguard against indexing errors
                    if len(window_data) != config.window_size:
                        break
                        
                    # --- CLASSIFICATION LOGIC (TARGET LABEL) ---
                    current_close = window_data['Close'].iloc[-1]
                    future_idx = end_idx - 1 + config.forecast_horizon
                    
                    # Safeguard against out-of-bounds future data
                    if future_idx >= len(df):
                        break
                        
                    future_close = df['Close'].iloc[future_idx]
                    pct_change = ((future_close - current_close) / current_close) * 100.0 if current_close > 0 else 0
                    
                    # 1 = "Good" (Growth above threshold), 0 = "Weak/Drop" (Skip)
                    target_label = 1 if pct_change >= config.target_threshold_pct else 0

                    # Extract metadata for verification
                    start_date = window_data.index[0].strftime("%Y-%m-%d %H:%M:%S")
                    end_date = window_data.index[-1].strftime("%Y-%m-%d %H:%M:%S")
                    start_close = round(window_data['Close'].iloc[0], 4)
                    end_close = round(current_close, 4)
                    
                    # --- NORMALIZATION (Min-Max 0-1) ---
                    if config.normalize_data:
                        min_val = window_data[['Low']].min().min()
                        max_val = window_data[['High']].max().max()
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
                    
                    # Generate unique filename
                    filename = f"{symbol}_{config.timeframe}_w{config.window_size}_{i:05d}.png"
                    filepath = os.path.join(images_dir, filename)
                    
                    # Configure base parameters for generation
                    plot_kwargs = dict(
                        type='candle', 
                        style=style, 
                        axisoff=True, 
                        savefig=dict(fname=filepath, dpi=100, bbox_inches='tight', pad_inches=0)
                    )
                    
                    # Conditionally add extra layers (addplots)
                    if apds:
                        plot_kwargs['addplot'] = apds
                        
                    # Generate and save image
                    mpf.plot(window_data, **plot_kwargs)
                    
                    # Save extended metadata to CSV
                    writer.writerow([
                        filename, symbol, config.timeframe, config.window_size, 
                        start_date, end_date, start_close, end_close,
                        target_label, round(future_close, 4), round(pct_change, 2)
                    ])
                    generated_count += 1
                    
            print(f"Successfully generated {generated_count} images for {symbol}.")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

if __name__ == "__main__":
    # Initialize configuration (default values defined in the class above)
    app_config = Config()
    
    # Run the main process
    generate_dataset_with_metadata(app_config)
    
    print(f"\nProcess completed. Check the '{app_config.output_dir}' folder.")