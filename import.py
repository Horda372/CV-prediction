import MetaTrader5 as mt5
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
    # List of symbols to fetch (Make sure they are available in your MT5 broker)
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY"])
    
    # Window size (number of candles in a single generated image)
    window_size: int = 20
    
    # Number of images to generate for each symbol
    num_images_per_symbol: int = 5
    
    # Timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d', '1w')
    timeframe: str = "1h"
    
    # Name of the main output directory
    output_dir: str = "dataset_market_cv"
    
    # Whether to normalize data to 0-1 range (Min-Max scaling)
    normalize_data: bool = True
    
    # List of periods for moving averages (will be drawn on images)
    moving_averages: List[int] = field(default_factory=lambda: [10, 20])
    
    # Forecast horizon (how many periods ahead we check to assign a label)
    forecast_horizon: int = 5
    
    # Minimum percentage growth required to assign label "1" (success/buy)
    # Lowered default since Forex/intraday moves in smaller percentages
    target_threshold_pct: float = 0.2

# Timeframe mapping for MT5
TIMEFRAME_MAPPING = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
    "1d": mt5.TIMEFRAME_D1,
    "1w": mt5.TIMEFRAME_W1,
}

# ==========================================
# MAIN SCRIPT LOGIC
# ==========================================
def generate_dataset_with_metadata(config: Config):
    """
    Fetches market data via MetaTrader5, generates candlestick chart images, 
    and creates a metadata file based on the provided configuration.
    """
    if config.timeframe not in TIMEFRAME_MAPPING:
        print(f"Error: Unsupported timeframe '{config.timeframe}'. Use one of {list(TIMEFRAME_MAPPING.keys())}")
        return

    mt5_timeframe = TIMEFRAME_MAPPING[config.timeframe]

    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"MT5 initialize() failed, error code = {mt5.last_error()}")
        return
        
    print(f"MetaTrader5 initialized successfully. Terminal version: {mt5.version()}")
    
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

    for symbol in config.symbols:
        print(f"\nProcessing: {symbol} (Timeframe: {config.timeframe})...")
        try:
            # Ensure the symbol is visible in Market Watch
            if not mt5.symbol_select(symbol, True):
                print(f"Warning: Failed to select symbol {symbol} in MT5. It may not exist in your broker. Skipping.")
                continue

            # Fetch historical data directly from MetaTrader 5
            # We fetch 'fetch_buffer' number of candles from the current time backwards
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, fetch_buffer)
            
            if rates is None or len(rates) < fetch_buffer:
                print(f"Warning: Insufficient data for {symbol} (Got {len(rates) if rates is not None else 0}, needed {fetch_buffer}). Skipping.")
                continue
                
            # Convert to Pandas DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'tick_volume': 'Volume'}, inplace=True)
            
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
                    start_close = round(window_data['Close'].iloc[0], 5)
                    end_close = round(current_close, 5)
                    
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
                        target_label, round(future_close, 5), round(pct_change, 3)
                    ])
                    generated_count += 1
                    
            print(f"Successfully generated {generated_count} images for {symbol}.")
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    # Shutdown MT5 connection after loop finishes
    mt5.shutdown()

if __name__ == "__main__":
    # Initialize configuration (default values defined in the class above)
    app_config = Config()
    
    # Run the main process
    generate_dataset_with_metadata(app_config)
    
    print(f"\nProcess completed. Check the '{app_config.output_dir}' folder.")